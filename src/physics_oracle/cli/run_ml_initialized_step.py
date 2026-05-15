"""Run an OpenFOAM step warm-started from an ML prediction.

Inputs:
  --prediction   path to a predictions.pt produced by physics-control-loop's
                 src/inference/inference.py (keys: pos, node_feat, target, meta)
  --mesh         path to a saved mesh.h5 (cell_centers, points, connectivity,
                 boundary_markers) — the per-case mesh from the dataset
  --work-dir     scratch directory; the OF case is written under
                 <work-dir>/of_case/

Modes (mutually exclusive; default --full-run):
  --only-residual    endTime=1 + solverInfo writeResidualFields=yes
                     → writes per-cell initialResidual:{U,p,k,omega} to 1/
  --full-run         endTime=500 (with existing residualControl early-stop)
                     → standard warm-start refinement, no per-cell residuals

Optional overrides (default: read from predictions.pt meta dict):
  --naca, --aoa-deg, --re
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree

from physics_oracle.core.case_spec import CaseSpec, compute_inlet_conditions
from physics_oracle.core.logging import setup_logging
from physics_oracle.openfoam_setup.case_setup import setup_case_with_initial_fields
from physics_oracle.openfoam_setup.extract import _read_text, latest_time_dir, parse_internal_field
from physics_oracle.openfoam_setup.mesh_h5_to_polymesh import write_polymesh_from_h5
from physics_oracle.openfoam_setup.runner import detect_convergence, parse_solver_log, run_simple_foam

LOG = setup_logging()


# Channel order in predictions.pt (from src/surrogate/gno/infer.py:27-28)
TARGET_COLS = ["u", "v", "w", "p", "k", "omega", "nut", "phi"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    pos: np.ndarray            # (N, 2)
    target: np.ndarray         # (N, 8) -> columns = TARGET_COLS
    meta: dict


def load_prediction(path: Path) -> Prediction:
    data = torch.load(path, weights_only=False, map_location="cpu")
    pos = data["pos"].cpu().numpy().astype(np.float64)
    target = data["target"].cpu().numpy().astype(np.float64)
    if pos.shape[1] >= 2:
        pos = pos[:, :2]
    if target.shape[1] != len(TARGET_COLS):
        raise ValueError(
            f"Expected target shape (N, {len(TARGET_COLS)}) with columns "
            f"{TARGET_COLS}, got {target.shape}."
        )
    meta = dict(data.get("meta", {}))
    return Prediction(pos=pos, target=target, meta=meta)


@dataclass
class Mesh:
    cell_centers: np.ndarray   # (Nc, 2)
    points: np.ndarray         # (Np, 2)
    connectivity: np.ndarray   # (Nc, 4)
    boundary_markers: np.ndarray  # (Nc,)


def load_mesh(path: Path) -> Mesh:
    with h5py.File(path, "r") as h:
        return Mesh(
            cell_centers=np.asarray(h["cell_centers"][:], dtype=np.float64),
            points=np.asarray(h["points"][:], dtype=np.float64),
            connectivity=np.asarray(h["connectivity"][:], dtype=np.int64),
            boundary_markers=np.asarray(h["boundary_markers"][:], dtype=np.int8),
        )


# ---------------------------------------------------------------------------
# Cell matching
# ---------------------------------------------------------------------------

def match_cells(pos: np.ndarray, cell_centers: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """KDTree-match prediction points to mesh cells.  Returns (N,) int64
    indices into `cell_centers`.  Raises if any point has no neighbour within
    `tol`."""
    tree = cKDTree(cell_centers)
    dist, idx = tree.query(pos, distance_upper_bound=tol)
    n_missing = int(np.sum(np.isinf(dist)))
    if n_missing > 0:
        max_d = float(dist[~np.isinf(dist)].max()) if np.any(~np.isinf(dist)) else float("inf")
        raise ValueError(
            f"{n_missing}/{len(pos)} prediction points have no matching mesh "
            f"cell within tol={tol}.  Max matched distance: {max_d:.3e}.  "
            f"The prediction file and the mesh file must come from the same case."
        )
    return idx.astype(np.int64)


# ---------------------------------------------------------------------------
# Build full-mesh initial fields
# ---------------------------------------------------------------------------

def build_full_fields(
    target: np.ndarray,
    indices: np.ndarray,
    n_cells: int,
    inlet,
    dtype=np.float32,
) -> dict[str, np.ndarray]:
    """Start every cell at the freestream / inlet values, then overwrite
    at `indices` with the prediction columns."""
    U = np.zeros((n_cells, 3), dtype=dtype)
    U[:, 0] = inlet.U_x
    U[:, 1] = inlet.U_y
    p = np.zeros(n_cells, dtype=dtype)
    k = np.full(n_cells, inlet.k_inlet, dtype=dtype)
    omega = np.full(n_cells, inlet.omega_inlet, dtype=dtype)
    nut = np.zeros(n_cells, dtype=dtype)

    U[indices, 0] = target[:, 0]                       # u
    U[indices, 1] = target[:, 1]                       # v
    # target[:, 2] = w  — discarded in 2D-by-extrusion
    p[indices] = target[:, 3]
    k[indices] = target[:, 4]
    omega[indices] = target[:, 5]
    nut[indices] = target[:, 6]
    # target[:, 7] = phi  — OF recomputes phi from U

    # Clamp k / omega to be strictly positive (turbulence-model sanity).
    k = np.clip(k, 1.0e-12, None)
    omega = np.clip(omega, 1.0e-9, None)
    return {"U": U, "p": p, "k": k, "omega": omega, "nut": nut}


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _summary(arr: np.ndarray, name: str) -> str:
    a = arr.ravel()
    return (f"{name:>20s}: |r|_max={np.abs(a).max():.4e}  "
            f"|r|_mean={np.abs(a).mean():.4e}  "
            f"|r|_L2={np.sqrt((a * a).sum()):.4e}")


def parse_residuals(of_case: Path) -> dict | None:
    """Return the global initial residuals captured by OF13's `residuals` fo
    as a ``{field: value}`` dict, or None if they could not be read.

    OF13 Foundation does not support per-cell residual fields; what we get
    instead is one scalar per field per iteration, written to
    ``postProcessing/residuals/0/residuals.dat``.  For a 1-iteration
    --only-residual run that's a single row: the initial residual of each
    field evaluated at the ML-prediction field — i.e., how far the ML
    prediction is from satisfying the discretised PDEs in the global L1/L2
    sense the solver uses.
    """
    pp_dir = of_case / "postProcessing" / "residuals"
    if not pp_dir.exists():
        LOG.warning("No postProcessing/residuals/ directory — residuals fo did not run")
        return None
    time_subdirs = sorted(pp_dir.iterdir(), key=lambda p: p.name)
    if not time_subdirs:
        LOG.warning("postProcessing/residuals/ is empty")
        return None
    dat_path = time_subdirs[-1] / "residuals.dat"
    if not dat_path.exists():
        LOG.warning("Missing %s", dat_path)
        return None

    text = dat_path.read_text()
    # Standard OF residuals.dat format:
    #   # Residuals
    #   # Time      Ux       Uy       p        k        omega
    #   1           1.5e-3   1e-3     8e-5     3e-6     2e-6
    header_cols: list[str] = []
    rows: list[list[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            header_cols = line.lstrip("#").split()
            continue
        parts = line.split()
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    if not rows:
        LOG.warning("No residual rows parsed from %s", dat_path)
        return None
    last = rows[-1]
    if header_cols and len(header_cols) >= len(last):
        names = header_cols[: len(last)]
    else:
        names = [f"col{i}" for i in range(len(last))]
    return {name: val for name, val in zip(names, last)}


def report_residuals(residuals: dict | None) -> None:
    """Pretty-print a residual dict from parse_residuals."""
    if residuals is None:
        print("No residuals captured.")
        return
    print("Initial residuals at this iteration (global, scalar per field):")
    for name, val in residuals.items():
        print(f"  {name:>10s}: {val:.4e}")
    print()
    print("NOTE: OF13 Foundation does not emit per-cell residual volScalarFields.")
    print("      Per-cell spatial residuals require custom postprocessing or ESI OF.")


def parse_convergence(of_case: Path) -> dict:
    """Parse the solver log into a convergence-summary dict."""
    log_path = of_case / "simpleFoam.log"
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    parsed = parse_solver_log(text)
    conv = detect_convergence(parsed["history"])
    return {
        "n_iter": parsed["n_iter"],
        "final_residuals": parsed["final"],
        "orders_dropped": conv["drops"],
        "converged": conv["overall"],
    }


def report_convergence(convergence: dict) -> None:
    """Pretty-print a convergence dict from parse_convergence."""
    print(f"Iterations: {convergence['n_iter']}")
    print(f"Final residuals: {convergence['final_residuals']}")
    print(f"Orders dropped: {convergence['orders_dropped']}")
    print(f"Converged (all fields ≥ 4 orders): {convergence['converged']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prediction", required=True, type=Path)
    p.add_argument("--mesh", required=True, type=Path)
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--naca", default=None,
                   help="4-digit NACA code (default: meta['naca_code'])")
    p.add_argument("--aoa-deg", type=float, default=None,
                   help="Angle of attack in degrees (default: meta['aoa_deg'])")
    p.add_argument("--re", type=float, default=None,
                   help="Reynolds number (default: meta['reynolds'])")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--only-residual", action="store_true",
                      help="Run 1 SIMPLE iter and capture spatial residuals")
    mode.add_argument("--full-run", action="store_true",
                      help="Run 500 SIMPLE iters with standard convergence criteria (default)")
    p.add_argument("--clean", action="store_true",
                   help="Delete --work-dir before starting")
    return p.parse_args()


@dataclass
class StepResult:
    """Result of one OpenFOAM warm-start step."""
    of_case_dir: Path
    mode: str                       # "only-residual" | "full-run"
    exit_code: int
    residuals: dict | None          # {field: scalar}  — only-residual mode
    convergence: dict | None        # parsed log info  — full-run mode


def run_step(
    prediction: Path,
    mesh_h5: Path,
    work_dir: Path,
    *,
    mode: str = "full-run",
    naca: str | None = None,
    aoa_deg: float | None = None,
    re: float | None = None,
    clean: bool = False,
) -> StepResult:
    """Run one OpenFOAM step warm-started from an ML prediction.

    Loads `prediction` (a predictions.pt), matches it to the mesh in
    `mesh_h5`, writes an OpenFOAM case under `work_dir/of_case/` with the
    prediction as the initial field (freestream outside the prediction's
    bounding box), runs foamRun, and returns a `StepResult`.

    Parameters
    ----------
    mode
        ``"only-residual"`` — endTime=1, returns global initial residuals.
        ``"full-run"``       — endTime=500, returns convergence info.
    naca, aoa_deg, re
        Override the values read from the prediction's `meta` dict.
    clean
        Delete `work_dir` before starting.
    """
    if mode not in ("only-residual", "full-run"):
        raise ValueError(f"mode must be 'only-residual' or 'full-run', got {mode!r}")
    prediction, mesh_h5, work_dir = Path(prediction), Path(mesh_h5), Path(work_dir)

    if clean and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # ----- load prediction & resolve metadata -----
    LOG.info("Loading prediction %s", prediction)
    pred = load_prediction(prediction)
    naca = naca or str(pred.meta.get("naca_code", "")).strip()
    aoa = aoa_deg if aoa_deg is not None else float(pred.meta.get("aoa_deg", "nan"))
    re = re if re is not None else float(pred.meta.get("reynolds",
                                                       pred.meta.get("Re", "nan")))
    if not naca or len(naca) != 4 or not naca.isdigit():
        raise ValueError(f"NACA code missing/invalid (got {naca!r}); pass naca=")
    if not np.isfinite(aoa):
        raise ValueError("AoA missing; pass aoa_deg=")
    if not np.isfinite(re):
        raise ValueError("Re missing; pass re=")
    LOG.info("case: NACA%s  aoa=%.2f deg  Re=%.3g", naca, aoa, re)

    spec = CaseSpec.build(naca, aoa, re, "ood_probe")
    inlet = compute_inlet_conditions(aoa, re)
    LOG.info("inlet: U_mag=%.4g  U=(%.4g, %.4g)  k=%.4g  omega=%.4g",
             inlet.U_mag, inlet.U_x, inlet.U_y, inlet.k_inlet, inlet.omega_inlet)

    # ----- load mesh & match -----
    LOG.info("Loading mesh %s", mesh_h5)
    mesh = load_mesh(mesh_h5)
    LOG.info("Mesh: %d cells, %d points", len(mesh.cell_centers), len(mesh.points))

    indices = match_cells(pred.pos, mesh.cell_centers)
    LOG.info("Matched %d prediction points to mesh cells (full mesh has %d)",
             len(indices), len(mesh.cell_centers))

    # ----- build initial fields, freestream outside the bbox -----
    fields = build_full_fields(pred.target, indices, len(mesh.cell_centers), inlet)
    LOG.info("Built full-mesh init: U range x=(%.3g, %.3g) y=(%.3g, %.3g), p range (%.3g, %.3g)",
             fields["U"][:, 0].min(), fields["U"][:, 0].max(),
             fields["U"][:, 1].min(), fields["U"][:, 1].max(),
             fields["p"].min(), fields["p"].max())

    # ----- write polyMesh -----
    of_case = work_dir / "of_case"
    polymesh_dir = of_case / "constant" / "polyMesh"
    LOG.info("Writing polyMesh -> %s", polymesh_dir)
    summary = write_polymesh_from_h5(mesh_h5, polymesh_dir)
    LOG.info("polyMesh: %s", summary)

    # ----- setup case + run -----
    only_residual = mode == "only-residual"
    end_time = 1 if only_residual else 500
    setup_case_with_initial_fields(
        of_case, spec, fields,
        end_time=end_time,
        enable_residual_capture=only_residual,
    )

    LOG.info("Running foamRun (endTime=%d, mode=%s)", end_time, mode)
    rc = run_simple_foam(of_case)
    LOG.info("foamRun exit code: %d", rc)

    residuals = parse_residuals(of_case) if only_residual else None
    convergence = parse_convergence(of_case) if not only_residual else None
    return StepResult(
        of_case_dir=of_case,
        mode=mode,
        exit_code=rc,
        residuals=residuals,
        convergence=convergence,
    )


def main() -> None:
    args = parse_args()
    mode = "only-residual" if args.only_residual else "full-run"
    try:
        result = run_step(
            prediction=args.prediction,
            mesh_h5=args.mesh,
            work_dir=args.work_dir,
            mode=mode,
            naca=args.naca,
            aoa_deg=args.aoa_deg,
            re=args.re,
            clean=args.clean,
        )
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))

    if result.mode == "only-residual":
        report_residuals(result.residuals)
    else:
        report_convergence(result.convergence)

    sys.exit(0 if result.exit_code == 0 else 1)


if __name__ == "__main__":
    main()
