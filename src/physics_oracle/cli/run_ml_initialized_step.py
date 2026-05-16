"""Run an OpenFOAM step warm-started from an ML prediction.

Inputs:
  --prediction   path to a predictions.pt produced by physics-control-loop's
                 src/inference/inference.py (keys: pos, node_feat, target, meta)
  --mesh         path to a saved mesh.h5 (cell_centers, points, connectivity,
                 boundary_markers) — the per-case mesh from the dataset
  --work-dir     scratch directory; the OF case is written under
                 <work-dir>/of_case/

Modes (mutually exclusive; default --full-run):
  --only-residual    Physics-consistency check.  Evaluate the per-cell residual
                     of every governing equation at the ML prediction.  A
                     ``residualFields`` coded function object (see
                     openfoam_setup/_residual_functions.py) is run via
                     foamPostProcess — no SIMPLE iteration — writing the
                     volScalarFields momentumResidual / continuityResidual /
                     kResidual / omegaResidual into the case's 0/ directory.
  --n-steps N        Solver-consistency probe.  Run N SIMPLE iterations
                     (1-5 recommended) warm-started from the prediction and
                     return OpenFOAM's own per-iteration residuals for
                     Ux/Uy/p/k/omega — so you can see whether convergence
                     starts or the solution blows up.
  --full-run         endTime=500 (with existing residualControl early-stop)
                     → standard warm-start refinement, no per-cell residuals

Residual output (--only-residual only):
  default            full per-cell residual field for each equation, returned as
                     numpy arrays and dumped to <work-dir>/residuals.npz
  --summary          reduce each field to scalar stats {max, mean, L2}

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
from physics_oracle.openfoam_setup._residual_functions import RESIDUAL_FIELDS
from physics_oracle.openfoam_setup.case_setup import setup_case_with_initial_fields
from physics_oracle.openfoam_setup.extract import _read_text, parse_internal_field
from physics_oracle.openfoam_setup.mesh_h5_to_polymesh import write_polymesh_from_h5
from physics_oracle.openfoam_setup.runner import (
    detect_convergence,
    parse_solver_log,
    run_residual_postprocess,
    run_simple_foam,
)

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

def _reduce(arr: np.ndarray) -> dict[str, float]:
    """Reduce a per-cell residual field to scalar summary statistics.

    `median` and `p99` are included alongside `mean`/`max` because the residual
    fields are heavy-tailed — the omega-equation residual in particular is large
    in the near-wall band (steep omega profile), so `mean`/`max` alone are
    misleading; `median` reflects the bulk of the field.
    """
    a = np.abs(np.asarray(arr, dtype=np.float64).ravel())
    return {
        "max": float(a.max()),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p99": float(np.percentile(a, 99)),
        "l2": float(np.sqrt((a * a).sum())),
    }


def parse_residuals(of_case: Path) -> dict[str, np.ndarray] | None:
    """Read the per-cell residual volScalarFields written by the
    ``residualFields`` coded function object (openfoam_setup/_residual_functions).

    Returns ``{field: (N,) ndarray}`` for momentumResidual / continuityResidual
    / kResidual / omegaResidual, or None if any field is missing.  The fields
    are evaluated at the ML prediction itself — foamPostProcess writes them
    into the case's ``0/`` directory with no SIMPLE iteration, so each value is
    the per-cell imbalance of that discretised governing equation.
    """
    zero = of_case / "0"
    out: dict[str, np.ndarray] = {}
    for name in RESIDUAL_FIELDS:
        try:
            arr = parse_internal_field(_read_text(zero / name))
        except (FileNotFoundError, ValueError) as exc:
            LOG.warning("Could not read residual field %s/%s: %s", zero, name, exc)
            return None
        out[name] = np.asarray(arr, dtype=np.float64).ravel()
    return out


def report_residuals(residuals: dict | None, mode: str) -> None:
    """Pretty-print residuals from `run_step`.

    `mode` is ``"spatial"`` (values are per-cell ndarrays) or ``"summary"``
    (values are ``{max, mean, l2}`` dicts).
    """
    if residuals is None:
        print("No residuals captured.")
        return
    if mode == "spatial":
        print("Per-cell residual fields (|r| summarised over cells):")
        for name, arr in residuals.items():
            s = _reduce(arr)
            print(f"  {name:>20s}: n={arr.size}  median={s['median']:.4e}  "
                  f"p99={s['p99']:.4e}  mean={s['mean']:.4e}  max={s['max']:.4e}")
        print("  (wall-function cells masked to 0; near-wall omegaResidual is "
              "large by nature — prefer median.)")
    else:
        print("Residual summary (per-cell field reduced to scalars):")
        for name, s in residuals.items():
            print(f"  {name:>20s}: median={s['median']:.4e}  p99={s['p99']:.4e}  "
                  f"mean={s['mean']:.4e}  max={s['max']:.4e}  L2={s['l2']:.4e}")


def parse_convergence(of_case: Path) -> dict:
    """Parse the solver log into a convergence-summary dict.

    ``history`` holds OpenFOAM's per-iteration initial residuals as
    ``{field: [r_1, r_2, ...]}`` for Ux/Uy/p/k/omega — the array used by the
    --n-steps probe to show whether convergence starts.
    """
    log_path = of_case / "simpleFoam.log"
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    parsed = parse_solver_log(text)
    conv = detect_convergence(parsed["history"])
    return {
        "n_iter": parsed["n_iter"],
        "fields": parsed["fields"],
        "history": parsed["history"],
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


def report_iterations(convergence: dict) -> None:
    """Pretty-print the per-iteration residual array from an --n-steps probe."""
    fields = convergence["fields"]
    history = convergence["history"]
    n = convergence["n_iter"]
    if n == 0:
        print("No iterations parsed from the solver log.")
        return
    print(f"Per-iteration initial residuals ({n} SIMPLE step(s)):")
    print("  iter  " + "  ".join(f"{f:>11s}" for f in fields))
    for i in range(n):
        row = "  ".join(f"{history[f][i]:11.4e}" for f in fields)
        print(f"  {i + 1:>4d}  {row}")

    # Verdict over the probe window.  Distinguish a genuine blow-up (huge or
    # NaN residuals) from residuals merely rising while still bounded — the
    # latter is often just a start-up transient and needs more steps to call.
    print()
    last = [history[f][-1] for f in fields]
    decreasing = []
    for f in fields:
        vals = [v for v in history[f] if v == v]  # drop NaN
        if len(vals) >= 2 and vals[0] > 0:
            decreasing.append(vals[-1] < vals[0])
    if any((v != v) or v > 1.0e3 for v in last):
        print("Verdict: residuals exploded (huge / NaN) — the prediction is "
              "outside the solver's stable basin.")
    elif decreasing and all(decreasing):
        print("Verdict: all residuals decreasing — convergence is starting.")
    elif decreasing and not any(decreasing):
        print("Verdict: residuals rising but still bounded — not yet "
              "converging; run more steps to tell a start-up transient from "
              "real divergence.")
    else:
        print("Verdict: mixed — inspect the per-field columns above.")


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
                      help="0-step: evaluate per-cell residual fields at the ML prediction")
    mode.add_argument("--n-steps", type=int, metavar="N", default=None,
                      help="N-step probe: run N SIMPLE iterations (1-5 recommended) "
                           "and report OpenFOAM's per-iteration residuals")
    mode.add_argument("--full-run", action="store_true",
                      help="Run 500 SIMPLE iters with standard convergence criteria (default)")
    p.add_argument("--summary", action="store_true",
                   help="--only-residual: reduce each residual field to scalar "
                        "{max,mean,L2} instead of returning the full per-cell field")
    p.add_argument("--clean", action="store_true",
                   help="Delete --work-dir before starting")
    return p.parse_args()


@dataclass
class StepResult:
    """Result of one OpenFOAM warm-start step."""
    of_case_dir: Path
    mode: str                       # "only-residual" | "n-step" | "full-run"
    exit_code: int
    residuals: dict | None          # only-residual mode (see residual_mode)
    convergence: dict | None        # parsed log info — n-step and full-run modes
    residual_mode: str | None = None  # "spatial" | "summary" — only-residual mode


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
    spatial_residuals: bool = True,
    n_steps: int = 1,
) -> StepResult:
    """Run one OpenFOAM step warm-started from an ML prediction.

    Loads `prediction` (a predictions.pt), matches it to the mesh in
    `mesh_h5`, writes an OpenFOAM case under `work_dir/of_case/` with the
    prediction as the initial field (freestream outside the prediction's
    bounding box), and returns a `StepResult`.

    Parameters
    ----------
    mode
        ``"only-residual"`` — evaluate per-cell residual fields at the ML
        prediction via foamPostProcess (no SIMPLE iteration).
        ``"n-step"``         — run `n_steps` SIMPLE iterations and return
        OpenFOAM's per-iteration residuals in ``StepResult.convergence``.
        ``"full-run"``       — endTime=500 SIMPLE run, returns convergence info.
    naca, aoa_deg, re
        Override the values read from the prediction's `meta` dict.
    clean
        Delete `work_dir` before starting.
    spatial_residuals
        Only-residual mode.  If True (default), ``StepResult.residuals`` holds
        the full per-cell residual field for each equation as ``(N,) ndarray``s
        (``residual_mode == "spatial"``).  If False, each field is reduced to a
        ``{max, mean, l2}`` dict (``residual_mode == "summary"``).
    n_steps
        n-step mode only.  Number of SIMPLE iterations to run (1-5 recommended).
    """
    if mode not in ("only-residual", "n-step", "full-run"):
        raise ValueError(
            f"mode must be 'only-residual', 'n-step' or 'full-run', got {mode!r}")
    if mode == "n-step" and n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
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
    if only_residual:
        end_time = 1
    elif mode == "n-step":
        end_time = n_steps
    else:
        end_time = 500
    setup_case_with_initial_fields(
        of_case, spec, fields,
        end_time=end_time,
        enable_residual_capture=only_residual,
    )

    residuals: dict | None = None
    convergence: dict | None = None
    residual_mode: str | None = None
    if only_residual:
        LOG.info("Evaluating per-cell residual fields via foamPostProcess")
        rc = run_residual_postprocess(of_case)
        LOG.info("foamPostProcess exit code: %d", rc)
        per_cell = parse_residuals(of_case)
        if per_cell is None or spatial_residuals:
            residuals, residual_mode = per_cell, "spatial"
        else:
            residuals = {n: _reduce(a) for n, a in per_cell.items()}
            residual_mode = "summary"
    else:
        LOG.info("Running foamRun (endTime=%d, mode=%s)", end_time, mode)
        rc = run_simple_foam(of_case)
        LOG.info("foamRun exit code: %d", rc)
        convergence = parse_convergence(of_case)

    return StepResult(
        of_case_dir=of_case,
        mode=mode,
        exit_code=rc,
        residuals=residuals,
        convergence=convergence,
        residual_mode=residual_mode,
    )


def main() -> None:
    args = parse_args()
    if args.only_residual:
        mode = "only-residual"
    elif args.n_steps is not None:
        mode = "n-step"
    else:
        mode = "full-run"
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
            spatial_residuals=not args.summary,
            n_steps=args.n_steps if args.n_steps is not None else 1,
        )
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))

    if result.mode == "only-residual":
        report_residuals(result.residuals, result.residual_mode or "spatial")
        if result.residual_mode == "spatial" and result.residuals is not None:
            npz_path = args.work_dir / "residuals.npz"
            np.savez(npz_path, **result.residuals)
            print()
            print(f"Per-cell residual arrays written to {npz_path}")
            print(f"OpenFOAM residual volScalarFields under {result.of_case_dir / '0'}")
    elif result.mode == "n-step":
        report_iterations(result.convergence)
        hist = result.convergence["history"]
        npz_path = args.work_dir / "iteration_residuals.npz"
        np.savez(npz_path, **{f: np.asarray(v, dtype=np.float64)
                              for f, v in hist.items()})
        print()
        print(f"Per-iteration residual arrays written to {npz_path}")
    else:
        report_convergence(result.convergence)

    sys.exit(0 if result.exit_code == 0 else 1)


if __name__ == "__main__":
    main()
