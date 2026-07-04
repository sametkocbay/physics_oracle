"""§5.7 — Run the steady-state incompressible solver and monitor convergence.

In OpenFOAM 13 simpleFoam is superseded by foamRun -solver incompressibleFluid.
Sources OF 13's bashrc, runs the solver, captures the log, and parses
residuals + force coefficients for downstream extraction (§6.4) and QC (§7).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

from physics_oracle.core.logging import setup_logging

LOG = setup_logging()

OPENFOAM_BASHRC = os.environ.get("OPENFOAM_BASHRC", "/opt/openfoam13/etc/bashrc")


def run_potential_init(of_case_dir: Path, log_path: Path | None = None,
                       timeout: int = 10 * 60) -> int:
    """Best-effort potential-flow initialisation of U (and p) before the steady
    solver.

    Cold-starting incompressibleFluid from a uniform field at high Re / high AoA
    overshoots on the first pressure solve and diverges (linear solver blows up
    -> SIGFPE). A divergence-free potential-flow start avoids that. This is
    NON-FATAL: on any failure we log and let foamRun cold-start exactly as
    before, so adding it can only help. Requires the ``Phi`` solver and
    ``potentialFlow`` block in fvSolution (see configs/openfoam.yaml).
    """
    log_path = log_path or (of_case_dir / "potentialFoam.log")
    cmd = (
        f"source {OPENFOAM_BASHRC} && "
        f"cd {of_case_dir.resolve()} && potentialFoam -writep"
    )
    try:
        with log_path.open("w") as f:
            proc = subprocess.run(
                ["bash", "-c", cmd],
                stdout=f, stderr=subprocess.STDOUT,
                timeout=timeout, check=False,
            )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        LOG.warning("[%s] potentialFoam init timed out — cold-starting", of_case_dir.name)
        return 1
    if rc != 0:
        LOG.warning("[%s] potentialFoam init exit %d — cold-starting foamRun",
                    of_case_dir.name, rc)
    else:
        LOG.info("[%s] potentialFoam init OK", of_case_dir.name)
    return rc


def run_simple_foam(of_case_dir: Path, log_path: Path | None = None,
                    timeout: int = 60 * 60 * 6) -> int:
    """Run foamRun -solver incompressibleFluid in `of_case_dir`. Returns the exit code.

    OpenFOAM 13 replaced simpleFoam with foamRun -solver incompressibleFluid.
    OF 13 needs its bashrc sourced; we wrap the call in a plain bash -c (not -lc)
    so that cluster login scripts do not reset PATH/LD_LIBRARY_PATH after sourcing.
    The log is written to simpleFoam.log for compatibility with the log parser.

    A best-effort potentialFoam pre-step seeds a divergence-free field so the
    steady solver does not blow up on a cold start (see run_potential_init).
    """
    run_potential_init(of_case_dir)
    log_path = log_path or (of_case_dir / "simpleFoam.log")
    cmd = (
        f"set -e && "
        f"source {OPENFOAM_BASHRC} && "
        f"cd {of_case_dir.resolve()} && foamRun -solver incompressibleFluid"
    )
    LOG.info("[%s] running foamRun (timeout %ds) — log %s",
             of_case_dir.name, timeout, log_path.name)
    with log_path.open("w") as f:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            stdout=f, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
    LOG.info("[%s] foamRun exit %d", of_case_dir.name, proc.returncode)
    return proc.returncode


def run_residual_postprocess(of_case_dir: Path, log_path: Path | None = None,
                             timeout: int = 60 * 30) -> int:
    """Evaluate the ``residualFields`` coded function object on the ``0/`` ML
    fields without running the solver.  Returns the exit code.

    The case must have been written with
    ``setup_case_with_initial_fields(enable_residual_capture=True)`` so that
    controlDict carries the coded FO.  ``foamPostProcess -solver
    incompressibleFluid -time 0`` constructs the incompressible solver in
    post-processing mode (loading U/p/k/omega/nut), executes the controlDict
    functions, and the FO writes the per-cell residual volScalarFields into
    ``0/``.  No SIMPLE iteration runs, so the residual is measured exactly at
    the ML prediction.  The log is written to ``residuals.log``.
    """
    log_path = log_path or (of_case_dir / "residuals.log")
    cmd = (
        f"set -e && "
        f"source {OPENFOAM_BASHRC} && "
        f"cd {of_case_dir.resolve()} && "
        f"foamPostProcess -solver incompressibleFluid -time 0"
    )
    LOG.info("[%s] running foamPostProcess (timeout %ds) — log %s",
             of_case_dir.name, timeout, log_path.name)
    with log_path.open("w") as f:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            stdout=f, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
    LOG.info("[%s] foamPostProcess exit %d", of_case_dir.name, proc.returncode)
    return proc.returncode


def run_geometry_export(of_case_dir: Path, log_path: Path | None = None,
                        timeout: int = 60 * 30) -> int:
    """Run the `meshGeometry` coded function object via `foamPostProcess`.

    Identical mechanism to `run_residual_postprocess` — `foamPostProcess`
    constructs the mesh and executes the controlDict functions on the `0/`
    fields with no SIMPLE iteration.  The geometry FO
    (`openfoam_setup/_geometry_export.py`) writes Sf/V/weights/... into `0/`.
    The log is written to `geometry.log`.
    """
    log_path = log_path or (of_case_dir / "geometry.log")
    cmd = (
        f"set -e && "
        f"source {OPENFOAM_BASHRC} && "
        f"cd {of_case_dir.resolve()} && "
        f"foamPostProcess -solver incompressibleFluid -time 0"
    )
    LOG.info("[%s] running foamPostProcess for mesh geometry (timeout %ds)",
             of_case_dir.name, timeout)
    with log_path.open("w") as f:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            stdout=f, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
    LOG.info("[%s] foamPostProcess (geometry) exit %d", of_case_dir.name, proc.returncode)
    return proc.returncode


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# OF residuals look like:
# smoothSolver:  Solving for Ux, Initial residual = 1.5e-3, Final residual = 2e-5, ...
# DICPCG:  Solving for p, Initial residual = ...
RES_RE = re.compile(
    r"Solving for (?P<field>\w+),\s+Initial residual\s*=\s*(?P<res>[\d.eE+-]+)"
)
# OF13 prints "Time = 1s" (with units suffix); also tolerate plain numbers.
TIME_RE = re.compile(r"^Time\s*=\s*[\d.eE+-]+s?\s*$", re.MULTILINE)


def parse_solver_log(log_text: str) -> dict:
    """Return per-iteration residual history and final iteration count."""
    fields = ["Ux", "Uy", "p", "k", "omega"]
    history: dict[str, list[float]] = {f: [] for f in fields}

    # Split log by "Time = N" blocks; inside each block collect first residual
    # for each field (multiple non-orthogonal correctors may report several).
    blocks = TIME_RE.split(log_text)
    # blocks[0] is preamble; subsequent are per-iteration
    for block in blocks[1:]:
        seen: set[str] = set()
        for m in RES_RE.finditer(block):
            field = m.group("field")
            if field in history and field not in seen:
                history[field].append(float(m.group("res")))
                seen.add(field)
        # Pad missing fields with NaN so columns stay aligned
        for f in fields:
            if f not in seen:
                history[f].append(float("nan"))

    n_iter = len(history["Ux"])
    final = {
        f: (history[f][-1] if history[f] and not _isnan(history[f][-1]) else None)
        for f in fields
    }
    return {
        "n_iter": n_iter,
        "history": history,
        "final": final,
        "fields": fields,
    }


def _isnan(x: float) -> bool:
    return x != x


# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------

def detect_convergence(history: dict[str, list[float]]) -> dict:
    """Apply §5.7 criterion: residuals dropped ≥ 4 orders of magnitude."""
    drops: dict[str, float] = {}
    converged_fields: dict[str, bool] = {}
    for field, vals in history.items():
        clean = [v for v in vals if not _isnan(v) and v > 0]
        if len(clean) < 2:
            drops[field] = 0.0
            converged_fields[field] = False
            continue
        ratio = clean[0] / clean[-1]
        import math
        drop = math.log10(ratio) if ratio > 0 else 0.0
        drops[field] = drop
        converged_fields[field] = drop >= 4.0
    overall = all(converged_fields.values()) if converged_fields else False
    return {"drops": drops, "fields": converged_fields, "overall": overall}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run simpleFoam in an OpenFOAM case dir.")
    p.add_argument("of_case", type=Path)
    p.add_argument("--timeout", type=int, default=6 * 3600)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rc = run_simple_foam(args.of_case, timeout=args.timeout)
    log = (args.of_case / "simpleFoam.log").read_text(errors="replace")
    parsed = parse_solver_log(log)
    LOG.info("Iterations: %d, final residuals: %s", parsed["n_iter"], parsed["final"])
    conv = detect_convergence(parsed["history"])
    LOG.info("Drops: %s   Converged: %s", conv["drops"], conv["overall"])
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
