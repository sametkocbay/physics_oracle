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
    """Potential-flow initialisation of U (and p).  NOT part of the standard
    solver path — kept for diagnostics/manual experiments only.

    Empirically (see scripts/diagnose_robustness.py, July 2026): seeding the
    steady RANS solve with the potential field *hurts* convergence.  The
    potential flow carries zero circulation (no Kutta condition), so the
    solver must shed a starting vortex that drifts along the wake-cut sliver
    cells and leaves the p residual limit-cycling around 1e-2 instead of
    converging; a plain uniform cold start converges fine.  Cold-start
    robustness at high Re / high AoA is provided by the two-stage startup
    continuation instead (see ``_run_two_stage``).

    ``-initialiseUBCs`` is REQUIRED: without it potentialFoam zeroes U and the
    boundary flux (see its createFields.H), solves a garbage potential problem,
    and *writes that garbage into 0/U before* the ``-writep`` step — poisoning
    the subsequent foamRun even though this wrapper reports "non-fatal".
    ``-writep`` additionally needs a ``div(div(phi,U))`` entry in divSchemes.

    To keep the best-effort promise honest, 0/U and 0/p are snapshotted first
    and restored whenever potentialFoam exits non-zero.  Requires the ``Phi``
    solver and ``potentialFlow`` block in fvSolution (see configs/openfoam.yaml).
    """
    log_path = log_path or (of_case_dir / "potentialFoam.log")
    zero = of_case_dir / "0"
    backups = {}
    for name in ("U", "p"):
        f = zero / name
        if f.exists():
            backups[name] = f.read_bytes()

    cmd = (
        f"source {OPENFOAM_BASHRC} && "
        f"cd {of_case_dir.resolve()} && potentialFoam -initialiseUBCs -writep"
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
        rc = 1
    if rc != 0:
        # potentialFoam writes U/phi before the -writep step, so a failure can
        # leave a partial/garbage init behind — restore the pristine fields.
        for name, data in backups.items():
            (zero / name).write_bytes(data)
        LOG.warning("[%s] potentialFoam init exit %d — restored 0/{U,p}, "
                    "cold-starting foamRun", of_case_dir.name, rc)
    else:
        LOG.info("[%s] potentialFoam init OK", of_case_dir.name)
    return rc


def _foam_run_once(of_case_dir: Path, log_path: Path, timeout: int,
                   append: bool = False) -> int:
    """One foamRun invocation; log truncated or appended.  Returns rc
    (1 on timeout, matching the best-effort convention)."""
    cmd = (
        f"set -e && "
        f"source {OPENFOAM_BASHRC} && "
        f"cd {of_case_dir.resolve()} && foamRun -solver incompressibleFluid"
    )
    try:
        with log_path.open("a" if append else "w") as f:
            proc = subprocess.run(
                ["bash", "-c", cmd],
                stdout=f, stderr=subprocess.STDOUT,
                timeout=timeout, check=False,
            )
        return proc.returncode
    except subprocess.TimeoutExpired:
        LOG.warning("[%s] foamRun timed out after %ds", of_case_dir.name, timeout)
        return 1


def needs_startup_stage(spec) -> bool:
    """True when the case lies outside the trained envelope and should run
    the two-stage startup continuation (configs/openfoam.yaml `startup:`)."""
    from physics_oracle.openfoam_setup.case_setup import startup_config
    gate = startup_config().get("gate", {})
    if not gate:
        return False
    return (abs(spec.aoa_deg) > float(gate.get("aoa_abs_deg_over", 5.0))
            or spec.Re > float(gate.get("re_over", 5.0e5)))


def _count_iters(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    return len(TIME_RE.findall(log_path.read_text(errors="replace")))


def _run_two_stage(of_case_dir: Path, spec, log_path: Path,
                   timeout: int, end_time: int) -> int:
    """Stage A (first-order, heavy relaxation, limitMag/bound constraints)
    then stage B (production schemes, startFrom latestTime).  Both stages
    append to the same simpleFoam.log so parse_solver_log sees one
    continuous history.  Writes run_info.yaml next to the log."""
    import time as _time

    import yaml as _yaml

    from physics_oracle.openfoam_setup.case_setup import (
        startup_config, write_stage_system)

    stage_a_end = int((startup_config().get("stage_a", {})
                       or {}).get("iterations", 400))
    info: dict = {"two_stage": True, "stage_a_end": stage_a_end}

    write_stage_system(of_case_dir, spec, "A", stage_a_end, end_time)

    t0 = _time.time()
    budget_a = min(max(timeout // 4, 60), 3600)
    LOG.info("[%s] stage A: foamRun (first-order, endTime=%d, timeout %ds)",
             of_case_dir.name, stage_a_end, budget_a)
    rc_a = _foam_run_once(of_case_dir, log_path, budget_a, append=False)
    n_a = _count_iters(log_path)
    info["stage_a"] = {"rc": rc_a, "n_iter": n_a,
                       "wall_s": round(_time.time() - t0, 1)}
    if rc_a != 0:
        LOG.warning("[%s] stage A exited rc=%d after %d iters — stage B "
                    "restarts from the latest written time (0 if none)",
                    of_case_dir.name, rc_a, n_a)

    write_stage_system(of_case_dir, spec, "B", stage_a_end, end_time)
    t1 = _time.time()
    budget_b = max(int(timeout - (t1 - t0)), 60)
    LOG.info("[%s] stage B: foamRun (production schemes, endTime=%d, timeout %ds)",
             of_case_dir.name, end_time, budget_b)
    rc_b = _foam_run_once(of_case_dir, log_path, budget_b, append=True)
    info["stage_b"] = {"rc": rc_b, "n_iter": _count_iters(log_path) - n_a,
                       "wall_s": round(_time.time() - t1, 1)}

    (of_case_dir / "run_info.yaml").write_text(
        _yaml.safe_dump(info, sort_keys=False))
    LOG.info("[%s] two-stage run done: stage A rc=%d (%d it), stage B rc=%d (%d it)",
             of_case_dir.name, rc_a, n_a, rc_b, info["stage_b"]["n_iter"])
    return rc_b


def run_simple_foam(of_case_dir: Path, log_path: Path | None = None,
                    timeout: int = 60 * 60 * 6, spec=None,
                    end_time: int = 5000) -> int:
    """Run foamRun -solver incompressibleFluid in `of_case_dir`. Returns the exit code.

    OpenFOAM 13 replaced simpleFoam with foamRun -solver incompressibleFluid.
    OF 13 needs its bashrc sourced; we wrap the call in a plain bash -c (not -lc)
    so that cluster login scripts do not reset PATH/LD_LIBRARY_PATH after sourcing.
    The log is written to simpleFoam.log for compatibility with the log parser.

    The solve cold-starts from the uniform 0/ fields (or the ML-provided
    fields for warm-started cases — which is why no potentialFoam pre-step
    runs here; it would overwrite them, and it demonstrably hurts
    convergence, see run_potential_init).

    When `spec` is given and the case lies outside the trained envelope
    (see `needs_startup_stage`), the run is split into a robust first-order
    stage A and a production stage B (`_run_two_stage`).  With `spec=None`
    the behavior is exactly the single-stage path used everywhere else.
    """
    log_path = log_path or (of_case_dir / "simpleFoam.log")
    if spec is not None and needs_startup_stage(spec):
        return _run_two_stage(of_case_dir, spec, log_path, timeout, end_time)

    LOG.info("[%s] running foamRun (timeout %ds) — log %s",
             of_case_dir.name, timeout, log_path.name)
    rc = _foam_run_once(of_case_dir, log_path, timeout, append=False)
    LOG.info("[%s] foamRun exit %d", of_case_dir.name, rc)
    return rc


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

def detect_convergence(history: dict[str, list[float]],
                       ref_window: int = 10, tail_window: int = 50) -> dict:
    """Apply §5.7 criterion: residuals dropped ≥ 4 orders of magnitude.

    The reference residual is the *maximum* over the first ``ref_window``
    iterations rather than the very first value.  A good initial guess
    (two-stage continuation) lowers the iteration-1 residual, which would
    make a drop measured from it spuriously strict — while residualControl
    exits on *absolute* thresholds.  Using the startup maximum is
    monotone-safe: it can only be >= the first value, so every case accepted
    under the old criterion is still accepted.

    The final residual is the *median* over the last ``tail_window``
    iterations rather than the single last value: for a converged run the
    two are identical, but a mild residual limit cycle would otherwise make
    acceptance depend on which phase of the cycle iteration N landed on.
    """
    import math
    import statistics
    drops: dict[str, float] = {}
    converged_fields: dict[str, bool] = {}
    for field, vals in history.items():
        clean = [v for v in vals if not _isnan(v) and v > 0]
        if len(clean) < 2:
            drops[field] = 0.0
            converged_fields[field] = False
            continue
        ref = max(clean[:max(1, ref_window)])
        tail = statistics.median(clean[-max(1, tail_window):])
        ratio = ref / tail
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
