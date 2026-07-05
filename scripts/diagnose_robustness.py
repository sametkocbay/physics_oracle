"""Stage-0 robustness diagnostics for high-AoA / high-Re cases.

Runs a fixed matrix of representative cases through the existing pipeline
pieces (case setup, C-mesh, potentialFoam init, foamRun) with instrumentation
between the steps, then classifies each failure:

    pressure_blowup      p residual explodes / GAMG FATAL
    momentum_blowup      Ux/Uy residual explodes first
    turbulence_blowup    bounding-k/omega storm precedes death
    slow_no_converge     survives to endTime but drop < 4 orders
    converged            all fields dropped >= 4 orders

Writes one ``diagnosis.yaml`` per case plus ``diagnosis_summary.yaml`` in the
dataset root, and prints a summary table.

Usage:
    PHYSICS_ORACLE_DATASET_ROOT=/path/to/dataset_diag \
        python scripts/diagnose_robustness.py [--max-iter 5000] [--workers 3]
"""
from __future__ import annotations

import argparse
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# --------------------------------------------------------------------------
# Default matrix: 5 expected-failing high-AoA / high-Re cases + 1 in-band
# control (D6 must pass before AND after any fix, with unchanged Cl/Cd).
# --------------------------------------------------------------------------
MATRIX = [
    ("D1", "0012", 10.0, 2.0e6),   # high Re, wall-cell AR ~650
    ("D2", "2412", 12.0, 1.0e6),   # camber + max AoA
    ("D3", "0012", 12.0, 5.0e5),   # isolates AoA from Re
    ("D4", "0006", 10.0, 1.0e6),   # thin section, sharp-TE worst case
    ("D5", "4412",  8.0, 2.0e6),   # high camber + max Re
    ("D6", "2412",  4.0, 3.0e5),   # in-band control
]


FATAL_RE = re.compile(r"FOAM FATAL", re.IGNORECASE)
# Case-sensitive: the startup banner "sigFpe : Enabling floating point
# exception trapping" must NOT match — only a real FPE crash signature does.
SIGFPE_RE = re.compile(r"sigFpe::sigHandler|Floating point exception")
SOLVING_RE = re.compile(r"Solving for (\w+)")
BOUNDING_RE = re.compile(r"bounding (\w+)")
TIME_LINE_RE = re.compile(r"^Time = ([\d.eE+-]+)s?\s*$", re.MULTILINE)
VECTOR_RE = re.compile(r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+[-\d.eE+]+\s*\)")


def _analyze_log(log_text: str, parse_solver_log, detect_convergence) -> dict:
    parsed = parse_solver_log(log_text)
    conv = detect_convergence(parsed["history"])
    times = TIME_LINE_RE.findall(log_text)

    # Iteration where any field's residual first exceeds 1 after iter 5.
    first_explode_iter, explode_field = None, None
    for f, vals in parsed["history"].items():
        for i, v in enumerate(vals):
            if i >= 5 and v == v and v > 1.0:
                if first_explode_iter is None or i < first_explode_iter:
                    first_explode_iter, explode_field = i, f
                break

    bounding = BOUNDING_RE.findall(log_text)
    solving = SOLVING_RE.findall(log_text)
    return {
        "n_iter": parsed["n_iter"],
        "last_time": times[-1] if times else None,
        "final_residuals": parsed["final"],
        "orders_drop": conv["drops"],
        "converged": conv["overall"],
        "fatal": bool(FATAL_RE.search(log_text)),
        "sigfpe": bool(SIGFPE_RE.search(log_text)),
        "fatal_excerpt": _fatal_excerpt(log_text),
        "first_explode_iter": first_explode_iter,
        "first_explode_field": explode_field,
        "bounding_k": bounding.count("k"),
        "bounding_omega": bounding.count("omega"),
        "last_field_solved": solving[-1] if solving else None,
    }


def _fatal_excerpt(log_text: str) -> str | None:
    m = FATAL_RE.search(log_text)
    if not m:
        return None
    return log_text[m.start():m.start() + 400]


def _classify(diag: dict) -> str:
    run = diag["foam_run"]
    if run["converged"]:
        return "converged"
    if run["fatal"] or run["sigfpe"] or diag["foam_rc"] != 0:
        f = run["first_explode_field"]
        if run["bounding_k"] + run["bounding_omega"] > 50 and f in (None, "k", "omega"):
            return "turbulence_blowup"
        if f == "p" or (run["fatal_excerpt"] and "GAMG" in run["fatal_excerpt"]):
            return "pressure_blowup"
        if f in ("Ux", "Uy"):
            return "momentum_blowup"
        return "unclassified_crash"
    return "slow_no_converge"


def run_one(entry: tuple, max_iter: int, timeout: int) -> dict:
    import yaml
    from physics_oracle.core.case_spec import CaseSpec
    from physics_oracle.meshing.c_mesh import (
        _first_cell_height, build_c_mesh_nodes, generate_c_mesh)
    from physics_oracle.openfoam_setup.case_setup import setup_openfoam_case
    from physics_oracle.openfoam_setup.runner import (
        detect_convergence, needs_startup_stage, parse_solver_log,
        run_simple_foam)

    tag, naca, aoa, re_val = entry
    spec = CaseSpec.build(naca, aoa, re_val, split="diagnostic")
    of_case = spec.of_case_dir
    inl = spec.inlet()
    diag: dict = {"tag": tag, "case_id": spec.case_id,
                  "naca": naca, "aoa_deg": aoa, "Re": re_val,
                  "U_inf": inl.U_mag, "k_inlet": inl.k_inlet,
                  "omega_inlet": inl.omega_inlet,
                  "visc_ratio_inlet": inl.k_inlet / inl.omega_inlet / inl.nu}

    try:
        # 1. Case + mesh
        setup_openfoam_case(of_case, spec, end_time=max_iter)
        diag["mesh_quality"] = generate_c_mesh(of_case, spec.case_id)
        u_size = re_val * inl.nu  # chord = 1
        diag["y1"] = _first_cell_height(0.8, re_val, u_size, inl.nu)
        nodes, _, _ = build_c_mesh_nodes(naca, re_val, aoa)
        diag["n_layers"] = int(nodes.shape[1] - 1)

        # 2. the production solver path (uniform cold start; single- or
        #    two-stage per the gate).  No potentialFoam probe: the init was
        #    shown to hurt convergence and is no longer part of the pipeline.
        diag["two_stage_gate"] = needs_startup_stage(spec)
        t0 = time.time()
        diag["foam_rc"] = run_simple_foam(of_case, timeout=timeout,
                                          spec=spec, end_time=max_iter)
        diag["foam_wall_s"] = round(time.time() - t0, 1)
        run_info = of_case / "run_info.yaml"
        if run_info.exists():
            diag["run_info"] = yaml.safe_load(run_info.read_text())
        log_text = (of_case / "simpleFoam.log").read_text(errors="replace")
        diag["foam_run"] = _analyze_log(log_text, parse_solver_log,
                                        detect_convergence)
        diag["classification"] = _classify(diag)
    except Exception as exc:  # keep the matrix going
        diag["error"] = f"{exc.__class__.__name__}: {exc}"
        diag["classification"] = "pipeline_error"

    (spec.case_dir / "diagnosis.yaml").write_text(
        yaml.safe_dump(diag, sort_keys=False))
    return diag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--timeout", type=int, default=7200,
                    help="per-case foamRun timeout [s]")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of tags to run, e.g. --only D1 D4")
    args = ap.parse_args()

    if "PHYSICS_ORACLE_DATASET_ROOT" not in os.environ:
        raise SystemExit("Set PHYSICS_ORACLE_DATASET_ROOT to a scratch dir "
                         "so diagnostics never touch a real dataset.")

    import yaml
    from physics_oracle.core.paths import DATASET_ROOT

    matrix = [e for e in MATRIX if args.only is None or e[0] in args.only]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(
            lambda e: run_one(e, args.max_iter, args.timeout), matrix))

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / "diagnosis_summary.yaml").write_text(
        yaml.safe_dump(results, sort_keys=False))

    hdr = f"{'tag':4} {'case':24} {'class':22} {'rc':>3} {'iters':>6} {'minDrop':>8}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for d in results:
        run = d.get("foam_run", {})
        drops = run.get("orders_drop", {})
        min_drop = min(drops.values()) if drops else float("nan")
        print(f"{d['tag']:4} {d['case_id']:24} {d['classification']:22} "
              f"{d.get('foam_rc', -1):>3} {run.get('n_iter', 0):>6} {min_drop:>8.2f}")


if __name__ == "__main__":
    main()
