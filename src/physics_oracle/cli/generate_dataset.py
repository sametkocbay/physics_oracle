"""End-to-end driver — the single entry point for §1–§9 of the dataset spec.

Usage:
    uv run python scripts/generate_dataset.py \\
        --n-profiles 50 --n-cases 200 --n-ood 10 --seed 0

What it does, in order:
    1.  Sample NACA 4-digit profiles via Latin Hypercube (§2.1).
    2.  Assign profiles to train/val/test splits (§2.3).
    3.  Sample (profile, AoA, log Re) cases via LHS within the operating
        envelope (§2.2).
    4.  Sample OOD probe cases (§2.3).
    5.  Write splits/<split>.txt and manifest.yaml.
    6.  For each case: setup OF case → mesh → run simpleFoam → extract
        HDF5 → quality check → write meta.yaml.

Per-case flags ``--skip-of`` (skips meshing+solver+extraction; useful for the
manifest+splits-only smoke test) and ``--max-iter`` (overrides §5.7 endTime
for development) are provided.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from physics_oracle.core import (
    CASES_DIR,
    DATASET_ROOT,
    ENVELOPE,
    MANIFEST_PATH,
    MESH_VERSION,
    NU,
    OOD_ENVELOPE,
    OPENFOAM_CONFIG_PATH,
    OPENFOAM_VERSION,
    SPLITS_DIR,
    CaseSpec,
    setup_logging,
)
from physics_oracle.geometry.sampling import (
    assign_splits,
    collect_existing_codes,
    sample_cases,
    sample_fill_cases,
    sample_naca_profiles,
    sample_ood_cases,
    sample_ood_set,
)
from physics_oracle.meshing import generate_c_mesh, generate_mesh
from physics_oracle.openfoam_setup import (
    append_rejection,
    extract_case,
    quality_check,
    run_simple_foam,
    setup_openfoam_case,
)

LOG = setup_logging()


def _to_native(obj):
    """Recursively coerce numpy scalars / arrays to Python types for yaml."""
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.str_, np.bytes_)):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Manifest, splits
# ---------------------------------------------------------------------------

def write_splits(cases: list[CaseSpec]) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": [], "ood_probe": []}
    for c in cases:
        by_split.setdefault(c.split, []).append(c.case_id)
    for split, ids in by_split.items():
        (SPLITS_DIR / f"{split}.txt").write_text("\n".join(sorted(ids)) + "\n")


def solver_settings_hash() -> str:
    """md5 of configs/openfoam.yaml — replaces the old fvSchemes+fvSolution hash."""
    return hashlib.md5(OPENFOAM_CONFIG_PATH.read_bytes()).hexdigest()


def write_manifest(profiles: list[dict], splits: dict[str, list[str]],
                   cases: list[CaseSpec], ood_cases: list[CaseSpec],
                   seed: int, args: argparse.Namespace) -> None:
    payload = {
        "openfoam_version": OPENFOAM_VERSION,
        "mesh_version": MESH_VERSION,
        "solver_settings_hash": solver_settings_hash(),
        "generation_timestamp": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "envelope": ENVELOPE,
        "kinematic_viscosity": NU,
        "seeds": {"global": seed, "geometry_lhs": seed,
                  "case_lhs": seed + 1, "ood": seed + 2},
        "sampling": {
            "n_profiles_requested": args.n_profiles,
            "n_profiles_actual": len(profiles),
            "n_cases_requested": args.n_cases,
            "n_cases_actual": len(cases),
            "n_ood_requested": args.n_ood,
            "n_ood_actual": len(ood_cases),
        },
        "profile_splits": splits,
        "profiles": profiles,
        "cases": [c.case_id for c in cases],
        "ood_cases": [c.case_id for c in ood_cases],
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(yaml.safe_dump(_to_native(payload), sort_keys=False))
    LOG.info("Wrote manifest %s", MANIFEST_PATH)


# ---------------------------------------------------------------------------
# Per-case meta.yaml (§6.5)
# ---------------------------------------------------------------------------

def write_meta(spec: CaseSpec, qc_result: dict, mesh_quality: dict | None,
               solver_hash: str, solver_run_info: dict | None = None) -> None:
    inl = spec.inlet()
    meta = {
        "case_id": str(spec.case_id),
        "naca_code": str(spec.naca_code),
        "aoa_deg": float(spec.aoa_deg),
        "Re": float(spec.Re),
        "U_inlet": [float(inl.U_x), float(inl.U_y)],
        "U_mag": float(inl.U_mag),
        "nu": float(inl.nu),
        "chord": 1.0,
        "k_inlet": float(inl.k_inlet),
        "omega_inlet": float(inl.omega_inlet),
        "mesh_version": MESH_VERSION,
        "openfoam_version": OPENFOAM_VERSION,
        "solver_settings_hash": solver_hash,
        "generation_timestamp": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "converged": bool(qc_result.get("metrics", {}).get("converged", False)),
        "iterations_to_convergence": int(qc_result.get("metrics", {}).get("n_iter", 0)),
        "flags": list(spec.flags) + qc_result.get("flags", []),
        "rejections": qc_result.get("rejections", []),
        "split": spec.split,
        "qc_metrics": _to_native(qc_result.get("metrics", {})),
        "mesh_quality": _to_native(mesh_quality),
    }
    if solver_run_info is not None:
        meta["solver_run_info"] = _to_native(solver_run_info)
    spec.case_dir.mkdir(parents=True, exist_ok=True)
    (spec.case_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


# ---------------------------------------------------------------------------
# Per-case pipeline
# ---------------------------------------------------------------------------

def run_case(spec: CaseSpec, args: argparse.Namespace, solver_hash: str) -> dict:
    LOG.info("=== %s (split=%s) ===", spec.case_id, spec.split)
    if (spec.case_dir / "fields.h5").exists():
        LOG.info("[%s] already complete, skipping", spec.case_id)
        return {"accepted": True, "skipped": True}
    spec.case_dir.mkdir(parents=True, exist_ok=True)
    spec.of_case_dir.mkdir(parents=True, exist_ok=True)

    mesh_quality = None
    qc_result = {"accepted": False, "rejections": ["pipeline_not_run"],
                 "flags": [], "metrics": {}}

    try:
        setup_openfoam_case(spec.of_case_dir, spec, end_time=args.max_iter)

        if args.skip_of:
            LOG.info("[%s] --skip-of: writing meta only", spec.case_id)
            qc_result = {"accepted": False, "rejections": ["skip_of"],
                         "flags": ["dry_run"], "metrics": {"n_iter": 0,
                                                            "converged": False}}
        else:
            mesher = generate_c_mesh if args.c_mesh else generate_mesh
            mesh_quality = mesher(spec.of_case_dir, spec.case_id)
            if mesh_quality and mesh_quality.get("errors"):
                # checkMesh reported hard errors (negative volumes, open
                # cells, ...) — the solver would SIGFPE on such a mesh, so
                # reject cleanly instead of running it.  Seen for extreme
                # forward-camber profiles whose freestream-aligned wake cut
                # folds the TFI grid at strongly negative AoA.
                qc_result = {"accepted": False, "rejections": ["mesh_invalid"],
                             "flags": [], "metrics": {}}
                append_rejection(spec.case_id, qc_result["rejections"])
                LOG.warning("[%s] rejected: invalid mesh (%s)", spec.case_id,
                            mesh_quality["errors"][0].strip())
                write_meta(spec, qc_result, mesh_quality, solver_hash)
                return qc_result
            rc = run_simple_foam(spec.of_case_dir, timeout=args.solver_timeout,
                                 spec=spec, end_time=args.max_iter)
            if rc != 0:
                LOG.warning("[%s] simpleFoam exited with rc=%d; "
                            "extraction may reflect unconverged fields",
                            spec.case_id, rc)
            extract_case(spec.of_case_dir, spec.case_dir, spec.case_id)
            qc_result = quality_check(spec.case_dir, max_iter=args.max_iter)
            if not qc_result["accepted"]:
                append_rejection(spec.case_id, qc_result["rejections"])
                LOG.warning("[%s] rejected: %s", spec.case_id, qc_result["rejections"])
            else:
                LOG.info("[%s] accepted", spec.case_id)

    except Exception as exc:
        LOG.error("[%s] pipeline failed: %s", spec.case_id, exc)
        LOG.debug(traceback.format_exc())
        append_rejection(spec.case_id, [f"exception:{exc.__class__.__name__}",
                                        str(exc)[:120]])
        qc_result = {"accepted": False, "rejections": [str(exc)],
                     "flags": ["exception"], "metrics": {}}

    run_info_path = spec.of_case_dir / "run_info.yaml"
    solver_run_info = (yaml.safe_load(run_info_path.read_text())
                       if run_info_path.exists() else None)
    write_meta(spec, qc_result, mesh_quality, solver_hash, solver_run_info)
    return qc_result


# ---------------------------------------------------------------------------
# Fill-mode helpers
# ---------------------------------------------------------------------------

def get_existing_cases() -> tuple[set[str], set[str]]:
    """Return (all_tried_ids, accepted_ids) by scanning CASES_DIR on disk."""
    all_tried: set[str] = set()
    accepted: set[str] = set()
    if CASES_DIR.exists():
        for d in CASES_DIR.iterdir():
            if d.is_dir():
                all_tried.add(d.name)
                if (d / "fields.h5").exists():
                    accepted.add(d.name)
    return all_tried, accepted


def get_split_case_ids() -> set[str]:
    ids: set[str] = set()
    if SPLITS_DIR.exists():
        for f in SPLITS_DIR.glob("*.txt"):
            for line in f.read_text().splitlines():
                line = line.strip()
                if line:
                    ids.add(line)
    return ids


def append_to_splits(new_cases: list[CaseSpec]) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, set[str]] = {}
    for f in SPLITS_DIR.glob("*.txt"):
        content = f.read_text().strip()
        by_split[f.stem] = set(content.splitlines()) if content else set()
    for c in new_cases:
        by_split.setdefault(c.split, set()).add(c.case_id)
    for split_name, ids in by_split.items():
        (SPLITS_DIR / f"{split_name}.txt").write_text("\n".join(sorted(ids)) + "\n")


# ---------------------------------------------------------------------------
# Reproducibility check (§9 — re-run a few cases and compare fields)
# ---------------------------------------------------------------------------

def repro_check(case_specs: list[CaseSpec], n: int) -> dict:
    import h5py
    out: dict[str, dict] = {}
    accepted = [s for s in case_specs if (s.case_dir / "fields.h5").exists()][:n]
    for spec in accepted:
        with h5py.File(spec.case_dir / "fields.h5", "r") as h:
            U = h["U"][:]
            p = h["p"][:]
        h_orig = hashlib.md5(np.ascontiguousarray(np.concatenate([U.ravel(),
                                                                  p.ravel()])).tobytes()).hexdigest()
        out[spec.case_id] = {"hash": h_orig, "n_cells": len(p)}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end CFD dataset generation pipeline.")
    p.add_argument("--n-profiles", type=int, default=50,
                   help="Unique NACA profiles to sample (target ≥160 for full run).")
    p.add_argument("--n-cases", type=int, default=100,
                   help="Cases to draw via joint LHS (target ~500 for full run).")
    p.add_argument("--n-ood", type=int, default=10, help="OOD probe cases.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iter", type=int, default=5000,
                   help="Solver endTime (§5.7 max iterations).")
    p.add_argument("--solver-timeout", type=int, default=6 * 3600,
                   help="Per-case solver wall-clock timeout (seconds).")
    p.add_argument("--c-mesh", action="store_true",
                   help="Use the structured C-mesh generator instead of the default Gmsh.")
    p.add_argument("--skip-of", action="store_true",
                   help="Skip mesh generation, solver, and extraction (manifest+meta only).")
    p.add_argument("--cases", nargs="*",
                   help="Run only these case IDs (must already be in the sampled set).")
    p.add_argument("--repro-cases", type=int, default=0,
                   help="At the end, hash N accepted cases for the §9 repro checklist.")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of cases to run in parallel (default: 1 = serial).")
    p.add_argument("--fill", action="store_true",
                   help="Fill mode: scan cases/ and splits/ for already-run cases, then "
                        "sample only new cases until --n-cases accepted cases exist.")
    p.add_argument("--ood-fill", action="store_true",
                   help="OOD mode: sample atypical-geometry / high-AoA cases (split='ood') "
                        "and run until --n-ood accepted cases exist. Re stays mesh-valid.")
    p.add_argument("--ood-oversample", type=int, default=3,
                   help="Candidates drawn per requested OOD case (high-AoA cases reject "
                        "more often, so a buffer is needed). Default 3.")
    p.add_argument("--aoa-range", "--aoa_range", dest="aoa_range", nargs=2,
                   type=float, metavar=("MIN", "MAX"), default=None,
                   help="Override the OOD |AoA| band (degrees), e.g. --aoa-range 12 15 "
                        "for a stricter high-angle probe. Sign is still randomised +/-. "
                        "Only affects --ood-fill; default keeps the 6-12 deg envelope.")
    p.add_argument("--re-range", "--re_range", dest="re_range", nargs=2,
                   type=float, metavar=("MIN", "MAX"), default=None,
                   help="Override the OOD Reynolds band, e.g. --re-range 1e6 3e6 to push "
                        "above the trained/mesh-valid 1e5-5e5. NOTE: first-cell height is "
                        "fixed, so higher Re raises y+ past wall-function validity and QC "
                        "may reject cases. Only affects --ood-fill.")
    p.add_argument("--exclude-existing", nargs="*", default=[],
                   help="Dataset roots whose NACA profiles must NOT be reused by --ood-fill "
                        "(e.g. the in-domain ML dataset and raw OpenFOAM dataset).")
    return p.parse_args()


def _run_cases_parallel(all_cases: list[CaseSpec], args: argparse.Namespace,
                         solver_hash: str) -> tuple[int, int]:
    n_accepted = n_rejected = 0
    LOG.info("Running %d cases with %d parallel workers", len(all_cases), args.workers)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, spec, args, solver_hash): spec
                   for spec in all_cases}
        n_done = 0
        for fut in as_completed(futures):
            spec = futures[fut]
            n_done += 1
            try:
                result = fut.result()
            except Exception as exc:
                LOG.error("[%s] worker raised: %s", spec.case_id, exc)
                result = {"accepted": False}
            if result.get("accepted"):
                n_accepted += 1
            else:
                n_rejected += 1
            LOG.info("Progress %d/%d — accepted %d  rejected %d  [%s]",
                     n_done, len(all_cases), n_accepted, n_rejected, spec.case_id)
    return n_accepted, n_rejected


def _run_cases_serial(all_cases: list[CaseSpec], args: argparse.Namespace,
                      solver_hash: str, stop_after_accepted: int | None = None
                      ) -> tuple[int, int, list[CaseSpec]]:
    n_accepted = n_rejected = 0
    cases_run: list[CaseSpec] = []
    for i, spec in enumerate(all_cases, 1):
        LOG.info("--- case %d / %d ---", i, len(all_cases))
        result = run_case(spec, args, solver_hash)
        cases_run.append(spec)
        if result.get("accepted"):
            n_accepted += 1
        else:
            n_rejected += 1
        if stop_after_accepted is not None and n_accepted >= stop_after_accepted:
            LOG.info("Reached %d accepted cases — stopping early.", n_accepted)
            break
    return n_accepted, n_rejected, cases_run


def _run_ood_fill(args: argparse.Namespace) -> None:
    """Sample + run OOD cases (split='ood') until --n-ood accepted exist.

    OOD cases live in their own DATASET_ROOT (set PHYSICS_ORACLE_DATASET_ROOT)
    so they never mix with the in-domain cases. Geometries already present in
    --exclude-existing roots are never reused.
    """
    solver_hash = solver_settings_hash()
    t0 = time.time()

    if args.aoa_range is not None:
        lo, hi = args.aoa_range
        if lo <= 0 or hi <= 0 or hi <= lo:
            raise SystemExit(
                f"--aoa-range needs 0 < MIN < MAX (got {lo} {hi}); values are "
                "|AoA| magnitudes in degrees, sign is randomised at sampling.")
        OOD_ENVELOPE["aoa_abs_min_deg"] = lo
        OOD_ENVELOPE["aoa_abs_max_deg"] = hi
        LOG.info("OOD |AoA| band overridden to [%.1f, %.1f] deg via --aoa-range", lo, hi)

    if args.re_range is not None:
        re_lo, re_hi = args.re_range
        if re_lo <= 0 or re_hi <= 0 or re_hi <= re_lo:
            raise SystemExit(
                f"--re-range needs 0 < MIN < MAX (got {re_lo} {re_hi}).")
        OOD_ENVELOPE["re_min"] = re_lo
        OOD_ENVELOPE["re_max"] = re_hi
        LOG.info("OOD Reynolds band overridden to [%.3g, %.3g] via --re-range", re_lo, re_hi)
        if re_hi > ENVELOPE["re_max"]:
            LOG.warning("OOD re_max %.3g exceeds the mesh-valid trained band (%.3g); "
                        "y+ may leave wall-function validity and QC could reject cases.",
                        re_hi, ENVELOPE["re_max"])

    existing_tried, existing_accepted = get_existing_cases()
    exclude_ids = existing_tried | get_split_case_ids()
    exclude_codes = collect_existing_codes(*args.exclude_existing)
    LOG.info("OOD fill: %d accepted on disk, %d profiles excluded as already-seen",
             len(existing_accepted), len(exclude_codes))

    n_more = args.n_ood - len(existing_accepted)
    if n_more <= 0:
        LOG.info("OOD target of %d already reached. Nothing to do.", args.n_ood)
        return

    candidates = sample_ood_set(n_more, exclude_codes, exclude_ids,
                                args.seed, oversample=args.ood_oversample)

    if args.workers > 1:
        # Parallel path runs the whole pool; the build step caps output at n_ood.
        n_accepted, n_rejected = _run_cases_parallel(candidates, args, solver_hash)
        cases_run = candidates
    else:
        n_accepted, n_rejected, cases_run = _run_cases_serial(
            candidates, args, solver_hash, stop_after_accepted=n_more)

    append_to_splits(cases_run)

    total_accepted = len(existing_accepted) + n_accepted
    LOG.info("OOD fill done: +%d accepted, +%d rejected in %.1f s (total accepted %d / %d)",
             n_accepted, n_rejected, time.time() - t0, total_accepted, args.n_ood)
    if total_accepted < args.n_ood:
        LOG.warning("Still %d short of %d OOD cases — re-run ./generate_ood with a "
                    "different --seed to draw more candidates.",
                    args.n_ood - total_accepted, args.n_ood)


def main() -> None:
    args = parse_args()
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    if args.ood_fill:
        _run_ood_fill(args)
        return

    LOG.info("§2.1  sampling %d unique NACA profiles (seed=%d)",
             args.n_profiles, args.seed)
    profiles = sample_naca_profiles(args.n_profiles, seed=args.seed)
    splits = assign_splits(profiles, seed=args.seed)
    LOG.info("Splits: train=%d val=%d test=%d", *(len(splits[k]) for k in ("train", "val", "test")))

    solver_hash = solver_settings_hash()
    t0 = time.time()

    if args.fill:
        existing_tried, existing_accepted = get_existing_cases()
        split_ids = get_split_case_ids()
        exclude_ids = existing_tried | split_ids

        n_accepted_existing = len(existing_accepted)
        n_more_needed = args.n_cases - n_accepted_existing

        LOG.info(
            "Fill mode: %d accepted on disk, %d in splits, target %d → need %d more accepted",
            n_accepted_existing, len(split_ids), args.n_cases, max(0, n_more_needed),
        )

        if n_more_needed <= 0:
            LOG.info("Target already reached. Nothing to do.")
            return

        all_cases = sample_fill_cases(profiles, splits, exclude_ids,
                                      n_more_needed, args.seed)

        if args.workers > 1:
            n_accepted, n_rejected = _run_cases_parallel(all_cases, args, solver_hash)
            cases_run = all_cases
        else:
            n_accepted, n_rejected, cases_run = _run_cases_serial(
                all_cases, args, solver_hash, stop_after_accepted=n_more_needed)

        append_to_splits(cases_run)

        dt_s = time.time() - t0
        total_accepted = n_accepted_existing + n_accepted
        LOG.info(
            "Fill done: +%d accepted, +%d rejected in %.1f s  (total accepted: %d / %d)",
            n_accepted, n_rejected, dt_s, total_accepted, args.n_cases,
        )
        if total_accepted < args.n_cases:
            LOG.warning(
                "Still %d short of target — re-run with --fill to sample more candidates.",
                args.n_cases - total_accepted,
            )
        return

    LOG.info("§2.2  sampling %d in-domain cases", args.n_cases)
    cases = sample_cases(profiles, splits, n_cases=args.n_cases, seed=args.seed)

    LOG.info("§2.3  sampling %d OOD probe cases", args.n_ood)
    ood_cases = sample_ood_cases(profiles, n_ood=args.n_ood, seed=args.seed)

    all_cases = cases + ood_cases
    if args.cases:
        wanted = set(args.cases)
        all_cases = [c for c in all_cases if c.case_id in wanted]
        LOG.info("Filtered to %d cases on --cases", len(all_cases))

    write_splits(all_cases)
    write_manifest(profiles, splits, cases, ood_cases, args.seed, args)

    if args.workers > 1:
        n_accepted, n_rejected = _run_cases_parallel(all_cases, args, solver_hash)
    else:
        n_accepted, n_rejected, _ = _run_cases_serial(all_cases, args, solver_hash)

    dt_s = time.time() - t0

    LOG.info("DONE: %d accepted, %d rejected in %.1f s", n_accepted, n_rejected, dt_s)
    LOG.info("Acceptance rate: %.1f%%",
             100.0 * n_accepted / max(1, len(all_cases)))

    if args.repro_cases > 0:
        repro = repro_check(all_cases, args.repro_cases)
        (DATASET_ROOT / "repro_hashes.json").write_text(json.dumps(repro, indent=2))
        LOG.info("Wrote %d reproducibility hashes", len(repro))


if __name__ == "__main__":
    main()
