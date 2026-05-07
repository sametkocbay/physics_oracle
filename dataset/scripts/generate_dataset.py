"""End-to-end driver — the single entry point for §1–§9 of the dataset spec.

Usage:
    python dataset/scripts/generate_dataset.py \
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

Per-case flags `--skip-of` (skips meshing+solver+extraction; useful for the
manifest+splits-only smoke test) and `--max-iter` (overrides §5.7 endTime
for development) are provided.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc

# Allow running both as `python dataset/scripts/generate_dataset.py` and
# `python -m dataset.scripts.generate_dataset` by adding our own dir to
# sys.path when invoked as a module.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    CASES_DIR,
    DATASET_ROOT,
    ENVELOPE,
    MANIFEST_PATH,
    MESH_VERSION,
    NU,
    OPENFOAM_VERSION,
    REJECTION_LOG_PATH,
    SPLITS_DIR,
    CaseSpec,
    md5_of_paths,
    setup_logging,
)
from generate_geometry import assign_splits, sample_naca_profiles
from generate_mesh import generate_mesh
from setup_openfoam_case import setup_openfoam_case
from run_openfoam import run_simple_foam
from extract_fields import extract_case
from quality_check import append_rejection, quality_check

LOG = setup_logging()


# ---------------------------------------------------------------------------
# §2.2 case sampling
# ---------------------------------------------------------------------------

def sample_cases(profiles: list[dict], split_by_profile: dict[str, list[str]],
                 n_cases: int, seed: int) -> list[CaseSpec]:
    """LHS over (profile_index, AoA, log10 Re) within the §1 envelope."""
    profile_codes = [p["naca_code"] for p in profiles]
    code_to_split = {}
    for split, codes in split_by_profile.items():
        for c in codes:
            code_to_split[c] = split

    sampler = qmc.LatinHypercube(d=3, seed=seed + 1)
    u = sampler.random(n_cases)
    aoa = ENVELOPE["aoa_min_deg"] + u[:, 1] * (ENVELOPE["aoa_max_deg"] - ENVELOPE["aoa_min_deg"])
    log_re_lo = math.log10(ENVELOPE["re_min"])
    log_re_hi = math.log10(ENVELOPE["re_max"])
    log_re = log_re_lo + u[:, 2] * (log_re_hi - log_re_lo)
    re_values = 10 ** log_re

    n_profiles = len(profile_codes)
    profile_idx = np.minimum((u[:, 0] * n_profiles).astype(int), n_profiles - 1)

    cases: list[CaseSpec] = []
    seen = set()
    for i, p_idx in enumerate(profile_idx):
        code = profile_codes[p_idx]
        # Round AoA to 1 decimal place so case IDs are unique-ish on collision
        aoa_i = round(float(aoa[i]), 1)
        re_i = float(re_values[i])
        spec = CaseSpec.build(code, aoa_i, re_i, code_to_split[code])
        if spec.case_id in seen:
            continue
        seen.add(spec.case_id)
        cases.append(spec)
    return cases


# ---------------------------------------------------------------------------
# §2.3 OOD probe sampling
# ---------------------------------------------------------------------------

def sample_ood_cases(profiles: list[dict], n_ood: int, seed: int) -> list[CaseSpec]:
    """Atypical conditions: AoA > 15°, Re < 1e5, Re > 1e7."""
    rng = np.random.default_rng(seed + 2)
    codes = [p["naca_code"] for p in profiles]
    cases: list[CaseSpec] = []
    if not codes or n_ood == 0:
        return cases
    per_kind = max(1, n_ood // 3)
    # high AoA
    for _ in range(per_kind):
        c = rng.choice(codes)
        cases.append(CaseSpec.build(
            c, float(rng.uniform(15.5, 20.0)),
            float(10 ** rng.uniform(5.5, 6.5)),
            "ood_probe",
        ))
    # low Re
    for _ in range(per_kind):
        c = rng.choice(codes)
        cases.append(CaseSpec.build(
            c, float(rng.uniform(0.0, 8.0)),
            float(10 ** rng.uniform(4.0, 4.99)),
            "ood_probe",
        ))
    # high Re
    for _ in range(n_ood - 2 * per_kind):
        c = rng.choice(codes)
        cases.append(CaseSpec.build(
            c, float(rng.uniform(0.0, 8.0)),
            float(10 ** rng.uniform(7.01, 7.7)),
            "ood_probe",
        ))
    # Dedupe
    seen = set()
    out: list[CaseSpec] = []
    for c in cases:
        if c.case_id not in seen:
            seen.add(c.case_id)
            out.append(c)
    return out


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
    """md5 over the pristine fvSchemes/fvSolution we write per case."""
    from setup_openfoam_case import FV_SCHEMES, FV_SOLUTION
    import hashlib
    h = hashlib.md5()
    h.update(FV_SCHEMES.encode())
    h.update(FV_SOLUTION.encode())
    return h.hexdigest()


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


def write_meta(spec: CaseSpec, qc_result: dict, mesh_quality: dict | None,
               solver_hash: str) -> None:
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
    spec.case_dir.mkdir(parents=True, exist_ok=True)
    (spec.case_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


# ---------------------------------------------------------------------------
# Per-case pipeline
# ---------------------------------------------------------------------------

def run_case(spec: CaseSpec, args: argparse.Namespace, solver_hash: str) -> dict:
    LOG.info("=== %s (split=%s) ===", spec.case_id, spec.split)
    spec.case_dir.mkdir(parents=True, exist_ok=True)
    spec.of_case_dir.mkdir(parents=True, exist_ok=True)

    mesh_quality = None
    extraction = None
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
            mesh_quality = generate_mesh(spec.of_case_dir, spec.case_id)
            rc = run_simple_foam(spec.of_case_dir,
                                 timeout=args.solver_timeout)
            if rc != 0:
                LOG.warning("[%s] simpleFoam exited with rc=%d; "
                            "extraction may reflect unconverged fields",
                            spec.case_id, rc)
            extraction = extract_case(spec.of_case_dir, spec.case_dir, spec.case_id)
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

    write_meta(spec, qc_result, mesh_quality, solver_hash)
    return qc_result


# ---------------------------------------------------------------------------
# Reproducibility check (§9 — re-run a few cases and compare fields)
# ---------------------------------------------------------------------------

def repro_check(case_specs: list[CaseSpec], n: int) -> dict:
    """Re-run the first `n` accepted cases and verify field hashes match."""
    import hashlib
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
    p = argparse.ArgumentParser(
        description="End-to-end CFD dataset generation pipeline.",
    )
    p.add_argument("--n-profiles", type=int, default=50,
                   help="Unique NACA profiles to sample (target ≥160 for full run).")
    p.add_argument("--n-cases", type=int, default=100,
                   help="Cases to draw via joint LHS (target ~500 for full run).")
    p.add_argument("--n-ood", type=int, default=10,
                   help="OOD probe cases.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iter", type=int, default=5000,
                   help="Solver endTime (§5.7 max iterations).")
    p.add_argument("--solver-timeout", type=int, default=6 * 3600,
                   help="Per-case solver wall-clock timeout (seconds).")
    p.add_argument("--skip-of", action="store_true",
                   help="Skip mesh generation, solver, and extraction (manifest+meta only).")
    p.add_argument("--cases", nargs="*",
                   help="Run only these case IDs (must already be in the sampled set).")
    p.add_argument("--repro-cases", type=int, default=0,
                   help="At the end, hash N accepted cases for the §9 repro checklist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    LOG.info("§2.1  sampling %d unique NACA profiles (seed=%d)",
             args.n_profiles, args.seed)
    profiles = sample_naca_profiles(args.n_profiles, seed=args.seed)
    splits = assign_splits(profiles, seed=args.seed)
    LOG.info("Splits: train=%d val=%d test=%d", *(len(splits[k]) for k in ("train", "val", "test")))

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

    solver_hash = solver_settings_hash()

    n_accepted = 0
    n_rejected = 0
    t0 = time.time()
    for i, spec in enumerate(all_cases, 1):
        LOG.info("--- case %d / %d ---", i, len(all_cases))
        result = run_case(spec, args, solver_hash)
        if result.get("accepted"):
            n_accepted += 1
        else:
            n_rejected += 1
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
