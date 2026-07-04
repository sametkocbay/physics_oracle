"""LHS sampling: profiles, splits, in-domain cases, OOD probe cases.

Pulled from the original generate_geometry.py and generate_dataset.py so all
sampling logic lives in one place under src/geometry/.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from physics_oracle.core.case_spec import CaseSpec
from physics_oracle.core.envelope import ENVELOPE, OOD_BUCKET_WEIGHTS, OOD_ENVELOPE
from physics_oracle.core.logging import setup_logging

from .naca import naca4_code

LOG = setup_logging()

_CODE_RE = re.compile(r"NACA(\d{4})")


# ---------------------------------------------------------------------------
# §2.1 NACA profile sampling
# ---------------------------------------------------------------------------

def sample_naca_profiles(n_profiles: int, seed: int) -> list[dict]:
    """Latin Hypercube over (camber%, position%, thickness%) (§2.1).

    Oversamples and de-duplicates to land on `n_profiles` unique 4-digit codes.
    """
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    bounds_lo = np.array([
        ENVELOPE["camber_pct_min"],
        ENVELOPE["camber_pos_pct_min"],
        ENVELOPE["thickness_pct_min"],
    ])
    bounds_hi = np.array([
        ENVELOPE["camber_pct_max"],
        ENVELOPE["camber_pos_pct_max"],
        ENVELOPE["thickness_pct_max"],
    ])

    profiles: dict[str, dict] = {}
    oversample = 4
    attempts = 0
    while len(profiles) < n_profiles and attempts < 32:
        n_draw = max(n_profiles * oversample, 32)
        u = sampler.random(n_draw)
        x = qmc.scale(u, bounds_lo, bounds_hi)
        for camber, position, thickness in x:
            code = naca4_code(camber, position, thickness)
            if code in profiles:
                continue
            profiles[code] = {
                "naca_code": code,
                "camber_pct": float(camber),
                "camber_pos_pct": float(position),
                "thickness_pct": float(thickness),
            }
            if len(profiles) >= n_profiles:
                break
        attempts += 1

    if len(profiles) < n_profiles:
        LOG.warning(
            "Only %d unique NACA codes found after %d LHS draws (requested %d).",
            len(profiles), attempts, n_profiles,
        )
    return list(profiles.values())[:n_profiles]


def assign_splits(
    profiles: list[dict],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Profile-level split — val/test profiles are completely unseen (§2.3)."""
    rng = np.random.default_rng(seed)
    codes = [p["naca_code"] for p in profiles]
    rng.shuffle(codes)
    n = len(codes)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    return {
        "train": sorted(codes[:n_train]),
        "val":   sorted(codes[n_train:n_train + n_val]),
        "test":  sorted(codes[n_train + n_val:]),
    }


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
    for _ in range(per_kind):
        c = rng.choice(codes)
        cases.append(CaseSpec.build(
            c, float(rng.uniform(15.5, 20.0)),
            float(10 ** rng.uniform(5.5, 6.5)),
            "ood_probe",
        ))
    for _ in range(per_kind):
        c = rng.choice(codes)
        cases.append(CaseSpec.build(
            c, float(rng.uniform(0.0, 8.0)),
            float(10 ** rng.uniform(4.0, 4.99)),
            "ood_probe",
        ))
    for _ in range(n_ood - 2 * per_kind):
        c = rng.choice(codes)
        cases.append(CaseSpec.build(
            c, float(rng.uniform(0.0, 8.0)),
            float(10 ** rng.uniform(7.01, 7.7)),
            "ood_probe",
        ))
    seen = set()
    out: list[CaseSpec] = []
    for c in cases:
        if c.case_id not in seen:
            seen.add(c.case_id)
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# OOD test-set sampling (atypical geometry + high |AoA|, mesh-valid Re)
# ---------------------------------------------------------------------------

def collect_existing_codes(*roots: "str | Path") -> set[str]:
    """Collect every 4-digit NACA code already present under ``roots``.

    Scans the top level of each root (case-id directories, as in the raw
    OpenFOAM dataset) plus one level of split subfolders (``train/`` ``val/``
    ``test/`` ``ood/`` of the ML dataset, whose ``.npz`` files are named after
    the case id).  Used so the OOD sampler only proposes geometries that are
    not already in the dataset.
    """
    codes: set[str] = set()

    def _scan(d: Path) -> None:
        try:
            entries = list(d.iterdir())
        except (PermissionError, NotADirectoryError, FileNotFoundError):
            return
        for entry in entries:
            m = _CODE_RE.search(entry.name)
            if m:
                codes.add(m.group(1))

    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        _scan(root)
        for sub in root.iterdir():
            if sub.is_dir() and not _CODE_RE.search(sub.name):
                _scan(sub)
    return codes


_BUCKET_RANGES = {
    "thin": ("thin_thickness_pct", (ENVELOPE["camber_pct_min"], ENVELOPE["camber_pct_max"])),
    "thick": ("thick_thickness_pct", (ENVELOPE["camber_pct_min"], ENVELOPE["camber_pct_max"])),
    "high_camber": ("thickness_in_domain", OOD_ENVELOPE["high_camber_pct"]),
}


def _sample_ood_bucket(name: str, n: int, exclude_codes: set[str],
                       seen_ids: set[str], rng: np.random.Generator) -> list[CaseSpec]:
    """Draw ``n`` OOD candidate cases for one atypical-geometry family."""
    thickness_key, camber_range = _BUCKET_RANGES[name]
    if thickness_key == "thickness_in_domain":
        t_lo, t_hi = ENVELOPE["thickness_pct_min"], ENVELOPE["thickness_pct_max"]
    else:
        t_lo, t_hi = OOD_ENVELOPE[thickness_key]
    c_lo, c_hi = camber_range
    p_lo, p_hi = OOD_ENVELOPE["camber_pos_pct"]
    a_lo, a_hi = OOD_ENVELOPE["aoa_abs_min_deg"], OOD_ENVELOPE["aoa_abs_max_deg"]
    log_re_lo = math.log10(OOD_ENVELOPE["re_min"])
    log_re_hi = math.log10(OOD_ENVELOPE["re_max"])

    out: list[CaseSpec] = []
    attempts = 0
    while len(out) < n and attempts < n * 200 + 200:
        attempts += 1
        thickness = rng.uniform(t_lo, t_hi)
        camber = rng.uniform(c_lo, c_hi)
        position = rng.uniform(p_lo, p_hi)
        code = naca4_code(camber, position, thickness)
        if code in exclude_codes:
            continue  # geometry must not already be in the dataset
        aoa = round(float(rng.uniform(a_lo, a_hi)) * rng.choice([-1.0, 1.0]), 1)
        re_val = float(10 ** rng.uniform(log_re_lo, log_re_hi))
        spec = CaseSpec.build(code, aoa, re_val, "ood")
        if spec.case_id in seen_ids:
            continue
        seen_ids.add(spec.case_id)
        out.append(spec)
    if len(out) < n:
        LOG.warning("OOD bucket %r: only %d/%d candidates after %d attempts",
                    name, len(out), n, attempts)
    return out


def sample_ood_set(n_target: int, exclude_codes: set[str], exclude_ids: set[str],
                   seed: int, oversample: int = 3) -> list[CaseSpec]:
    """Sample an oversampled pool of OOD candidate cases (split='ood').

    OOD-ness comes from atypical geometry (thin / thick / strongly cambered
    sections absent from the trained set) combined with |AoA| beyond the
    trained +/-5 deg, which drives |Cl| past anything in-domain.  Reynolds is
    held inside the mesh-valid band.  ``oversample`` cases per target are
    drawn so the caller can keep running until ``n_target`` converge.
    """
    rng = np.random.default_rng(seed + 7)
    n_pool = max(n_target * max(1, oversample), n_target + 8)

    total_w = sum(OOD_BUCKET_WEIGHTS.values())
    seen_ids = set(exclude_ids)
    cases: list[CaseSpec] = []
    for name, weight in OOD_BUCKET_WEIGHTS.items():
        n_bucket = max(1, int(round(n_pool * weight / total_w)))
        cases.extend(_sample_ood_bucket(name, n_bucket, exclude_codes, seen_ids, rng))

    rng.shuffle(cases)
    LOG.info("Sampled %d OOD candidates (target %d accepted, oversample x%d)",
             len(cases), n_target, oversample)
    return cases


def sample_fill_cases(profiles: list[dict], splits: dict[str, list[str]],
                      exclude_ids: set[str], n_needed: int,
                      base_seed: int) -> list[CaseSpec]:
    """Sample new cases that don't overlap with exclude_ids.

    Samples in batches with increasing seeds until we have at least
    2 * n_needed candidates.
    """
    new_cases: list[CaseSpec] = []
    seen_new: set[str] = set()
    seed_offset = 10_000
    target_buffer = n_needed * 2

    while len(new_cases) < target_buffer and seed_offset < 10_000_000:
        batch = sample_cases(profiles, splits,
                             n_cases=max(n_needed * 5, 100),
                             seed=base_seed + seed_offset)
        for c in batch:
            if c.case_id not in exclude_ids and c.case_id not in seen_new:
                new_cases.append(c)
                seen_new.add(c.case_id)
        seed_offset += 10_000

    LOG.info("Sampled %d fill candidates (want %d accepted)", len(new_cases), n_needed)
    return new_cases
