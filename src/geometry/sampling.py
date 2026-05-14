"""LHS sampling: profiles, splits, in-domain cases, OOD probe cases.

Pulled from the original generate_geometry.py and generate_dataset.py so all
sampling logic lives in one place under src/geometry/.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import qmc

from core.case_spec import CaseSpec
from core.envelope import ENVELOPE
from core.logging import setup_logging

from .naca import naca4_code

LOG = setup_logging()


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
