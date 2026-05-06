"""§2.1 — NACA 4-digit profile generation + Latin Hypercube sampling.

CLI:
    python generate_geometry.py --n-profiles 210 --seed 0 --out profiles.json

The output JSON lists unique profiles (NACA codes) with their continuous LHS
samples — used by the orchestrator to build the joint sample space.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from common import ENVELOPE, naca4_code, naca4_coordinates, setup_logging

LOG = setup_logging()


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate NACA 4-digit profile pool via LHS.")
    p.add_argument("--n-profiles", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    profiles = sample_naca_profiles(args.n_profiles, seed=args.seed)
    splits = assign_splits(profiles, seed=args.seed)
    payload = {"profiles": profiles, "splits": splits, "seed": args.seed}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        LOG.info("Wrote %d profiles to %s", len(profiles), args.out)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
