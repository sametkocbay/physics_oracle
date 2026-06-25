"""correct_sdf.py — Recompute the ``sdf`` channel of an ML dataset exactly.

The ``sdf`` channel in the ML ``.npz`` files was originally a nearest-*vertex*
KD-tree query against the mesh boundary points.  That over-estimates the true
wall distance (point-to-vertex instead of point-to-surface) and inherits the
mesh's faceting of the airfoil.

This tool recomputes ``sdf`` as the exact distance to the *analytic* NACA
curve, using :func:`physics_oracle.geometry.distance.naca_surface_distance`.
It needs nothing but each file's own ``naca_code``, ``x`` and ``y`` arrays, so
it is fully idempotent — re-running it reproduces the same result and never
degrades the data.

Usage
-----
    physics-oracle-correct-sdf --dataset-dir ../NACA_4_Digit_for_ML --dry-run
    physics-oracle-correct-sdf --dataset-dir ../NACA_4_Digit_for_ML

Each ``<split>/<case>.npz`` is rewritten in place (atomically); every other
channel and the original dtypes are preserved.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from physics_oracle.core.paths import ML_DATASET_DIR
from physics_oracle.geometry.distance import build_naca_distance_fn

DEFAULT_SPLITS = ("train", "val", "test", "ood")


def _recompute_sdf(npz_path: Path, distance_fn) -> dict:
    """Return stats comparing the recomputed sdf with the stored one."""
    with np.load(npz_path, allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}

    old_sdf = arrays["sdf"]
    points = np.column_stack([arrays["x"], arrays["y"]]).astype(np.float64)
    new_sdf = distance_fn(points).astype(old_sdf.dtype)
    arrays["sdf"] = new_sdf

    diff = new_sdf.astype(np.float64) - old_sdf.astype(np.float64)
    return {
        "arrays": arrays,
        "max_abs_change": float(np.abs(diff).max()),
        "mean_change": float(diff.mean()),
        "n_negative": int((new_sdf < 0).sum()),
    }


def _write_atomic(npz_path: Path, arrays: dict) -> None:
    # Temp name ends in .npz so np.savez_compressed doesn't append a second
    # extension; os.replace then swaps it in atomically.
    tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npz_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Recompute the sdf channel exactly.")
    p.add_argument("--dataset-dir", type=Path, default=ML_DATASET_DIR,
                   help="ML dataset root containing split subfolders.")
    p.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS),
                   help="Split subfolders to process.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report changes without rewriting any file.")
    args = p.parse_args()

    # Collect every .npz, grouped by NACA code so the analytic curve + KD-tree
    # are built once per airfoil rather than once per case.
    by_code: dict[str, list[Path]] = defaultdict(list)
    for split in args.splits:
        split_dir = args.dataset_dir / split
        if not split_dir.is_dir():
            continue
        for npz_path in sorted(split_dir.glob("*.npz")):
            with np.load(npz_path, allow_pickle=True) as data:
                if "sdf" not in data.files or "naca_code" not in data.files:
                    print(f"[SKIP] {npz_path.name} — missing sdf/naca_code")
                    continue
                code = str(data["naca_code"])
            by_code[code].append(npz_path)

    total = sum(len(v) for v in by_code.values())
    print(f"{total} files across {len(by_code)} NACA codes"
          f"{' (dry run)' if args.dry_run else ''}\n")

    n_done = 0
    worst = (0.0, "")
    for code in sorted(by_code):
        distance_fn = build_naca_distance_fn(code)
        for npz_path in by_code[code]:
            stats = _recompute_sdf(npz_path, distance_fn)
            if stats["n_negative"]:
                print(f"[WARN] {npz_path.name} — {stats['n_negative']} negative "
                      f"sdf values (point inside airfoil?)")
            if not args.dry_run:
                _write_atomic(npz_path, stats["arrays"])
            n_done += 1
            if stats["max_abs_change"] > worst[0]:
                worst = (stats["max_abs_change"], npz_path.name)
            print(f"[{'DRY' if args.dry_run else 'OK '}] {npz_path.name}  "
                  f"max|Δsdf|={stats['max_abs_change']:.5f}  "
                  f"mean Δ={stats['mean_change']:+.5f}")

    print(f"\n{n_done} files {'checked' if args.dry_run else 'rewritten'}.  "
          f"Largest single correction: {worst[0]:.5f} ({worst[1]})")


if __name__ == "__main__":
    main()
