"""build_ml_dataset.py — Assemble the ML-ready point-cloud dataset.

For every converged case in dataset/cases/ it:
  1. Loads cell-center coordinates from mesh.h5
  2. Crops to the training bounding box
  3. Assembles all required channels
  4. Writes one .npz (or .h5) per case into the appropriate split subfolder

Output folder structure (HuggingFace-compatible):
  ML_dataset/
  ├── train/
  │   ├── <case_id>.npz
  │   └── metadata.csv
  ├── val/
  │   ├── <case_id>.npz
  │   └── metadata.csv
  ├── test/
  │   ├── <case_id>.npz
  │   └── metadata.csv
  └── ood/          (if any ood cases exist)
      ├── <case_id>.npz
      └── metadata.csv

Bounding box (chord = 1, LE at x=0, TE at x=1):
  x ∈ [-1.5, 3.5]   (1.5c in front of LE, 2.5c behind TE)
  y ∈ [-1.5, 1.5]   (1.5c above and below)

Output arrays per file (N = number of cells inside the bounding box):
  x, y             (N,) float32  — cell-center coordinates
  sdf              (N,) float32  — distance to nearest airfoil surface (≥ 0 outside)
  u_init           (N,) float32  — inlet Ux (uniform initial condition)
  v_init           (N,) float32  — inlet Uy (uniform initial condition)
  u, v             (N,) float32  — solved velocity components
  p                (N,) float32  — kinematic pressure
  omega            (N,) float32  — specific dissipation rate
  k                (N,) float32  — turbulent kinetic energy
  nut              (N,) float32  — turbulent viscosity
  reynolds         ()   float32  — Reynolds number (scalar)
  angle_of_attack  ()   float32  — angle of attack in degrees (scalar)
  naca_code        ()   str      — NACA 4-digit code (scalar)
  cl               ()   float32  — lift coefficient (scalar)
  cd               ()   float32  — drag coefficient (scalar)
  is_wall          (N,) uint8    — 1 if cell is adjacent to airfoil wall, else 0
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Bounding box in chord-normalized coordinates
# ---------------------------------------------------------------------------
X_MIN = -1.5
X_MAX = 3.5    # TE is at x=1, so 1 + 2.5 = 3.5
Y_MIN = -1.5
Y_MAX = 1.5

CSV_FIELDNAMES = [
    "file_name", "case_id", "naca_code", "aoa_deg", "reynolds",
    "u_inlet_x", "u_inlet_y", "u_mag", "cl", "cd",
]


def _read_cl_cd(case_dir: Path) -> tuple[float, float]:
    conv_path = case_dir / "convergence.h5"
    if not conv_path.exists():
        return float("nan"), float("nan")
    with h5py.File(conv_path, "r") as h:
        cl = float(h["cl_history"][-1]) if "cl_history" in h else float("nan")
        cd = float(h["cd_history"][-1]) if "cd_history" in h else float("nan")
    return cl, cd


def build_sample(case_dir: Path, crop: bool = True, dtype: np.dtype = np.float32) -> tuple[dict, str] | None:
    """Return (sample_arrays, split_name) or None if the case should be skipped."""
    meta_path = case_dir / "meta.yaml"
    mesh_path = case_dir / "mesh.h5"
    fields_path = case_dir / "fields.h5"

    for p in (meta_path, mesh_path, fields_path):
        if not p.exists():
            return None

    meta = yaml.safe_load(meta_path.read_text())
    if not meta.get("converged", False):
        return None

    re = float(meta["Re"])
    aoa = float(meta.get("aoa_deg", float("nan")))
    naca = str(meta.get("naca_code", ""))
    split = str(meta.get("split", "train"))
    u_inlet = meta["U_inlet"]
    ux_init = float(u_inlet[0])
    vy_init = float(u_inlet[1])

    with h5py.File(mesh_path, "r") as h:
        cell_centers = h["cell_centers"][:]
        boundary_markers = h["boundary_markers"][:]

    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    if crop:
        mask = (x >= X_MIN) & (x <= X_MAX) & (y >= Y_MIN) & (y <= Y_MAX)
        if not mask.any():
            return None
    else:
        mask = np.ones(len(x), dtype=bool)

    with h5py.File(fields_path, "r") as h:
        U = h["U"][:]
        p_arr = h["p"][:].ravel()
        k = h["k"][:].ravel()
        omega = h["omega"][:].ravel()
        nut = h["nut"][:].ravel()
        sdf = h["wall_distance"][:].ravel()

    cl, cd = _read_cl_cd(case_dir)

    n = int(mask.sum())
    sample = {
        "x":             x[mask].astype(dtype),
        "y":             y[mask].astype(dtype),
        "sdf":           sdf[mask].astype(dtype),
        "u_init":        np.full(n, ux_init, dtype=dtype),
        "v_init":        np.full(n, vy_init, dtype=dtype),
        "u":             U[mask, 0].astype(dtype),
        "v":             U[mask, 1].astype(dtype),
        "p":             p_arr[mask].astype(dtype),
        "omega":         omega[mask].astype(dtype),
        "k":             k[mask].astype(dtype),
        "nut":           nut[mask].astype(dtype),
        "reynolds":      dtype(re),
        "angle_of_attack": dtype(aoa),
        "naca_code":     np.array(naca),
        "cl":            dtype(cl),
        "cd":            dtype(cd),
        "is_wall":       (boundary_markers[mask] == 1).astype(np.uint8),
    }
    return sample, split


def write_npz(out_path: Path, sample: dict) -> None:
    np.savez_compressed(out_path, **sample)


def write_h5(out_path: Path, sample: dict) -> None:
    with h5py.File(out_path, "w") as h:
        for key, val in sample.items():
            arr = np.asarray(val)
            if arr.ndim == 0:
                h.attrs[key] = float(arr) if arr.dtype.kind != "U" else str(arr)
            else:
                h.create_dataset(key, data=arr, compression="gzip", compression_opts=4)


def main() -> None:
    p = argparse.ArgumentParser(description="Build ML-ready point-cloud dataset.")
    p.add_argument("--cases-dir", type=Path,
                   default=Path(__file__).resolve().parents[2] / "dataset" / "cases")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parents[2] / "ML_dataset")
    p.add_argument("--fmt", choices=["npz", "h5"], default="npz",
                   help="Output format (default: npz)")
    p.add_argument("--no-crop", action="store_true",
                   help="Skip bounding-box crop — export all mesh cells")
    p.add_argument("--double", action="store_true",
                   help="Save arrays as float64 instead of float32")
    p.add_argument("--file", metavar="CASE_ID",
                   help="Process a single case by name (e.g. NACA0012_p3.0_2.5e5)")
    args = p.parse_args()

    crop = not args.no_crop
    dtype = np.float64 if args.double else np.float32

    # csv_rows_by_split[split] = list of row dicts
    csv_rows_by_split: dict[str, list[dict]] = defaultdict(list)
    n_ok = n_skip = 0

    if args.file:
        candidates = [args.cases_dir / args.file]
    else:
        candidates = sorted(args.cases_dir.iterdir())

    for case_dir in candidates:
        if not case_dir.is_dir():
            print(f"[ERROR] Not a directory: {case_dir}")
            n_skip += 1
            continue
        case_id = case_dir.name
        result = build_sample(case_dir, crop=crop, dtype=dtype)
        if result is None:
            print(f"[SKIP] {case_id}")
            n_skip += 1
            continue

        sample, split = result
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{case_id}.{args.fmt}"
        out_path = split_dir / fname
        if args.fmt == "npz":
            write_npz(out_path, sample)
        else:
            write_h5(out_path, sample)

        meta = yaml.safe_load((case_dir / "meta.yaml").read_text())
        u_inlet = meta["U_inlet"]
        csv_rows_by_split[split].append({
            "file_name":  fname,
            "case_id":    case_id,
            "naca_code":  meta.get("naca_code", ""),
            "aoa_deg":    meta.get("aoa_deg", float("nan")),
            "reynolds":   meta["Re"],
            "u_inlet_x":  float(u_inlet[0]),
            "u_inlet_y":  float(u_inlet[1]),
            "u_mag":      meta.get("U_mag", float("nan")),
            "cl":         float(sample["cl"]),
            "cd":         float(sample["cd"]),
        })

        n_pts = int(sample["x"].shape[0])
        print(f"[OK]   {case_id} -> {split}/{fname}  ({n_pts:,} points)")
        n_ok += 1

    for split, rows in csv_rows_by_split.items():
        csv_path = args.output_dir / split / "metadata.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Metadata CSV -> {csv_path}  ({len(rows)} rows)")

    print(f"\nDone: {n_ok} written, {n_skip} skipped")


if __name__ == "__main__":
    main()
