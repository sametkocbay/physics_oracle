"""build_ml_dataset.py — Assemble the ML-ready point-cloud dataset.

For every converged case in dataset/cases/ it:
  1. Loads cell-center coordinates from mesh.h5
  2. Crops to the bounding box configured in configs/postprocess.yaml
  3. Assembles all required channels
  4. Writes one .npz (or .h5) per case into the appropriate split subfolder

Output folder structure (HuggingFace-compatible):
  dataset/ML_dataset/
  ├── train/<case_id>.npz + metadata.csv
  ├── val/  <case_id>.npz + metadata.csv
  ├── test/ <case_id>.npz + metadata.csv
  └── ood/  <case_id>.npz + metadata.csv  (if any ood cases exist)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import yaml

from core.paths import CASES_DIR, ML_DATASET_DIR, POSTPROCESS_CONFIG_PATH


def _load_config() -> dict:
    return yaml.safe_load(POSTPROCESS_CONFIG_PATH.read_text())


def _read_cl_cd(case_dir: Path) -> tuple[float, float]:
    conv_path = case_dir / "convergence.h5"
    if not conv_path.exists():
        return float("nan"), float("nan")
    with h5py.File(conv_path, "r") as h:
        cl = float(h["cl_history"][-1]) if "cl_history" in h else float("nan")
        cd = float(h["cd_history"][-1]) if "cd_history" in h else float("nan")
    return cl, cd


def build_sample(case_dir: Path, bbox: dict, crop: bool, dtype: np.dtype):
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
    x_min, x_max = bbox["x"]
    y_min, y_max = bbox["y"]
    if crop:
        mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
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
    cfg = _load_config()
    ml = cfg["ml_dataset"]
    csv_columns = ml["metadata_csv"]["columns"]
    default_fmt = ml.get("output_format", "npz")
    default_dtype = ml.get("dtype", "float32")
    default_crop = bool(ml.get("crop", True))

    p = argparse.ArgumentParser(description="Build ML-ready point-cloud dataset.")
    p.add_argument("--cases-dir", type=Path, default=CASES_DIR)
    p.add_argument("--output-dir", type=Path, default=ML_DATASET_DIR)
    p.add_argument("--fmt", choices=["npz", "h5"], default=default_fmt)
    p.add_argument("--no-crop", action="store_true",
                   help="Skip bounding-box crop — export all mesh cells")
    p.add_argument("--double", action="store_true",
                   help="Save arrays as float64 instead of float32")
    p.add_argument("--file", metavar="CASE_ID",
                   help="Process a single case by name (e.g. NACA0012_p3.0_2.5e5)")
    args = p.parse_args()

    bbox = ml["bounding_box"]
    crop = default_crop and not args.no_crop
    dtype = np.float64 if (args.double or default_dtype == "float64") else np.float32

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
        result = build_sample(case_dir, bbox=bbox, crop=crop, dtype=dtype)
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
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Metadata CSV -> {csv_path}  ({len(rows)} rows)")

    print(f"\nDone: {n_ok} written, {n_skip} skipped")


if __name__ == "__main__":
    main()
