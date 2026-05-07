"""visualize_npz.py — Interpolated field images from ML_dataset .npz files.

Loads one or more .npz files produced by build_ml_dataset.py, interpolates
u, v, p, and sdf onto a regular grid, and saves a 2×2 panel PNG.

Usage:
    python utils/visualize_npz.py ML_dataset/NACA2412_p5.0_3.0e5.npz
    python utils/visualize_npz.py ML_dataset/*.npz --out plots/
    python utils/visualize_npz.py ML_dataset/NACA2412_p5.0_3.0e5.npz --resolution 800
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator

# Must match the bounding box in build_ml_dataset.py
X_MIN, X_MAX = -1.5, 3.5
Y_MIN, Y_MAX = -1.5, 1.5

# (dataset key, panel title, colormap, is_diverging)
PANELS = [
    ("u",   r"$u$ (velocity x)",      "RdBu_r",  True),
    ("v",   r"$v$ (velocity y)",      "RdBu_r",  True),
    ("p",   r"$p$ (pressure)",        "coolwarm", True),
    ("sdf", r"SDF (wall distance)",   "viridis",  False),
]

# SDF threshold below which cells are considered inside/on the airfoil
_WALL_SDF = 0.005


def _color_limits(data: np.ndarray, diverging: bool, center: float) -> tuple[float, float]:
    """2nd–98th percentile limits; diverging maps are symmetric around center."""
    lo = float(np.nanpercentile(data, 2))
    hi = float(np.nanpercentile(data, 98))
    if diverging:
        half = max(abs(lo - center), abs(hi - center))
        return center - half, center + half
    return lo, hi


def visualize(npz_path: Path, out_dir: Path, resolution: int, dpi: int) -> Path:
    data = np.load(npz_path)

    x = data["x"].astype(np.float64)
    y = data["y"].astype(np.float64)
    re = float(data["reynolds"])
    u_inf = float(data["u_init"][0])
    v_inf = float(data["v_init"][0])

    # Build the Delaunay triangulation once for all fields
    field_matrix = np.column_stack([data[key].astype(np.float64) for key, *_ in PANELS])
    interp = LinearNDInterpolator(np.column_stack([x, y]), field_matrix)

    # Regular grid (aspect ratio matches bounding box)
    nx = resolution
    ny = int(resolution * (Y_MAX - Y_MIN) / (X_MAX - X_MIN))
    xi = np.linspace(X_MIN, X_MAX, nx)
    yi = np.linspace(Y_MIN, Y_MAX, ny)
    Xi, Yi = np.meshgrid(xi, yi)

    grid_vals = interp(Xi, Yi)          # (ny, nx, n_fields)

    # Mask the airfoil solid region using the interpolated SDF
    sdf_idx = next(i for i, (k, *_) in enumerate(PANELS) if k == "sdf")
    sdf_grid = grid_vals[:, :, sdf_idx]
    airfoil_mask = sdf_grid < _WALL_SDF

    # Freestream centres for diverging colormaps
    centers = {"u": u_inf, "v": v_inf, "p": 0.0, "sdf": 0.0}

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)
    fig.suptitle(
        f"{npz_path.stem}   Re = {re:.3g}   "
        rf"$u_\infty$ = ({u_inf:.2f}, {v_inf:.2f})",
        fontsize=11,
    )

    for ax, (fi, (key, title, cmap, diverging)) in zip(axes.flat, enumerate(PANELS)):
        img = grid_vals[:, :, fi].copy()
        img[airfoil_mask] = np.nan

        finite = img[np.isfinite(img)]
        vmin, vmax = _color_limits(finite, diverging, centers[key])

        im = ax.imshow(
            img,
            origin="lower",
            extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            interpolation="bilinear",
        )
        # Airfoil outline from SDF = _WALL_SDF contour
        ax.contour(Xi, Yi, sdf_grid, levels=[_WALL_SDF], colors="k", linewidths=0.8)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x / c")
        ax.set_ylabel("y / c")
        # Leading / trailing edge guides
        ax.axvline(0.0, color="gray", lw=0.5, ls="--", alpha=0.6)
        ax.axvline(1.0, color="gray", lw=0.5, ls="--", alpha=0.6)
        fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)

    out_path = out_dir / f"{npz_path.stem}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def visualize_points(npz_path: Path, out_dir: Path, dpi: int) -> Path:
    data = np.load(npz_path)

    x = data["x"].astype(np.float64)
    y = data["y"].astype(np.float64)
    is_wall = data["is_wall"].astype(bool)
    re = float(data["reynolds"])

    fluid_mask = ~is_wall

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    fig.suptitle(
        f"{npz_path.stem}   Re = {re:.3g}   "
        f"fluid={fluid_mask.sum():,}  wall={is_wall.sum():,}",
        fontsize=11,
    )

    ax.scatter(x[fluid_mask], y[fluid_mask], s=0.5, c="#4c9be8", linewidths=0,
               rasterized=True, label=f"fluid ({fluid_mask.sum():,})")
    ax.scatter(x[is_wall], y[is_wall], s=1.5, c="#e84c4c", linewidths=0,
               rasterized=True, label=f"wall ({is_wall.sum():,})")

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.axvline(0.0, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax.axvline(1.0, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax.legend(loc="upper right", markerscale=6, fontsize=9, framealpha=0.8)

    out_path = out_dir / f"{npz_path.stem}_points.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize ML_dataset .npz files.")
    p.add_argument("files", nargs="+", type=Path,
                   help=".npz file(s) to visualize")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: same folder as each .npz)")
    p.add_argument("--resolution", type=int, default=600,
                   help="Grid width in pixels; height auto-scales to match aspect ratio (default: 600)")
    p.add_argument("--dpi", type=int, default=150,
                   help="Output image DPI (default: 150)")
    p.add_argument("--points-only", action="store_true",
                   help="Only render the point-cloud plot (skip interpolated fields)")
    p.add_argument("--no-points", action="store_true",
                   help="Skip the point-cloud plot")
    args = p.parse_args()

    for npz_path in sorted(args.files):
        out_dir = args.out if args.out else npz_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  {npz_path.name} ...", end=" ", flush=True)
        parts = []
        if not args.points_only:
            out = visualize(npz_path, out_dir, args.resolution, args.dpi)
            parts.append(out.name)
        if not args.no_points:
            out_pts = visualize_points(npz_path, out_dir, args.dpi)
            parts.append(out_pts.name)
        print("-> " + ", ".join(parts))


if __name__ == "__main__":
    main()
