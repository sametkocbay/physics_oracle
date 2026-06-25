"""Exact distance from 2D points to the analytic NACA airfoil surface.

The dataset's ``sdf`` / ``wall_distance`` channel used to be a nearest-*vertex*
query against the mesh boundary points (``cKDTree.query(cell_centers)``).  That
carries two errors: it over-estimates between vertices (point-to-vertex instead
of point-to-edge), and the mesh vertices are themselves a faceted approximation
of the airfoil.

Here we instead compute distance to the *analytic* NACA 4-digit curve.  The
airfoil in this project sits unrotated with its chord on the x-axis from
(0, 0) to (chord, 0) — angle of attack is carried by the inlet velocity, not the
geometry — so the curve generated from the NACA code shares the cell-center
coordinate frame directly, with no transform needed.

Accuracy (vs. an 800k-segment reference): ~1e-9 mean, ~1e-5 worst case at a
handful of wake points behind the blunt trailing edge — at or below the float32
precision the channel is stored at, and ~3 orders of magnitude better than the
old nearest-vertex query.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from physics_oracle.core.envelope import CHORD
from physics_oracle.geometry.naca import naca4_coordinates

# Defaults — see module docstring for the accuracy these buy.
DEFAULT_N_POINTS = 80_000
DEFAULT_K = 24


def _segment_distance_sq(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared distance from points ``p`` to segments ``a``->``b`` (all (M, 2))."""
    ab = b - a
    ap = p - a
    denom = np.einsum("ij,ij->i", ab, ab)
    safe = np.where(denom > 0.0, denom, 1.0)
    t = np.clip(np.einsum("ij,ij->i", ap, ab) / safe, 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = p - proj
    return np.einsum("ij,ij->i", d, d)


def build_naca_distance_fn(
    naca_code: str,
    chord: float = CHORD,
    n_points: int = DEFAULT_N_POINTS,
    k: int = DEFAULT_K,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a reusable distance function for one NACA airfoil.

    The analytic curve and its segment KD-tree are built once here; the returned
    callable can then be applied to many point clouds (e.g. every case sharing
    the same NACA code) without rebuilding them.

    Returns a callable ``fn(points) -> distances`` where ``points`` is (M, 2)
    and the result is (M,) float64 unsigned distance to the closed airfoil
    contour (upper/lower surfaces + the blunt open-TE base).
    """
    # Analytic curve, ordered TE-upper -> LE -> TE-lower.  Rolling by one closes
    # the contour: the final segment spans the blunt trailing-edge base.
    curve = naca4_coordinates(naca_code, n_points=n_points, chord=chord)
    seg_a = curve
    seg_b = np.roll(curve, -1, axis=0)
    tree = cKDTree(0.5 * (seg_a + seg_b))
    k_eff = min(k, len(seg_a))

    def distance(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"points must be (M, 2), got {points.shape}")
        _, idx = tree.query(points, k=k_eff)
        if idx.ndim == 1:  # k_eff == 1
            idx = idx[:, None]
        best = np.full(len(points), np.inf)
        for j in range(idx.shape[1]):
            sidx = idx[:, j]
            best = np.minimum(
                best, _segment_distance_sq(points, seg_a[sidx], seg_b[sidx])
            )
        return np.sqrt(best)

    return distance


def naca_surface_distance(
    points: np.ndarray,
    naca_code: str,
    chord: float = CHORD,
    n_points: int = DEFAULT_N_POINTS,
    k: int = DEFAULT_K,
) -> np.ndarray:
    """Unsigned distance from 2D ``points`` to the analytic NACA airfoil surface.

    One-shot convenience wrapper around :func:`build_naca_distance_fn`.  When
    processing many cases that share a NACA code, build the function once with
    :func:`build_naca_distance_fn` instead and reuse it.

    Parameters
    ----------
    points
        (M, 2) array of query coordinates, in the airfoil's own frame
        (chord along +x from the origin).
    naca_code
        4-digit NACA code, e.g. ``"0012"`` or ``"2412"``.
    chord
        Chord length the airfoil was generated with (1.0 throughout this repo).
    n_points, k
        Polyline density and candidate-segment count — see
        :func:`build_naca_distance_fn` and the module docstring.
    """
    return build_naca_distance_fn(naca_code, chord, n_points, k)(points)
