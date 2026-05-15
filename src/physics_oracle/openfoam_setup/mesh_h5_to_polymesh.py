"""Convert a saved ``mesh.h5`` back into an OpenFOAM ``constant/polyMesh/`` tree.

The saved h5 stores the 2D-collapsed mesh produced by extract.extract_case:
points (P, 2), connectivity (Nc, 4), boundary_markers (Nc,), cell_centers (Nc, 2).

OpenFOAM cases need a 3D polyMesh.  We extrude one cell in z (dz = 0.01 by
default, matching meshing/gmsh_mesh.py), build hex cells, deduplicate
side faces from shared 2D edges, and classify each boundary face by the
per-cell boundary_marker (1=airfoilWalls, 2=inlet, 3=outlet, 4=top|bottom).

The output polyMesh is suitable as input to foamRun.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from .of_writer import (
    write_boundary,
    write_face_list,
    write_label_list,
    write_points,
)


# Patch types (matches meshing/gmsh_mesh.py:PATCH_TYPES).
_PATCH_TYPES = {
    "frontAndBack":  "empty",
    "airfoilWalls":  "wall",
    "inlet":         "patch",
    "outlet":        "patch",
    "top":           "patch",
    "bottom":        "patch",
}

# Order of boundary patches in the boundary file (matches dataset generation).
_PATCH_ORDER = ("frontAndBack", "airfoilWalls", "inlet", "outlet", "top", "bottom")


# ---------------------------------------------------------------------------
# Quad orientation
# ---------------------------------------------------------------------------

def _signed_area(verts: np.ndarray) -> float:
    """Shoelace formula for a quad ordered (v0, v1, v2, v3) in 2D."""
    x = verts[:, 0]
    y = verts[:, 1]
    return 0.5 * (
        x[0] * (y[1] - y[3])
        + x[1] * (y[2] - y[0])
        + x[2] * (y[3] - y[1])
        + x[3] * (y[0] - y[2])
    )


def _normalize_ccw(connectivity: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Reverse any quad whose signed area is negative so all quads are CCW."""
    conn = connectivity.copy()
    for ci in range(len(conn)):
        verts = points2[conn[ci]]
        if _signed_area(verts) < 0:
            conn[ci] = conn[ci, ::-1]
    return conn


# ---------------------------------------------------------------------------
# Patch classification
# ---------------------------------------------------------------------------

def _classify_boundary_edge(
    marker: int, edge_midpoint: np.ndarray
) -> str | None:
    """Map a cell's boundary_marker (and the edge midpoint, used to split
    marker=4 into top vs bottom) onto a patch name.

    Returns None for marker=0 — interior cell, not expected to appear here.
    """
    if marker == 1:
        return "airfoilWalls"
    if marker == 2:
        return "inlet"
    if marker == 3:
        return "outlet"
    if marker == 4:
        return "top" if edge_midpoint[1] >= 0.0 else "bottom"
    return None


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

def _build_directed_edges(connectivity: np.ndarray):
    """Return a dict mapping each undirected edge frozenset({va, vb}) to a
    list of (cell_id, directed (va, vb)) entries.  In a well-formed mesh
    each edge has 1 (boundary) or 2 (internal) entries."""
    edges: dict[frozenset, list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    for ci, quad in enumerate(connectivity):
        for k in range(4):
            va = int(quad[k])
            vb = int(quad[(k + 1) % 4])
            edges[frozenset((va, vb))].append((ci, (va, vb)))
    return edges


def write_polymesh_from_h5(
    mesh_h5: Path,
    polymesh_dir: Path,
    *,
    dz: float = 0.01,
) -> dict:
    """Convert a saved mesh.h5 → a polyMesh/ directory tree.

    Returns a dict with face counts (internal + per-patch) for inspection.
    """
    mesh_h5 = Path(mesh_h5)
    polymesh_dir = Path(polymesh_dir)
    polymesh_dir.mkdir(parents=True, exist_ok=True)

    # ----- load -----
    with h5py.File(mesh_h5, "r") as h:
        points2 = np.asarray(h["points"][:], dtype=np.float64)         # (P, 2)
        connectivity = np.asarray(h["connectivity"][:], dtype=np.int64)  # (Nc, 4)
        boundary_markers = np.asarray(h["boundary_markers"][:], dtype=np.int8)  # (Nc,)

    n_p = len(points2)
    n_c = len(connectivity)

    # Normalize quad orientation so all cells have positive signed area.
    connectivity = _normalize_ccw(connectivity, points2)

    # ----- 3D points (extrude in z) -----
    pts3 = np.zeros((2 * n_p, 3), dtype=np.float64)
    pts3[:n_p, :2] = points2                       # z = 0
    pts3[n_p:, :2] = points2
    pts3[n_p:, 2] = dz                              # z = dz

    # ----- edge → list[(cell, directed)] -----
    edges = _build_directed_edges(connectivity)

    # ----- classify side faces -----
    internal_faces: list[tuple[int, int, list[int]]] = []   # (owner, neighbour, face_verts)
    boundary_side_faces: dict[str, list[tuple[int, list[int]]]] = {
        p: [] for p in _PATCH_ORDER
    }

    for edge, members in edges.items():
        if len(members) == 1:
            # Boundary side face
            c, (va, vb) = members[0]
            edge_mid = 0.5 * (points2[va] + points2[vb])
            patch = _classify_boundary_edge(int(boundary_markers[c]), edge_mid)
            if patch is None:
                # Fallback: classify by geometry
                patch = _fallback_patch_from_geometry(edge_mid, points2)
            face_verts = [va, vb, vb + n_p, va + n_p]
            boundary_side_faces[patch].append((c, face_verts))
        elif len(members) == 2:
            # Internal face
            (c0, (va0, vb0)), (c1, (va1, vb1)) = members
            owner, neighbour = sorted((c0, c1))
            if owner == c0:
                va, vb = va0, vb0
            else:
                va, vb = va1, vb1
            # Face viewed from owner: (va, vb, vb_top, va_top) — outward normal
            # from owner, i.e., toward neighbour.
            face_verts = [va, vb, vb + n_p, va + n_p]
            internal_faces.append((owner, neighbour, face_verts))
        else:
            raise ValueError(
                f"Edge {tuple(edge)} appears in {len(members)} cells — "
                f"mesh is not 2-manifold."
            )

    # Sort internal faces by (owner, neighbour) — OF canonical order.
    internal_faces.sort(key=lambda x: (x[0], x[1]))

    # ----- front / back faces -----
    # Front (z=0): outward normal = -z, so reverse the CCW quad to make it CW.
    # Back (z=dz): outward normal = +z, keep CCW.
    front_faces = [(ci, list(connectivity[ci, ::-1])) for ci in range(n_c)]
    back_faces = [(ci, [int(v) + n_p for v in connectivity[ci]]) for ci in range(n_c)]

    # ----- assemble final face / owner / neighbour arrays -----
    all_faces: list[list[int]] = []
    all_owners: list[int] = []
    all_neighbours: list[int] = []

    # Internal faces first.
    for owner, neighbour, verts in internal_faces:
        all_faces.append(verts)
        all_owners.append(owner)
        all_neighbours.append(neighbour)

    # Boundary faces grouped by patch.
    patch_entries = []
    start = len(all_faces)
    for patch_name in _PATCH_ORDER:
        if patch_name == "frontAndBack":
            patch_face_list = front_faces + back_faces
        else:
            patch_face_list = boundary_side_faces[patch_name]
        n_patch = len(patch_face_list)
        if n_patch == 0:
            continue
        for owner, verts in patch_face_list:
            all_faces.append(verts)
            all_owners.append(owner)
        patch_entries.append({
            "name": patch_name,
            "type": _PATCH_TYPES[patch_name],
            "nFaces": n_patch,
            "startFace": start,
        })
        start += n_patch

    # ----- write files -----
    write_points(polymesh_dir / "points", pts3)
    write_face_list(polymesh_dir / "faces", all_faces)
    write_label_list(polymesh_dir / "owner", all_owners, obj="owner")
    write_label_list(polymesh_dir / "neighbour", all_neighbours, obj="neighbour")
    write_boundary(polymesh_dir / "boundary", patch_entries)

    # ----- summary -----
    summary = {
        "n_cells": n_c,
        "n_points": 2 * n_p,
        "n_internal_faces": len(internal_faces),
        "n_boundary_faces": sum(p["nFaces"] for p in patch_entries),
        "patches": {p["name"]: p["nFaces"] for p in patch_entries},
    }
    return summary


# ---------------------------------------------------------------------------
# Fallback geometry classifier (defensive — kicks in only if the per-cell
# marker is 0 yet we found a boundary edge for that cell)
# ---------------------------------------------------------------------------

def _fallback_patch_from_geometry(
    edge_mid: np.ndarray, all_points: np.ndarray
) -> str:
    x_min, x_max = float(all_points[:, 0].min()), float(all_points[:, 0].max())
    y_min, y_max = float(all_points[:, 1].min()), float(all_points[:, 1].max())
    span_x = x_max - x_min
    span_y = y_max - y_min
    eps = 0.01 * max(span_x, span_y)
    x, y = float(edge_mid[0]), float(edge_mid[1])
    if x < x_min + eps:
        return "inlet"
    if x > x_max - eps:
        return "outlet"
    if y > y_max - eps:
        return "top"
    if y < y_min + eps:
        return "bottom"
    # Otherwise assume it's near the airfoil.
    return "airfoilWalls"
