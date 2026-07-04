"""§6 — Parse a converged OpenFOAM case and write the four HDF5 artifacts.

Outputs (per case):
  fields.h5      — U, p, k, omega, nut, wall_distance
  mesh.h5        — cell_centers, points, connectivity, boundary_markers
  geometry.h5    — airfoil_coordinates
  convergence.h5 — residual_history, cl_history, cd_history, y_plus, …

The simulation is 2D-by-extrusion (1 cell in z); we collapse the 3D mesh to
its z=0 face for the dataset arrays.
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np

from physics_oracle.core.case_spec import parse_case_id
from physics_oracle.core.logging import setup_logging
from physics_oracle.geometry.distance import naca_surface_distance
from physics_oracle.geometry.naca import naca4_coordinates
from physics_oracle.openfoam_setup.runner import detect_convergence, parse_solver_log


def _first_converged_iter(history: dict[str, list[float]]) -> int:
    """Return the first iteration at which all fields' residual ratios had
    dropped ≥ 4 orders below the initial value (§5.7).  Returns 0 if never."""
    import math
    fields = list(history.keys())
    initials: dict[str, float] = {}
    for f in fields:
        vals = [v for v in history[f] if v == v and v > 0]
        if vals:
            initials[f] = vals[0]
    n = max(len(v) for v in history.values()) if history else 0
    for i in range(n):
        ok = True
        for f in fields:
            if f not in initials:
                continue
            v = history[f][i] if i < len(history[f]) else None
            if v is None or v != v or v <= 0:
                ok = False
                break
            if math.log10(initials[f] / v) < 4.0:
                ok = False
                break
        if ok:
            return i + 1
    return 0

LOG = setup_logging()


# ---------------------------------------------------------------------------
# OpenFOAM file parsing
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    """Read OpenFOAM file, transparently handling .gz."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return f.read()
    if path.exists():
        return path.read_text()
    if path.with_suffix(path.suffix + ".gz").exists():
        with gzip.open(str(path) + ".gz", "rt") as f:
            return f.read()
    raise FileNotFoundError(path)


_NUM_RE = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"


def _extract_wall_yplus(path: Path) -> np.ndarray:
    """Pull the airfoilWalls patch values from a yPlus volScalarField file."""
    text = _read_text(path)
    text_clean = re.sub(r"//[^\n]*", "", text)
    text_clean = re.sub(r"/\*.*?\*/", "", text_clean, flags=re.DOTALL)
    m = re.search(r"airfoilWalls\s*\{(.*?)\}", text_clean, flags=re.DOTALL)
    if not m:
        return np.array([])
    body = m.group(1)
    # value can be `uniform N;` or `nonuniform List<scalar> N (...)`
    mu = re.search(r"value\s+uniform\s+([-\d.eE+]+)\s*;", body)
    if mu:
        return np.array([float(mu.group(1))])
    mn = re.search(r"value\s+nonuniform\s+List<scalar>\s*(\d+)?\s*\(([^)]*)\)",
                   body, flags=re.DOTALL)
    if mn:
        nums = re.findall(_NUM_RE, mn.group(2))
        return np.asarray([float(x) for x in nums], dtype=float)
    return np.array([])


def parse_internal_field(text: str) -> np.ndarray:
    """Parse the `internalField` block from an OpenFOAM volField file.

    Returns a (N,) array for scalar fields, or (N, 3) for vector/tensor fields.
    Handles uniform and nonuniform List<...> syntax.
    """
    # Strip comments to keep regex sane
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    m = re.search(r"internalField\s+([^;]+);", text, flags=re.DOTALL)
    if not m:
        raise ValueError("internalField not found")
    body = m.group(1).strip()

    # uniform scalar / vector
    if body.startswith("uniform"):
        rest = body[len("uniform"):].strip()
        if rest.startswith("("):
            vals = [float(x) for x in re.findall(_NUM_RE, rest)]
            return np.array(vals, dtype=float).reshape(1, -1)
        return np.array([float(rest)], dtype=float)

    # nonuniform List<scalar> N (...)  OR List<vector> N ((..) (..) ...)
    m2 = re.search(r"nonuniform\s+List<(\w+)>\s*(\d+)?\s*\((.*?)\)\s*$",
                   body, flags=re.DOTALL)
    if not m2:
        raise ValueError("Could not parse nonuniform List in internalField")
    typ = m2.group(1)
    payload = m2.group(3)
    if typ == "scalar":
        nums = re.findall(_NUM_RE, payload)
        return np.array(nums, dtype=float)
    # vector / tensor — split into tuples (a b c)
    tuples = re.findall(r"\(([^)]*)\)", payload)
    rows = [[float(x) for x in re.findall(_NUM_RE, t)] for t in tuples]
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# polyMesh parsing
# ---------------------------------------------------------------------------

def _strip_foam_header(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    m = re.search(r"FoamFile\s*\{[^}]*\}", text, flags=re.DOTALL)
    if m:
        text = text[m.end():]
    return text


def parse_points_file(path: Path) -> np.ndarray:
    text = _strip_foam_header(_read_text(path))
    tuples = re.findall(r"\(([^)]*)\)", text)
    if not tuples:
        raise ValueError(f"No points found in {path}")
    rows = [[float(x) for x in re.findall(_NUM_RE, t)] for t in tuples]
    # First match might be the count (no, count is bare integer) — drop empty
    rows = [r for r in rows if len(r) == 3]
    return np.asarray(rows, dtype=float)


def parse_faces_file(path: Path) -> list[list[int]]:
    """Each face = list of point indices (variable length)."""
    text = _strip_foam_header(_read_text(path))
    # Faces look like: 4(p0 p1 p2 p3) or 3(p0 p1 p2)
    faces = []
    for m in re.finditer(r"(\d+)\(([^)]*)\)", text):
        n = int(m.group(1))
        verts = [int(x) for x in m.group(2).split()]
        if len(verts) == n:
            faces.append(verts)
    return faces


def parse_int_list(path: Path) -> np.ndarray:
    text = _strip_foam_header(_read_text(path))
    # The first big "N\n(\n...)\n" block — pull all integers between ( ... )
    m = re.search(r"\(\s*(.*?)\s*\)", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No list in {path}")
    nums = re.findall(r"-?\d+", m.group(1))
    return np.asarray([int(x) for x in nums], dtype=np.int64)


def parse_boundary_file(path: Path) -> list[dict]:
    """Return [{name, type, nFaces, startFace}, ...]."""
    text = _read_text(path)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    m = re.search(r"FoamFile\s*\{[^}]*\}", text, flags=re.DOTALL)
    if m:
        text = text[m.end():]
    # Each patch: name { type X; ...  nFaces N; startFace S; }
    out = []
    for pm in re.finditer(
        r"(\w+)\s*\{([^}]*)\}", text, flags=re.DOTALL
    ):
        name = pm.group(1)
        body = pm.group(2)
        if "nFaces" not in body or "startFace" not in body:
            continue
        nF = int(re.search(r"nFaces\s+(\d+)", body).group(1))
        sF = int(re.search(r"startFace\s+(\d+)", body).group(1))
        ty = re.search(r"type\s+(\w+)", body)
        out.append({
            "name": name,
            "type": ty.group(1) if ty else "patch",
            "nFaces": nF,
            "startFace": sF,
        })
    return out


# ---------------------------------------------------------------------------
# Cell-from-face reconstruction (2D collapse)
# ---------------------------------------------------------------------------

def build_cells(faces: list[list[int]], owner: np.ndarray,
                neighbour: np.ndarray, n_cells: int) -> list[set[int]]:
    """Return a list of vertex-id sets per cell."""
    cells: list[set[int]] = [set() for _ in range(n_cells)]
    for fi, fverts in enumerate(faces):
        cells[owner[fi]].update(fverts)
        if fi < len(neighbour):
            cells[neighbour[fi]].update(fverts)
    return cells


def project_2d_mesh(points3: np.ndarray, faces: list[list[int]],
                    owner: np.ndarray, neighbour: np.ndarray,
                    n_cells: int, dz_tol: float = 1e-6) -> dict:
    """Collapse 1-cell-thick extruded 3D mesh to its z=0 face.

    Strategy: each 3D cell's "front" face is the polygon at z≈0.  For each
    cell, find the face whose vertices all have z≈z_min — that gives the 2D
    polygon and the 2D cell is the polygon itself.
    """
    z = points3[:, 2]
    z_min = float(z.min())
    z_front_mask = np.abs(z - z_min) < dz_tol
    front_pts3 = np.where(z_front_mask)[0]                 # 3D point indices on front

    # 2D point set: the front-plane points in (x, y)
    points2 = points3[z_front_mask][:, :2]
    # Map: 3D point id → 2D point id
    pt3_to_2 = -np.ones(len(points3), dtype=np.int64)
    pt3_to_2[front_pts3] = np.arange(len(front_pts3))

    # For each cell, find face whose vertices are all on the front plane
    connectivity: list[list[int]] = []
    for ci in range(n_cells):
        cell_face_ids = []
        # Iterate faces only once: gather list of cell-face links
        # (for performance we precomputed below)
        connectivity.append([])

    cell_to_faces: list[list[int]] = [[] for _ in range(n_cells)]
    for fi in range(len(faces)):
        cell_to_faces[owner[fi]].append(fi)
        if fi < len(neighbour):
            cell_to_faces[neighbour[fi]].append(fi)

    n_with_quad = 0
    n_with_tri = 0
    for ci in range(n_cells):
        polygon: list[int] | None = None
        for fi in cell_to_faces[ci]:
            verts = faces[fi]
            if all(z_front_mask[v] for v in verts):
                polygon = verts
                break
        if polygon is None:
            connectivity[ci] = []
            continue
        # Map 3D verts → 2D verts
        polygon2 = [int(pt3_to_2[v]) for v in polygon]
        connectivity[ci] = polygon2
        if len(polygon2) == 4:
            n_with_quad += 1
        elif len(polygon2) == 3:
            n_with_tri += 1

    # Pad to a rectangular array with -1
    max_verts = max((len(c) for c in connectivity), default=4)
    conn_arr = -np.ones((n_cells, max_verts), dtype=np.int64)
    for ci, c in enumerate(connectivity):
        conn_arr[ci, :len(c)] = c
    LOG.info("2D mesh: %d cells (quads=%d, tris=%d), %d points",
             n_cells, n_with_quad, n_with_tri, len(points2))
    return {"points2": points2, "connectivity": conn_arr,
            "pt3_to_2": pt3_to_2, "z_front_mask": z_front_mask}


def boundary_markers(faces: list[list[int]], owner: np.ndarray,
                     n_cells: int, patches: list[dict]) -> np.ndarray:
    """Per-cell marker (0=interior, 1=wall, 2=inlet, 3=outlet, 4=farfield)."""
    NAME_TO_CODE = {
        "airfoilWalls": 1, "airfoil": 1,
        "inlet": 2,
        "outlet": 3,
        "top": 4, "bottom": 4, "farfield": 4,
    }
    marker = np.zeros(n_cells, dtype=np.int8)
    for p in patches:
        code = NAME_TO_CODE.get(p["name"])
        if code is None:
            continue
        for fi in range(p["startFace"], p["startFace"] + p["nFaces"]):
            if fi < len(owner):
                ci = owner[fi]
                # Wall trumps any softer label
                if code == 1 or marker[ci] == 0:
                    marker[ci] = code
    return marker


# ---------------------------------------------------------------------------
# Wall-surface table (Option B: a per-face table on the airfoil boundary)
# ---------------------------------------------------------------------------

def _parse_patch_vector_field(path: Path, patch_name: str) -> np.ndarray:
    """Return the (M, 3) boundaryField values of a volVectorField on `patch_name`.

    Handles both `value nonuniform List<vector> M (...)` and the degenerate
    `value uniform (a b c)` form.  Returns an empty array if not found.
    """
    text = _read_text(path)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # The patch sub-dict has no nested braces, so a non-greedy grab to the first
    # closing brace isolates it (the vector list uses parentheses, not braces).
    m = re.search(patch_name + r"\s*\{(.*?)\}", text, flags=re.DOTALL)
    if not m:
        return np.empty((0, 3))
    body = m.group(1)
    vec = rf"\(\s*({_NUM_RE})\s+({_NUM_RE})\s+({_NUM_RE})\s*\)"
    trips = re.findall(vec, body)
    if trips:
        return np.asarray(trips, dtype=float)
    mu = re.search(r"uniform\s*" + vec, body)
    if mu:
        return np.asarray([mu.groups()], dtype=float)
    return np.empty((0, 3))


def wall_surface_table(faces: list[list[int]], owner: np.ndarray,
                       patches: list[dict], points3: np.ndarray,
                       cell_centers2: np.ndarray, p_internal: np.ndarray,
                       shear_vecs: np.ndarray) -> dict:
    """Build the airfoil-surface table (one row per wall face).

    All arrays are ordered by patch face, which is exactly the order OpenFOAM
    writes patch boundaryField values, so `shear_vecs[i]` corresponds to the
    i-th collected wall face.  Returns kinematic quantities (consistent with the
    solver's kinematic `p` and the [0 2 -2] wallShearStress units).

    Keys (M = number of wall faces):
      wall_xy     (M, 2)  face-center coordinates, same frame as cell_centers
      wall_normal (M, 2)  unit normal, pointing from the surface into the fluid
      wall_shear  (M, 2)  kinematic wall shear stress vector (tau_w / rho)
      wall_p      (M,)    kinematic surface pressure (= adjacent cell p; the
                          airfoil p BC is zeroGradient, so p_wall = p_owner)
      wall_length (M,)    2D edge length of the face (for surface integration)
      wall_cell   (M,)    owner cell index (link back to the volume cloud)
    """
    xy, normal, length, wp, cell_idx = [], [], [], [], []
    for pch in patches:
        if pch["name"] not in ("airfoilWalls", "airfoil"):
            continue
        for fi in range(pch["startFace"], pch["startFace"] + pch["nFaces"]):
            if fi >= len(owner):
                continue
            verts_xy = points3[faces[fi]][:, :2]
            fc = verts_xy.mean(axis=0)
            ci = int(owner[fi])
            oc = cell_centers2[ci]
            n = oc - fc
            nn = np.linalg.norm(n)
            n = n / nn if nn > 0 else np.array([0.0, 0.0])
            # 2D edge length: front/back extrusion vertices share xy, so the max
            # pairwise xy separation is the edge length.
            d = np.linalg.norm(verts_xy[:, None, :] - verts_xy[None, :, :], axis=-1)
            xy.append(fc)
            normal.append(n)
            length.append(float(d.max()))
            wp.append(float(p_internal[ci]))
            cell_idx.append(ci)

    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    normal = np.asarray(normal, dtype=float).reshape(-1, 2)
    length = np.asarray(length, dtype=float)
    wp = np.asarray(wp, dtype=float)
    cell_idx = np.asarray(cell_idx, dtype=np.int64)

    m = xy.shape[0]
    if shear_vecs.shape[0] == m:
        shear = shear_vecs[:, :2].astype(float)
    else:
        # Length mismatch (missing or stale wallShearStress) — store NaNs so the
        # table stays aligned rather than silently dropping the field.
        LOG.warning("wall shear count %d != %d wall faces; storing NaN shear",
                    shear_vecs.shape[0], m)
        shear = np.full((m, 2), np.nan)

    return {
        "wall_xy": xy,
        "wall_normal": normal,
        "wall_shear": shear,
        "wall_p": wp,
        "wall_length": length,
        "wall_cell": cell_idx,
    }


# ---------------------------------------------------------------------------
# Run OF post-processing helpers
# ---------------------------------------------------------------------------

def run_postprocess(of_case_dir: Path, func: str, timeout: int = 600,
                    solver: str | None = None) -> bool:
    """Run `foamPostProcess -func <func> -latestTime` (OF13 Foundation).

    ``solver`` constructs a solver module so model-dependent function objects
    (e.g. ``wallShearStress``, which needs ``nuEff`` from the turbulence model)
    can find the momentum-transport model in the database.
    """
    solver_opt = f"-solver {solver} " if solver else ""
    cmd = (
        "set -e && "
        f"if [ -z \"${{WM_PROJECT_DIR:-}}\" ]; then source {os.environ.get('OPENFOAM_BASHRC', '/opt/openfoam13/etc/bashrc')}; fi && "
        f"cd {of_case_dir.resolve()} && foamPostProcess {solver_opt}-func {func} -latestTime"
    )
    log = of_case_dir / f"postProcess.{func}.log"
    with log.open("w") as f:
        proc = subprocess.run(
            ["bash", "-lc", cmd], stdout=f, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
    return proc.returncode == 0


def latest_time_dir(of_case_dir: Path) -> Path:
    times = []
    for entry in of_case_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            times.append((float(entry.name), entry))
        except ValueError:
            continue
    if not times:
        raise FileNotFoundError(f"No time directories in {of_case_dir}")
    times.sort()
    # Skip 0 if a converged result exists
    nonzero = [t for t in times if t[0] > 0]
    return (nonzero[-1] if nonzero else times[-1])[1]


# ---------------------------------------------------------------------------
# forceCoeffs file parsing
# ---------------------------------------------------------------------------

def parse_force_coeffs(of_case_dir: Path) -> dict:
    """Read postProcessing/forceCoeffs1/<t>/coefficient.dat (OF13) or forceCoeffs.dat."""
    candidates = sorted((of_case_dir / "postProcessing" / "forceCoeffs1").glob("*/coefficient.dat"))
    if not candidates:
        candidates = sorted((of_case_dir / "postProcessing" / "forceCoeffs1").glob("*/forceCoeffs.dat"))
    if not candidates:
        return {"iter": np.array([]), "Cl": np.array([]), "Cd": np.array([])}

    text = candidates[0].read_text()
    header = []
    rows = []
    for line in text.splitlines():
        if line.startswith("#"):
            header.append(line)
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    if not rows:
        return {"iter": np.array([]), "Cl": np.array([]), "Cd": np.array([])}
    arr = np.array(rows)
    last_header = header[-1] if header else ""
    # lstrip("#") already removes the leading '#', so the remaining tokens line
    # up 1:1 with the data columns (token 0 == Time == arr column 0).  Header
    # layout is e.g. "# Time  Cm  Cd  Cl  Cl(f)  Cl(r)".
    cols = last_header.lstrip("#").split()
    def col(*names: str) -> int | None:
        for name in names:
            for i, c in enumerate(cols):
                if c == name:
                    return i
        return None
    cl_i = col("Cl", "Cl(t)", "Cl_total")
    cd_i = col("Cd", "Cd(t)", "Cd_total")
    # Fallbacks match the standard forceCoeffs.dat layout: Time Cm Cd Cl ...
    if cl_i is None: cl_i = 3 if arr.shape[1] > 3 else None
    if cd_i is None: cd_i = 2 if arr.shape[1] > 2 else None
    iters = arr[:, 0].astype(int)
    cl = arr[:, cl_i] if cl_i is not None and cl_i < arr.shape[1] else np.full(len(iters), np.nan)
    cd = arr[:, cd_i] if cd_i is not None and cd_i < arr.shape[1] else np.full(len(iters), np.nan)
    return {"iter": iters, "Cl": cl, "Cd": cd}


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_case(of_case_dir: Path, case_dir: Path, case_id: str) -> dict:
    """Parse OF outputs and write fields.h5, mesh.h5, geometry.h5, convergence.h5."""
    parsed = parse_case_id(case_id)
    polyMesh = of_case_dir / "constant" / "polyMesh"

    # Try to materialize cell centers; if it fails we'll fall back.
    cc_ok = run_postprocess(of_case_dir, "writeCellCentres")
    yp_ok = run_postprocess(of_case_dir, "yPlus")
    # wallShearStress needs the turbulence model, so it must be run through a
    # solver module (otherwise "Unable to find turbulence model in the database").
    ws_ok = run_postprocess(of_case_dir, "wallShearStress",
                            solver="incompressibleFluid")

    time_dir = latest_time_dir(of_case_dir)
    LOG.info("[%s] extracting from time dir %s", case_id, time_dir.name)

    # ---- field internals ----
    def load_field(name: str) -> np.ndarray:
        path = time_dir / name
        if not path.exists() and not (path.with_suffix(".gz")).exists():
            raise FileNotFoundError(f"missing field file {path}")
        return parse_internal_field(_read_text(path))

    U = load_field("U")
    p = load_field("p")
    k = load_field("k")
    omega = load_field("omega")
    nut = load_field("nut")
    n_cells = len(p)

    # ---- cell centers (OF13 names them Ccx/Ccy/Ccz) ----
    cx_path = next((time_dir / n for n in ("Ccx", "Cx") if (time_dir / n).exists()), None)
    cy_path = next((time_dir / n for n in ("Ccy", "Cy") if (time_dir / n).exists()), None)
    if cx_path and cy_path:
        Cx = parse_internal_field(_read_text(cx_path))
        Cy = parse_internal_field(_read_text(cy_path))
        cell_centers3 = np.column_stack([Cx, Cy, np.zeros_like(Cx)])
    else:
        cell_centers3 = None    # filled in after we parse polyMesh

    # ---- polyMesh ----
    points3 = parse_points_file(polyMesh / "points")
    faces = parse_faces_file(polyMesh / "faces")
    owner = parse_int_list(polyMesh / "owner")
    neighbour = parse_int_list(polyMesh / "neighbour")
    patches = parse_boundary_file(polyMesh / "boundary")

    proj = project_2d_mesh(points3, faces, owner, neighbour, n_cells)
    points2 = proj["points2"]
    connectivity = proj["connectivity"]

    # If cell centers weren't materialized, average each cell's vertex coords
    if cell_centers3 is None:
        centers = np.zeros((n_cells, 2))
        for ci in range(n_cells):
            verts = connectivity[ci]
            verts = verts[verts >= 0]
            if len(verts):
                centers[ci] = points2[verts].mean(axis=0)
        cell_centers2 = centers
    else:
        cell_centers2 = cell_centers3[:, :2]

    # ---- boundary markers ----
    markers = boundary_markers(faces, owner, n_cells, patches)

    # ---- wall-surface table (airfoil boundary, one row per wall face) ----
    ws_path = time_dir / "wallShearStress"
    shear_vecs = _parse_patch_vector_field(ws_path, "airfoilWalls") \
        if (ws_path.exists() or ws_path.with_suffix(".gz").exists()) \
        else np.empty((0, 3))
    p_wall_source = p.ravel() if p.ndim > 1 else p
    wall = wall_surface_table(faces, owner, patches, points3, cell_centers2,
                              p_wall_source, shear_vecs)

    # ---- airfoil polygon (from polyMesh, ordered) ----
    airfoil_pts3 = []
    for p_ in patches:
        if p_["name"] in ("airfoilWalls", "airfoil"):
            for fi in range(p_["startFace"], p_["startFace"] + p_["nFaces"]):
                airfoil_pts3.extend(faces[fi])
    airfoil_pts3 = np.array(sorted(set(airfoil_pts3)), dtype=int)
    airfoil_xy_unsorted = points3[airfoil_pts3][:, :2]
    # Use the analytic NACA polygon (TE -> upper -> LE -> lower -> TE) as the
    # canonical ordered representation; spec §6.3 wants this ordering.
    airfoil_coords = naca4_coordinates(parsed["naca_code"], n_points=200)

    # ---- wall distance ----
    # Exact distance from each cell centre to the analytic NACA surface.  This
    # replaces the old nearest-*vertex* KD-tree query against the faceted mesh
    # boundary, which over-estimated the true wall distance (point-to-vertex
    # instead of point-to-surface).  See geometry/distance.py.
    wall_distance = naca_surface_distance(cell_centers2, parsed["naca_code"])

    # ---- y+ ----  written by the yPlus function object as a volScalarField
    #               whose boundaryField on `airfoilWalls` carries the wall values.
    y_plus_arr = _extract_wall_yplus(time_dir / "yPlus") if (time_dir / "yPlus").exists() \
        else np.array([])
    if y_plus_arr.size == 0:
        # Fall back to the postProcessing/yPlus/0/yPlus.dat min/max/avg row.
        yp_dat = sorted((of_case_dir / "postProcessing" / "yPlus").glob("*/yPlus.dat"))
        if yp_dat:
            tail = yp_dat[-1].read_text().strip().splitlines()[-1].split()
            try:
                y_plus_arr = np.array([float(tail[2]), float(tail[3]), float(tail[4])])
            except (IndexError, ValueError):
                y_plus_arr = np.array([])

    # ---- convergence (residuals + cl/cd) ----
    log_path = of_case_dir / "simpleFoam.log"
    parsed_log = parse_solver_log(log_path.read_text(errors="replace")
                                   if log_path.exists() else "")
    conv = detect_convergence(parsed_log["history"])
    forces = parse_force_coeffs(of_case_dir)
    n_iter = parsed_log["n_iter"]

    fields_keys = ["U", "p", "k", "omega"]
    res_history = np.full((n_iter, 4), np.nan)
    if n_iter > 0:
        ux = np.array(parsed_log["history"]["Ux"], dtype=float)
        uy = np.array(parsed_log["history"]["Uy"], dtype=float)
        # Use the larger of Ux, Uy as the U residual (worst component)
        u_res = np.fmax(np.nan_to_num(ux, nan=-np.inf),
                        np.nan_to_num(uy, nan=-np.inf))
        u_res[u_res == -np.inf] = np.nan
        res_history[:, 0] = u_res
        res_history[:, 1] = parsed_log["history"]["p"]
        res_history[:, 2] = parsed_log["history"]["k"]
        res_history[:, 3] = parsed_log["history"]["omega"]

    converged_overall = conv["overall"]
    # iterations_to_convergence: first iter at which all fields' residuals
    # had dropped ≥ 4 orders.  If still converging, store n_iter so it's at
    # least informative for the QC step.
    iters_to_conv = _first_converged_iter(parsed_log["history"]) or n_iter

    # ---- write HDF5s ----
    case_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(case_dir / "fields.h5", "w") as h:
        h.create_dataset("U", data=U[:, :2] if U.ndim == 2 else U)
        h.create_dataset("p", data=p)
        h.create_dataset("k", data=k)
        h.create_dataset("omega", data=omega)
        h.create_dataset("nut", data=nut)
        h.create_dataset("wall_distance", data=wall_distance)

    with h5py.File(case_dir / "mesh.h5", "w") as h:
        h.create_dataset("cell_centers", data=cell_centers2)
        h.create_dataset("points", data=points2)
        h.create_dataset("connectivity", data=connectivity)
        h.create_dataset("boundary_markers", data=markers)
        g = h.create_group("wall")
        for key, arr in wall.items():
            g.create_dataset(key, data=arr)

    with h5py.File(case_dir / "geometry.h5", "w") as h:
        h.create_dataset("airfoil_coordinates", data=airfoil_coords)
        h.create_dataset("airfoil_mesh_points", data=airfoil_xy_unsorted)

    with h5py.File(case_dir / "convergence.h5", "w") as h:
        h.create_dataset("residual_history", data=res_history)
        h.create_dataset("cl_history", data=forces["Cl"])
        h.create_dataset("cd_history", data=forces["Cd"])
        h.create_dataset("iteration", data=forces["iter"])
        h.create_dataset("y_plus", data=y_plus_arr)
        h.attrs["iterations_to_convergence"] = iters_to_conv
        h.attrs["iterations_total"] = n_iter
        h.attrs["converged"] = bool(converged_overall)
        for f in fields_keys:
            v = parsed_log["final"].get("Ux" if f == "U" else f)
            if v is not None:
                h.attrs[f"final_residual_{f}"] = v
        for f, d in conv["drops"].items():
            h.attrs[f"orders_drop_{f}"] = d

    return {
        "n_cells": n_cells,
        "n_iter": n_iter,
        "converged": converged_overall,
        "drops": conv["drops"],
        "y_plus_max": float(np.nanmax(y_plus_arr)) if y_plus_arr.size else float("nan"),
        "cl_final": float(forces["Cl"][-1]) if forces["Cl"].size else float("nan"),
        "cd_final": float(forces["Cd"][-1]) if forces["Cd"].size else float("nan"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract dataset HDF5 files for one case.")
    p.add_argument("case_id")
    p.add_argument("--of-case", required=True, type=Path)
    p.add_argument("--case-dir", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = extract_case(args.of_case, args.case_dir, args.case_id)
    LOG.info("Extraction summary: %s", summary)


if __name__ == "__main__":
    main()
