"""Alternative meshing path: structured 2D C-mesh -> Gmsh .msh -> gmshToFoam.

This is a drop-in replacement for ``generate_mesh.generate_mesh`` selected by
the ``--c-mesh`` flag in ``generate_dataset.py``. Same signature
``(of_case_dir, case_id) -> quality_dict`` so the surrounding pipeline does
not need to know which mesher was used.

The C-mesh algorithm (closed-TE NACA + bisector-aligned wake + half-circle
far-field + TFI grid) is the one prototyped under ``prototype/cmesh/``; the
core is inlined here so this module is self-contained and the dataset path
does not depend on the prototype tree.

Pipeline this module implements:
    1. Build the structured (i, j) C-mesh node array.
    2. Extrude one cell in z and serialise to a Gmsh .msh (v2.2) file with
       physical groups: airfoilWalls / inlet / top / bottom / outlet /
       frontAndBack / fluid.
    3. Run ``gmshToFoam`` (already on PATH for the existing pipeline).
    4. Re-type patches via ``patch_boundary_file`` (reused from generate_mesh).
    5. Run ``checkMesh`` and return the parsed quality dict.

Wake-cut handling: the two wake strands occupy the same (x, y) at every
internal i-pair (k, ni-1-k). We merge those node IDs in the .msh so the
wake cut becomes an internal face. The pair at k=0 (outlet end) is *not*
merged so the LEFT face of i=0 cells and RIGHT face of i=ni-1 cells stay as
two separate outlet boundary faces.
"""
from __future__ import annotations

import io
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import math

import numpy as np
from scipy.optimize import brentq

from physics_oracle.core.case_spec import parse_case_id
from physics_oracle.core.envelope import CHORD, NU
from physics_oracle.core.logging import setup_logging
from physics_oracle.meshing.gmsh_mesh import parse_check_mesh, patch_boundary_file

LOG = setup_logging()

OPENFOAM_BASHRC = os.environ.get("OPENFOAM_BASHRC", "/opt/openfoam13/etc/bashrc")


def _of_run(args: list, cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run an OpenFOAM command with the OF environment sourced."""
    cmd = f"source {OPENFOAM_BASHRC} && " + " ".join(shlex.quote(str(a)) for a in args)
    return subprocess.run(
        ["bash", "-c", cmd],
        cwd=cwd, capture_output=True, text=True, timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Defaults (matched to the gmsh path's domain extent)
# ---------------------------------------------------------------------------

DOMAIN_D_FAR = 20.0          # half-circle radius (>=20c per spec)
DOMAIN_X_OUTLET = 25.0       # downstream extent
EXTRUSION_DZ = 0.01          # 1-cell-thick z extrusion (matches gmsh path)

N_AIRFOIL = 241              # nodes around airfoil (TE -> LE -> TE)
N_WAKE = 120                 # nodes per wake leg (TE -> outlet)
N_LAYERS = 120               # wall-normal cells (floor; grows if needed at high Re)
YPLUS_TARGET = 0.8           # wall y+ design target (held ~constant vs Re)
RE_DESIGN = 5.0e5            # legacy fixed design Re (optional override only)
GROWTH_MAX = 1.2             # cap on wall-normal geometric growth ratio (§4.2)
N_LAYERS_CAP = 300           # upper bound on the adaptive wall-normal cell count
WAKE_CUT_AR_CAP = 100.0      # max in-plane aspect ratio (dx / first spacing) of
                             # the wake-cut first-layer cells.  With a global y1
                             # first spacing the far-wake cut cells reached AR
                             # ~30,000 and ~90 deg non-orthogonality (centroid
                             # shifts of the slivers point the cell-to-cell
                             # vector along the cut), flooring the p residual
                             # and destabilising GAMG at high Re.


# ---------------------------------------------------------------------------
# Closed-TE NACA 4-digit (-0.1036 last coefficient, sharp at (chord, 0))
# ---------------------------------------------------------------------------

def _naca4_params(code: str) -> tuple[float, float, float]:
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"NACA code must be 4 digits, got {code!r}")
    return int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:]) / 100.0


def naca4_closed_te(naca_code: str, n_points: int = 121, chord: float = CHORD) -> np.ndarray:
    """Closed-TE NACA 4-digit. The dataset's open-TE generator in
    ``common.naca4_coordinates`` is intentionally used by the gmsh path; the
    C-mesh path needs the closed TE so the inner curve has no concave corner
    at the trailing edge.
    """
    m, p, t = _naca4_params(naca_code)
    beta = np.linspace(0.0, np.pi, n_points)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1036 * x ** 4
    )
    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            (m / p ** 2) * (2.0 * p * x - x ** 2),
            (m / (1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x ** 2),
        )
        dyc_dx = np.where(
            x < p,
            (2.0 * m / p ** 2) * (p - x),
            (2.0 * m / (1.0 - p) ** 2) * (p - x),
        )
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta);  yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta);  yl = yc - yt * np.cos(theta)
    upper = np.column_stack([xu[::-1], yu[::-1]])
    lower = np.column_stack([xl[1:], yl[1:]])
    return np.vstack([upper, lower]) * chord


# ---------------------------------------------------------------------------
# BL sizing + 1-D distributions
# ---------------------------------------------------------------------------

def _first_cell_height(yplus: float, re: float, U: float, nu: float) -> float:
    cf = 0.058 * re ** (-0.2)
    u_tau = U * np.sqrt(0.5 * cf)
    return float(yplus * nu / u_tau)


def _solve_growth(L: float, y1: float, n_cells: int) -> float:
    if y1 * n_cells >= L * (1.0 - 1e-12):
        return 1.0
    return float(brentq(
        lambda r: y1 * (r ** n_cells - 1.0) / (r - 1.0) - L,
        1.0 + 1e-12, 5.0, xtol=1e-12,
    ))


def _layers_for_growth(L: float, y1: float, r_max: float,
                       n_min: int, n_cap: int) -> int:
    """Smallest wall-normal cell count whose geometric distribution (first cell
    ``y1``, ratio <= ``r_max``) spans ``L``. Floored at ``n_min`` and capped at
    ``n_cap``. Keeps near-wall growth bounded when ``y1`` shrinks at high Re."""
    if y1 <= 0.0 or y1 * n_min >= L:
        return n_min
    n = math.ceil(math.log1p((L / y1) * (r_max - 1.0)) / math.log(r_max))
    return int(min(n_cap, max(n_min, n)))


def _geom_nodes(L: float, y1: float, n_cells: int) -> np.ndarray:
    r = _solve_growth(L, y1, n_cells)
    s = np.zeros(n_cells + 1)
    h = y1
    for k in range(n_cells):
        s[k + 1] = s[k] + h
        h *= r
    if s[-1] != 0.0:
        s *= L / s[-1]
    return s


# ---------------------------------------------------------------------------
# C-mesh construction (inner curve + far-field outer + TFI grid)
# ---------------------------------------------------------------------------

def build_c_mesh_nodes(
    naca_code: str,
    re_value: float,
    aoa_deg: float = 0.0,
    *,
    n_airfoil_target: int = N_AIRFOIL,
    n_wake: int = N_WAKE,
    n_layers: int = N_LAYERS,
    x_outlet: float = DOMAIN_X_OUTLET,
    d_far: float = DOMAIN_D_FAR,
    yplus_target: float = YPLUS_TARGET,
    re_design: float | None = None,
    chord: float = CHORD,
    nu: float = NU,
) -> tuple[np.ndarray, int, int]:
    """Return (nodes, n_airfoil, n_wake) where nodes has shape (ni, nj, 2).

    The wake is freestream-aligned: it leaves TE at slope tan(AoA) and stays
    on that slope all the way to the outlet, so the wake region is well
    aligned with the actual flow direction. For symmetric airfoils at AoA=0
    this gives a horizontal wake; for any non-zero AoA the sign of the wake
    slope matches the AoA sign.
    """
    import math
    n_half = (n_airfoil_target + 1) // 2
    airfoil = naca4_closed_te(naca_code, n_points=n_half, chord=chord)
    n_airfoil = airfoil.shape[0]

    m_inf = math.tan(math.radians(aoa_deg))   # freestream slope (AoA)

    # BL first-cell height sized for y+ at the *actual* case Re, so wall y+
    # stays ~yplus_target across the whole Reynolds range. (The old fixed
    # re_design=5e5 sizing let y+ drift from ~0.2 at Re 1e5 to ~4 at Re 3e6.)
    # Pass re_design explicitly to force a single dataset-wide wall spacing.
    re_size = re_value if re_design is None else re_design
    U_size = re_size * nu / chord
    y1 = _first_cell_height(yplus_target, re_size, U_size, nu)

    te_sp = 0.5 * (
        float(np.linalg.norm(airfoil[1] - airfoil[0]))
        + float(np.linalg.norm(airfoil[-1] - airfoil[-2]))
    )
    s_te = _geom_nodes(x_outlet - 1.0, te_sp, n_wake - 1)
    x_wake = 1.0 + s_te
    # Linear wake along the freestream direction: y = tan(AoA) * (x - 1).
    # Zero curvature so there is no focal-point concern from the wake itself.
    y_wake = m_inf * s_te

    wake_top = np.column_stack([x_wake[::-1], y_wake[::-1]])
    wake_bot = np.column_stack([x_wake,        y_wake])
    inner = np.vstack([wake_top, airfoil[1:-1], wake_bot])
    ni = inner.shape[0]

    # Far-field outer C: top horizontal -> half-circle (mid-chord, d_far)
    # -> bottom horizontal. Same construction as the prototype.
    chord_mid = 0.5
    ff = np.empty_like(inner)
    fr_top = np.linspace(0.0, 1.0, n_wake)
    ff[:n_wake, 0] = x_outlet - (x_outlet - chord_mid) * fr_top
    ff[:n_wake, 1] = d_far
    n_int = n_airfoil - 2
    fr_arc = (np.arange(n_int) + 1.0) / (n_airfoil - 1)
    th = 0.5 * np.pi + fr_arc * np.pi
    ff[n_wake:n_wake + n_int, 0] = chord_mid + d_far * np.cos(th)
    ff[n_wake:n_wake + n_int, 1] =             d_far * np.sin(th)
    fr_bot = np.linspace(0.0, 1.0, n_wake)
    ff[n_wake + n_int:, 0] = chord_mid + (x_outlet - chord_mid) * fr_bot
    ff[n_wake + n_int:, 1] = -d_far

    # Adaptive wall-normal resolution: shrinking y1 at high Re steepens the
    # geometric growth, so add layers to keep the ratio <= GROWTH_MAX. Floored
    # at the requested n_layers -- within the trained band (Re <=~3e6) it stays
    # at the floor so mesh topology is unchanged; only very high Re adds layers.
    L_max = max(float(np.linalg.norm(ff[i] - inner[i])) for i in range(ni))
    n_layers = _layers_for_growth(L_max, y1, GROWTH_MAX,
                                  n_min=n_layers, n_cap=N_LAYERS_CAP)
    growth = _solve_growth(L_max, y1, n_layers)
    LOG.info("BL sizing: Re=%.3g  y+_target=%.2f  y1=%.3e  n_layers=%d  max_growth=%.3f",
             re_size, yplus_target, y1, n_layers, growth)

    # Per-column first spacing: y1 on the airfoil (wall resolution), growing
    # along the wake cut so the cut's first-layer cells keep in-plane aspect
    # ratio <= WAKE_CUT_AR_CAP.  The cut is only a numerical interface in the
    # far wake; resolving it at y1 produced AR ~30k slivers whose ~90 deg
    # non-orthogonality floored the p residual (see WAKE_CUT_AR_CAP note).
    # dx along the cut is mirror-symmetric (top strand i <-> bottom ni-1-i),
    # so both sides of every cut face get the same first spacing.
    dxs = np.gradient(s_te)                    # local streamwise spacing, TE->outlet
    y1_col = np.full(ni, y1)
    y1_wake = np.maximum(y1, dxs / WAKE_CUT_AR_CAP)
    y1_col[:n_wake] = y1_wake[::-1]            # wake_top: outlet -> TE
    y1_col[ni - n_wake:] = y1_wake             # wake_bot: TE -> outlet

    # Pure TFI grid: linear interpolation inner -> ff with geometric j-clustering.
    nj = n_layers + 1
    nodes = np.empty((ni, nj, 2))
    for i in range(ni):
        L = float(np.linalg.norm(ff[i] - inner[i]))
        s = _geom_nodes(L, y1_col[i], n_layers)
        eta = s / max(L, 1e-30)
        nodes[i, :, 0] = (1.0 - eta) * inner[i, 0] + eta * ff[i, 0]
        nodes[i, :, 1] = (1.0 - eta) * inner[i, 1] + eta * ff[i, 1]
    return nodes, n_airfoil, n_wake


# ---------------------------------------------------------------------------
# Gmsh .msh v2.2 writer (with wake-cut node merging)
# ---------------------------------------------------------------------------

# Physical group IDs.  patch_boundary_file (re-used from generate_mesh) keys
# off the patch *names*, so the IDs themselves don't matter as long as the
# Physical Names lines below match.
PHYS_AIRFOIL = 1
PHYS_INLET   = 2
PHYS_OUTLET  = 3
PHYS_TOP     = 4
PHYS_BOTTOM  = 5
PHYS_FRONTBACK = 6
PHYS_FLUID   = 7


def _canonical_node_key(i: int, j: int, k: int, ni: int, n_wake: int) -> tuple[int, int, int]:
    """Canonical (i, j, k) lookup key for node-ID merging across the wake cut.

    Interior wake pairs (k_w, ni-1-k_w) for k_w in [1, n_wake-1] merge at
    every j. The outlet-end pair (i=0, i=ni-1) merges only at j=0 so the
    j=0 face at the outlet end becomes internal (paired with the wake_bot
    side) while the LEFT face of i=0 column and RIGHT face of i=ni-1 column
    stay as two separate outlet boundary faces at j>0.
    """
    # The wake_top (i in [0, n_wake-1]) and wake_bot (i in [ni-n_wake, ni-1])
    # strands occupy the SAME physical (x, y) line ONLY AT j=0 -- the wake
    # cut. For j > 0 they extend in opposite normal directions (+y vs -y)
    # and therefore live at distinct physical positions, so their nodes
    # must remain distinct. Merging only at j=0 turns the wake cut into a
    # single internal face shared by the two adjacent cells without
    # collapsing wake_top and wake_bot into duplicate cells.
    if j == 0:
        if 0 < i < n_wake:
            return (i, 0, k)
        if ni - n_wake <= i < ni - 1:
            return (ni - 1 - i, 0, k)
        if i == 0 or i == ni - 1:
            return (0, 0, k)
    return (i, j, k)


def write_gmsh_msh(
    nodes_2d: np.ndarray,
    n_airfoil: int,
    n_wake: int,
    msh_path: Path,
    dz: float = EXTRUSION_DZ,
) -> None:
    """Serialise the 2D structured mesh as a Gmsh .msh (v2.2) with the wake
    cut merged into internal faces and physical groups for OpenFOAM."""
    ni, nj, _ = nodes_2d.shape

    # Assign unique node IDs honouring wake-cut merging.
    node_id = np.zeros((ni, nj, 2), dtype=np.int64)
    canonical: dict[tuple[int, int, int], int] = {}
    next_id = 1
    for i in range(ni):
        for j in range(nj):
            for k in (0, 1):
                key = _canonical_node_key(i, j, k, ni, n_wake)
                nid = canonical.get(key)
                if nid is None:
                    nid = next_id
                    next_id += 1
                    canonical[key] = nid
                node_id[i, j, k] = nid
    n_nodes = next_id - 1

    # Resolve canonical keys back to (x, y, z) coordinates.
    coords = np.zeros((n_nodes, 3))
    for (ci, j, k), nid in canonical.items():
        coords[nid - 1, 0] = nodes_2d[ci, j, 0]
        coords[nid - 1, 1] = nodes_2d[ci, j, 1]
        coords[nid - 1, 2] = k * dz

    # ---------------------------------------------------------------- elements
    # Hex (type 5). Each cell stored with bottom face ordered to keep the
    # cell volume positive (signed area of bottom face must be positive in
    # (x, y) when viewed from +z so that gmshToFoam infers a positive volume).
    # Our i-sweep wraps CW around the lower half of the airfoil, so cells
    # there need their bottom-face winding reversed.
    hex_elems: list[list[int]] = []
    cell_node_ids: list[list[int]] = []   # per-cell vertex list, used below
    for i in range(ni - 1):
        for j in range(nj - 1):
            p0 = nodes_2d[i,     j    ]
            p1 = nodes_2d[i + 1, j    ]
            p2 = nodes_2d[i + 1, j + 1]
            p3 = nodes_2d[i,     j + 1]
            sa = ((p2[0] - p0[0]) * (p3[1] - p1[1])
                  - (p2[1] - p0[1]) * (p3[0] - p1[0]))
            n_a = int(node_id[i,     j,     0])
            n_b = int(node_id[i + 1, j,     0])
            n_c = int(node_id[i + 1, j + 1, 0])
            n_d = int(node_id[i,     j + 1, 0])
            n_e = int(node_id[i,     j,     1])
            n_f = int(node_id[i + 1, j,     1])
            n_g = int(node_id[i + 1, j + 1, 1])
            n_h = int(node_id[i,     j + 1, 1])
            if sa >= 0:
                ids = [n_a, n_b, n_c, n_d, n_e, n_f, n_g, n_h]
            else:
                ids = [n_a, n_d, n_c, n_b, n_e, n_h, n_g, n_f]
            hex_elems.append(ids)
            cell_node_ids.append(ids)

    # Boundary quads (type 3). The face vertex order must produce a normal
    # that points OUTWARD from the parent cell. Rather than threading
    # geometric assumptions through every face/region (which is brittle on
    # a wrap-around C topology), we orient each face via the cell-centroid
    # check: compute the face normal and the inward direction (cell
    # centroid - face centroid); if normal . inward > 0, reverse the winding.
    def _orient_outward(face_ids: list[int], cell_ids: list[int]) -> list[int]:
        fp = coords[[f - 1 for f in face_ids]]
        cc = coords[[c - 1 for c in cell_ids]].mean(axis=0)
        fc = fp.mean(axis=0)
        n = np.cross(fp[1] - fp[0], fp[2] - fp[1])
        if float(np.dot(n, cc - fc)) > 0.0:
            return [face_ids[0], face_ids[3], face_ids[2], face_ids[1]]
        return face_ids

    def _cell_index(i: int, j: int) -> int:
        return i * (nj - 1) + j

    def _push(phys: int, face_ids: list[int], i: int, j: int) -> None:
        bdy.append((phys, _orient_outward(face_ids, cell_node_ids[_cell_index(i, j)])))

    bdy: list[tuple[int, list[int]]] = []

    # airfoilWalls: j=0 face of cells in i ∈ [n_wake-1, n_wake+n_airfoil-3].
    for i in range(n_wake - 1, n_wake + n_airfoil - 2):
        face = [int(node_id[i,     0, 0]),
                int(node_id[i + 1, 0, 0]),
                int(node_id[i + 1, 0, 1]),
                int(node_id[i,     0, 1])]
        _push(PHYS_AIRFOIL, face, i, 0)

    # Outer (j=nj-1): top horizontal / half-circle inlet / bottom horizontal.
    j_last = nj - 2
    for i in range(ni - 1):
        if i < n_wake - 1:
            phys = PHYS_TOP
        elif i < n_wake + n_airfoil - 2:
            phys = PHYS_INLET
        else:
            phys = PHYS_BOTTOM
        face = [int(node_id[i,     nj - 1, 0]),
                int(node_id[i + 1, nj - 1, 0]),
                int(node_id[i + 1, nj - 1, 1]),
                int(node_id[i,     nj - 1, 1])]
        _push(phys, face, i, j_last)

    # Outlet: LEFT face of i=0 column + RIGHT face of i=ni-2 column.
    for j in range(nj - 1):
        face_left = [int(node_id[0, j,     0]),
                     int(node_id[0, j + 1, 0]),
                     int(node_id[0, j + 1, 1]),
                     int(node_id[0, j,     1])]
        _push(PHYS_OUTLET, face_left, 0, j)
        face_right = [int(node_id[ni - 1, j,     0]),
                      int(node_id[ni - 1, j + 1, 0]),
                      int(node_id[ni - 1, j + 1, 1]),
                      int(node_id[ni - 1, j,     1])]
        _push(PHYS_OUTLET, face_right, ni - 2, j)

    # frontAndBack: z=0 and z=dz face of every cell.
    for i in range(ni - 1):
        for j in range(nj - 1):
            face_front = [int(node_id[i,     j,     0]),
                          int(node_id[i + 1, j,     0]),
                          int(node_id[i + 1, j + 1, 0]),
                          int(node_id[i,     j + 1, 0])]
            _push(PHYS_FRONTBACK, face_front, i, j)
            face_back = [int(node_id[i,     j,     1]),
                         int(node_id[i + 1, j,     1]),
                         int(node_id[i + 1, j + 1, 1]),
                         int(node_id[i,     j + 1, 1])]
            _push(PHYS_FRONTBACK, face_back, i, j)

    n_elem = len(hex_elems) + len(bdy)

    # ---------------------------------------------------------------- write
    buf = io.StringIO()
    buf.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
    buf.write("$PhysicalNames\n7\n")
    buf.write(f'2 {PHYS_AIRFOIL} "airfoilWalls"\n')
    buf.write(f'2 {PHYS_INLET} "inlet"\n')
    buf.write(f'2 {PHYS_OUTLET} "outlet"\n')
    buf.write(f'2 {PHYS_TOP} "top"\n')
    buf.write(f'2 {PHYS_BOTTOM} "bottom"\n')
    buf.write(f'2 {PHYS_FRONTBACK} "frontAndBack"\n')
    buf.write(f'3 {PHYS_FLUID} "fluid"\n')
    buf.write("$EndPhysicalNames\n")
    buf.write(f"$Nodes\n{n_nodes}\n")
    for nid in range(1, n_nodes + 1):
        x, y, z = coords[nid - 1]
        buf.write(f"{nid} {x:.10e} {y:.10e} {z:.10e}\n")
    buf.write("$EndNodes\n")
    buf.write(f"$Elements\n{n_elem}\n")
    eid = 1
    for verts in hex_elems:
        # type 5 (hex), 2 tags: phys_id and elementary_id (use phys_id again)
        buf.write(f"{eid} 5 2 {PHYS_FLUID} {PHYS_FLUID} "
                  + " ".join(str(v) for v in verts) + "\n")
        eid += 1
    for phys, verts in bdy:
        buf.write(f"{eid} 3 2 {phys} {phys} "
                  + " ".join(str(v) for v in verts) + "\n")
        eid += 1
    buf.write("$EndElements\n")
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    msh_path.write_text(buf.getvalue())


# ---------------------------------------------------------------------------
# Top-level entry: same signature as generate_mesh.generate_mesh
# ---------------------------------------------------------------------------

def generate_c_mesh(of_case_dir: Path, case_id: str) -> dict:
    """Build a C-mesh for the case and import it into the OpenFOAM polyMesh.

    Drop-in replacement for ``generate_mesh.generate_mesh``.
    """
    parsed = parse_case_id(case_id)

    nodes, n_airfoil, n_wake = build_c_mesh_nodes(
        parsed["naca_code"], parsed["Re"], parsed["aoa_deg"]
    )
    LOG.info("[%s] c-mesh: ni=%d, nj=%d, cells=%d",
             case_id, nodes.shape[0], nodes.shape[1],
             (nodes.shape[0] - 1) * (nodes.shape[1] - 1))

    msh_path = of_case_dir / "naca_airfoil.msh"
    poly_mesh_dir = of_case_dir / "constant" / "polyMesh"
    if poly_mesh_dir.exists():
        shutil.rmtree(poly_mesh_dir)

    write_gmsh_msh(nodes, n_airfoil, n_wake, msh_path)

    LOG.info("[%s] gmshToFoam %s", case_id, msh_path.name)
    res = _of_run(["gmshToFoam", str(msh_path)], cwd=of_case_dir, timeout=600)
    (of_case_dir / "gmshToFoam.log").write_text(res.stdout + res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"gmshToFoam failed for {case_id}: see gmshToFoam.log")
    msh_path.unlink(missing_ok=True)

    patch_boundary_file(of_case_dir / "constant" / "polyMesh" / "boundary")

    LOG.info("[%s] checkMesh", case_id)
    res = _of_run(["checkMesh"], cwd=of_case_dir, timeout=600)
    log_text = res.stdout + res.stderr
    (of_case_dir / "checkMesh.log").write_text(log_text)
    quality = parse_check_mesh(log_text)
    if res.returncode != 0:
        LOG.warning("checkMesh nonzero exit for %s: %s", case_id, quality.get("errors"))
    LOG.info("[%s] c-mesh: %s cells, max non-orth %s, max skew %s",
             case_id, quality["n_cells"], quality["max_non_orthogonality"],
             quality["max_skewness"])
    return quality
