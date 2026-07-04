"""backfill_wall.py — Add the airfoil-surface table to already-extracted cases.

For every raw case that has an `of_case/` it:
  1. Runs `foamPostProcess -solver incompressibleFluid -func wallShearStress`
     on the converged solution (no re-solve).
  2. Rebuilds the wall table (face centers, outward normals, kinematic wall
     shear, kinematic surface pressure, edge length, owner-cell index).
  3. Writes it as a `/wall` group into the existing `mesh.h5`.

Cell centers are read from `mesh.h5` and surface pressure from `fields.h5`, so
only the boundary geometry is re-parsed from `constant/polyMesh`.  After this,
re-run `build_ml_dataset` to fold the `wall_*` arrays into the `.npz` files.

Usage:
  python -m physics_oracle.cli.backfill_wall --cases-dir ../NACA_4_Digit_Raw_OF \
      [--mesh-mirror ../NACA_4_Digit_for_ML/mesh] [--file CASE_ID]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

from physics_oracle.core.logging import setup_logging
from physics_oracle.core.paths import CASES_DIR
from physics_oracle.openfoam_setup.extract import (
    _parse_patch_vector_field,
    _read_text,
    latest_time_dir,
    parse_boundary_file,
    parse_faces_file,
    parse_int_list,
    parse_internal_field,
    parse_points_file,
    run_postprocess,
    wall_surface_table,
)

LOG = setup_logging()


def backfill_case(case_dir: Path, mesh_mirror: Path | None = None) -> str:
    """Return 'ok', 'skip:<reason>' for one case."""
    of_case = case_dir / "of_case"
    mesh_h5 = case_dir / "mesh.h5"
    fields_h5 = case_dir / "fields.h5"
    if not of_case.is_dir():
        return "skip:no of_case"
    if not mesh_h5.exists() or not fields_h5.exists():
        return "skip:no h5"

    if not run_postprocess(of_case, "wallShearStress", solver="incompressibleFluid"):
        return "skip:foamPostProcess failed"

    polyMesh = of_case / "constant" / "polyMesh"
    points3 = parse_points_file(polyMesh / "points")
    faces = parse_faces_file(polyMesh / "faces")
    owner = parse_int_list(polyMesh / "owner")
    patches = parse_boundary_file(polyMesh / "boundary")

    with h5py.File(mesh_h5, "r") as h:
        cell_centers2 = h["cell_centers"][:]
    with h5py.File(fields_h5, "r") as h:
        p_internal = h["p"][:].ravel()

    time_dir = latest_time_dir(of_case)
    ws_path = time_dir / "wallShearStress"
    if not (ws_path.exists() or ws_path.with_suffix(".gz").exists()):
        return "skip:no wallShearStress output"
    shear_vecs = _parse_patch_vector_field(ws_path, "airfoilWalls")

    wall = wall_surface_table(faces, owner, patches, points3, cell_centers2,
                              p_internal, shear_vecs)

    with h5py.File(mesh_h5, "a") as h:
        if "wall" in h:
            del h["wall"]
        g = h.create_group("wall")
        for key, arr in wall.items():
            g.create_dataset(key, data=arr)

    if mesh_mirror is not None:
        mirror = mesh_mirror / f"{case_dir.name}_mesh.h5"
        if mirror.exists():
            shutil.copy2(mesh_h5, mirror)

    return "ok"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases-dir", type=Path, default=CASES_DIR)
    p.add_argument("--mesh-mirror", type=Path, default=None,
                   help="If set, copy each updated mesh.h5 to "
                        "<mirror>/<case>_mesh.h5 (e.g. the ML dataset mesh/ dir).")
    p.add_argument("--file", metavar="CASE_ID", help="Process a single case.")
    args = p.parse_args()

    if args.file:
        candidates = [args.cases_dir / args.file]
    else:
        candidates = sorted(d for d in args.cases_dir.iterdir() if d.is_dir())

    counts: dict[str, int] = {}
    for case_dir in candidates:
        try:
            status = backfill_case(case_dir, args.mesh_mirror)
        except Exception as exc:                                  # noqa: BLE001
            status = f"skip:error({type(exc).__name__})"
            LOG.warning("[%s] %s: %s", case_dir.name, type(exc).__name__, exc)
        key = status.split(":", 1)[0]
        counts[key] = counts.get(key, 0) + 1
        LOG.info("[%s] %s", case_dir.name, status)

    LOG.info("Backfill done: %s", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
