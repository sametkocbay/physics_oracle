"""Export and load the OpenFOAM mesh geometry the differentiable residual needs.

`export_mesh_geometry` writes a minimal OpenFOAM case, runs the `meshGeometry`
coded function object (`openfoam_setup/_geometry_export.py`) via `foamPostProcess`,
parses the dumped fields, and returns a `MeshGeometry` of torch tensors.  The
result is cached next to `mesh.h5` (`<mesh>.geom.pt`) keyed by a content hash —
geometry depends only on the mesh, so OpenFOAM runs at most once per mesh.

The geometry tensors are constants (no gradient w.r.t. the prediction); they let
the FVM operators in `operators.py` reproduce OF's `linear`/`Gauss`/`corrected`
schemes exactly.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from physics_oracle.core.logging import setup_logging
from physics_oracle.openfoam_setup._geometry_export import (
    GEOMETRY_FIELDS,
    MESH_GEOMETRY_FO,
)
from physics_oracle.openfoam_setup.case_setup import setup_openfoam_case
from physics_oracle.openfoam_setup.extract import _read_text
from physics_oracle.openfoam_setup.mesh_h5_to_polymesh import write_polymesh_from_h5
from physics_oracle.openfoam_setup.of_writer import render_foam_dict
from physics_oracle.openfoam_setup.runner import run_geometry_export

LOG = setup_logging()

_NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


# ---------------------------------------------------------------------------
# OpenFOAM ascii parsers
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _extract_braced(text: str, brace_idx: int) -> tuple[str, int]:
    """`text[brace_idx]` is '{'; return (inner, index-after-closing-brace)."""
    depth = 0
    for i in range(brace_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_idx + 1:i], i + 1
    raise ValueError("unbalanced braces")


def _iter_blocks(body: str):
    """Yield (name, inner) for every ``name { ... }`` block in `body`."""
    pat = re.compile(r"([A-Za-z0-9_]+)\s*\{")
    i = 0
    while True:
        m = pat.search(body, i)
        if not m:
            return
        inner, end = _extract_braced(body, m.end() - 1)
        yield m.group(1), inner
        i = end


def _parse_value(token: str) -> np.ndarray:
    """Parse an OpenFOAM ``uniform``/``nonuniform List<...>`` value token."""
    token = token.strip()
    if token.startswith("uniform"):
        rest = token[len("uniform"):].strip()
        if rest.startswith("("):
            return np.array([[float(x) for x in re.findall(_NUM_RE, rest)]],
                            dtype=np.float64)
        return np.array([float(rest)], dtype=np.float64)
    m = re.search(r"nonuniform\s+List<(\w+)>\s*\d*\s*\((.*)\)\s*$",
                  token, flags=re.DOTALL)
    if not m:
        raise ValueError(f"cannot parse value token: {token[:80]!r}")
    typ, payload = m.group(1), m.group(2)
    if typ == "scalar":
        return np.array([float(x) for x in re.findall(_NUM_RE, payload)],
                        dtype=np.float64)
    tuples = re.findall(r"\(([^)]*)\)", payload)
    return np.array([[float(x) for x in re.findall(_NUM_RE, t)] for t in tuples],
                    dtype=np.float64)


def parse_foam_field(text: str) -> tuple[np.ndarray, dict[str, tuple[str, np.ndarray | None]]]:
    """Parse a vol/surface field file → (internalField, {patch: (type, value)})."""
    text = _strip_comments(text)
    mi = re.search(r"internalField\s+(.*?);", text, flags=re.DOTALL)
    if not mi:
        raise ValueError("internalField not found")
    internal = _parse_value(mi.group(1))
    boundary: dict[str, tuple[str, np.ndarray | None]] = {}
    mb = re.search(r"boundaryField", text)
    if mb:
        brace = text.index("{", mb.end())
        body, _ = _extract_braced(text, brace)
        for name, inner in _iter_blocks(body):
            tm = re.search(r"\btype\s+(\w+)", inner)
            vm = re.search(r"\bvalue\s+(.*?);", inner, flags=re.DOTALL)
            boundary[name] = (
                tm.group(1) if tm else "",
                _parse_value(vm.group(1)) if vm else None,
            )
    return internal, boundary


def _body_after_foamfile(text: str) -> str:
    text = _strip_comments(text)
    _, end = _extract_braced(text, text.index("{"))
    return text[end:]


def parse_label_list(path: Path) -> np.ndarray:
    """Parse an OpenFOAM ``labelList`` file (owner / neighbour)."""
    body = _body_after_foamfile(_read_text(path))
    m = re.search(r"(\d+)\s*\(", body)
    start = m.end()
    end = body.index(")", start)
    return np.array(re.findall(r"-?\d+", body[start:end]), dtype=np.int64)


def parse_polymesh_boundary(path: Path) -> list[dict]:
    """Parse ``constant/polyMesh/boundary`` → ordered list of patch dicts."""
    body = _body_after_foamfile(_read_text(path))
    m = re.search(r"\d+\s*\((.*)\)\s*$", body, flags=re.DOTALL)
    patches = []
    for name, inner in _iter_blocks(m.group(1)):
        patches.append({
            "name": name,
            "type": re.search(r"\btype\s+(\w+)", inner).group(1),
            "nFaces": int(re.search(r"\bnFaces\s+(\d+)", inner).group(1)),
            "startFace": int(re.search(r"\bstartFace\s+(\d+)", inner).group(1)),
        })
    return patches


# ---------------------------------------------------------------------------
# MeshGeometry
# ---------------------------------------------------------------------------

@dataclass
class MeshGeometry:
    """FVM mesh geometry as torch tensors (constants — no autograd)."""
    n_cells: int
    # internal faces (Fi = number of internal faces)
    owner: torch.Tensor          # (Fi,)  long  — owner cell
    neigh: torch.Tensor          # (Fi,)  long  — neighbour cell
    Sf: torch.Tensor             # (Fi,3)       — face-area vector (owner→neigh)
    magSf: torch.Tensor          # (Fi,)        — face-area magnitude
    w: torch.Tensor              # (Fi,)        — linear interp weight (owner)
    ndc: torch.Tensor            # (Fi,)        — nonOrthDeltaCoeffs
    ncorr: torch.Tensor          # (Fi,3)       — nonOrth correction vectors
    Cf: torch.Tensor             # (Fi,3)       — face centres
    # cells (Nc = n_cells)
    V: torch.Tensor              # (Nc,)        — cell volumes
    C: torch.Tensor              # (Nc,3)       — cell centres
    y: torch.Tensor              # (Nc,)        — wall distance
    # real (non-empty) boundary patches
    patches: list                # [{name,type,cells,Sf,magSf,Cf,ndc}]

    def to(self, device=None, dtype=None) -> "MeshGeometry":
        """Move/cast tensors — float tensors to `dtype`, index tensors stay long."""
        def mv(t):
            t = t.to(device) if device is not None else t
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype)
            return t
        return MeshGeometry(
            n_cells=self.n_cells,
            owner=self.owner.to(device) if device is not None else self.owner,
            neigh=self.neigh.to(device) if device is not None else self.neigh,
            Sf=mv(self.Sf), magSf=mv(self.magSf), w=mv(self.w), ndc=mv(self.ndc),
            ncorr=mv(self.ncorr), Cf=mv(self.Cf),
            V=mv(self.V), C=mv(self.C), y=mv(self.y),
            patches=[{
                "name": p["name"], "type": p["type"],
                "cells": p["cells"].to(device) if device is not None else p["cells"],
                "Sf": mv(p["Sf"]), "magSf": mv(p["magSf"]),
                "Cf": mv(p["Cf"]), "ndc": mv(p["ndc"]),
            } for p in self.patches],
        )


def _bc_array(val: np.ndarray | None, n: int, dim: int) -> np.ndarray:
    """Broadcast a parsed boundary value to (n,) or (n,dim)."""
    if val is None:
        return np.zeros((n,) if dim == 1 else (n, dim), dtype=np.float64)
    val = np.asarray(val, dtype=np.float64)
    if dim == 1:
        val = val.reshape(-1)
        return np.full(n, val[0]) if val.size == 1 else val
    val = val.reshape(-1, dim) if val.size != dim else val.reshape(1, dim)
    return np.broadcast_to(val, (n, dim)).copy() if val.shape[0] == 1 else val


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _parse_geometry(geom_case: Path) -> MeshGeometry:
    """Parse the geometry fields written by the meshGeometry FO + the polyMesh."""
    zero = geom_case / "0"
    polymesh = geom_case / "constant" / "polyMesh"

    owner_full = parse_label_list(polymesh / "owner")
    neigh = parse_label_list(polymesh / "neighbour")
    pm_patches = parse_polymesh_boundary(polymesh / "boundary")
    n_internal = len(neigh)

    fields = {}
    for name in GEOMETRY_FIELDS:
        fields[name] = parse_foam_field(_read_text(zero / name))

    V = fields["cellVolume"][0]
    C = fields["Cv"][0]
    y = fields["wallY"][0]
    n_cells = len(V)

    def internal(name):
        return fields[name][0][:n_internal]

    geom = MeshGeometry(
        n_cells=int(n_cells),
        owner=torch.from_numpy(owner_full[:n_internal].copy()),
        neigh=torch.from_numpy(neigh.copy()),
        Sf=torch.from_numpy(internal("Sf").copy()),
        magSf=torch.from_numpy(internal("magSf").copy()),
        w=torch.from_numpy(internal("meshWeights").copy()),
        ndc=torch.from_numpy(internal("nonOrthDeltaCoeffs").copy()),
        ncorr=torch.from_numpy(internal("nonOrthCorrectionVectors").copy()),
        Cf=torch.from_numpy(internal("Cf").copy()),
        V=torch.from_numpy(V.copy()),
        C=torch.from_numpy(C.copy()),
        y=torch.from_numpy(y.copy()),
        patches=[],
    )
    for p in pm_patches:
        if p["type"] == "empty":
            continue
        nm, nf, s = p["name"], p["nFaces"], p["startFace"]
        cells = owner_full[s:s + nf]
        geom.patches.append({
            "name": nm,
            "type": p["type"],
            "cells": torch.from_numpy(cells.copy()),
            "Sf": torch.from_numpy(_bc_array(fields["Sf"][1].get(nm, ("", None))[1], nf, 3)),
            "magSf": torch.from_numpy(_bc_array(fields["magSf"][1].get(nm, ("", None))[1], nf, 1)),
            "Cf": torch.from_numpy(_bc_array(fields["Cf"][1].get(nm, ("", None))[1], nf, 3)),
            "ndc": torch.from_numpy(_bc_array(
                fields["nonOrthDeltaCoeffs"][1].get(nm, ("", None))[1], nf, 1)),
        })
    return geom


def _hash_file(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def _inject_geometry_fo(controldict: Path) -> None:
    """Replace controlDict's `functions` block with the meshGeometry FO."""
    text = re.sub(r"functions\s*\{.*\}\s*$", "", controldict.read_text(),
                  flags=re.DOTALL)
    fo = render_foam_dict("dictionary", "_", {"meshGeometry": MESH_GEOMETRY_FO})
    text += "\nfunctions\n{\n" + fo[fo.index("meshGeometry"):] + "}\n"
    controldict.write_text(text)


def export_mesh_geometry(
    mesh_h5: Path,
    work_dir: Path,
    spec,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    use_cache: bool = True,
) -> MeshGeometry:
    """Return the OpenFOAM mesh geometry for `mesh_h5` as a `MeshGeometry`.

    Cached next to `mesh.h5` (`<mesh>.geom.pt`, keyed by file hash); OpenFOAM is
    run (once) only on a cache miss.  `spec` is any `CaseSpec` for the case — it
    drives the throwaway `0/` fields; geometry itself is mesh-only.
    """
    mesh_h5, work_dir = Path(mesh_h5), Path(work_dir)
    file_hash = _hash_file(mesh_h5)
    cache = mesh_h5.parent / (mesh_h5.stem + ".geom.pt")

    if use_cache and cache.exists():
        try:
            blob = torch.load(cache, weights_only=False)
            if blob.get("hash") == file_hash:
                LOG.info("Loaded cached mesh geometry: %s", cache)
                return blob["geom"].to(device, dtype)
            LOG.info("Geometry cache stale (mesh changed) — re-exporting")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Geometry cache unreadable (%s) — re-exporting", exc)

    geom_case = work_dir / "geom_case"
    if geom_case.exists():
        shutil.rmtree(geom_case)
    LOG.info("Exporting mesh geometry via OpenFOAM → %s", geom_case)
    write_polymesh_from_h5(mesh_h5, geom_case / "constant" / "polyMesh")
    setup_openfoam_case(geom_case, spec, end_time=1, write_interval=1)
    _inject_geometry_fo(geom_case / "system" / "controlDict")

    rc = run_geometry_export(geom_case)
    if rc != 0:
        raise RuntimeError(
            f"mesh geometry export failed (foamPostProcess exit {rc}); "
            f"see {geom_case / 'geometry.log'}")

    geom = _parse_geometry(geom_case)
    LOG.info("Mesh geometry: %d cells, %d internal faces, %d boundary patches",
             geom.n_cells, geom.owner.numel(), len(geom.patches))

    if use_cache:
        try:
            torch.save({"hash": file_hash, "geom": geom}, cache)
            LOG.info("Cached mesh geometry: %s", cache)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Could not write geometry cache %s (%s)", cache, exc)

    return geom.to(device, dtype)
