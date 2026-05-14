"""§5 — Build the OpenFOAM case directory tree for one case.

Reads configs/openfoam.yaml as the single source of truth for solver and
boundary-condition setup, substitutes per-case values (inlet velocity, k,
omega, lift/drag directions, endTime, writeInterval) into the loaded
structure, and renders 0/{U,p,k,omega,nut,phi}, constant/{momentumTransport,
physicalProperties}, system/{controlDict,fvSchemes,fvSolution} via the
of_writer dict serializer.

The 2D simulation uses a 1-cell-thick extruded mesh with frontAndBack=empty
and top/bottom=patch (consistent with the project's case_template/).
"""
from __future__ import annotations

import argparse
import copy
import math
from functools import lru_cache
from pathlib import Path

import yaml

from core.case_spec import CaseSpec, parse_case_id
from core.logging import setup_logging
from core.paths import OPENFOAM_CONFIG_PATH

from .of_writer import render_foam_dict

LOG = setup_logging()


@lru_cache(maxsize=1)
def _load_config() -> dict:
    return yaml.safe_load(OPENFOAM_CONFIG_PATH.read_text())


# ---------------------------------------------------------------------------
# Sentinel substitution
# ---------------------------------------------------------------------------

def _format_num(v: float) -> str:
    return f"{v:.10g}"


def _substitute(obj, mapping: dict[str, str]):
    """Recursively walk `obj` and replace sentinel strings."""
    if isinstance(obj, dict):
        return {k: _substitute(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(x, mapping) for x in obj]
    if isinstance(obj, str):
        out = obj
        for sentinel, value in mapping.items():
            out = out.replace(sentinel, value)
        # If the resulting string looks like a pure number, leave it as-is —
        # YAML parsers downstream don't see it; render_foam_dict treats all
        # str values uniformly.
        return out
    return obj


def _per_case_mapping(spec: CaseSpec, end_time: int, write_interval: int) -> dict[str, str]:
    inl = spec.inlet()
    aoa_rad = math.radians(spec.aoa_deg)
    return {
        "__U_X__":     _format_num(inl.U_x),
        "__U_Y__":     _format_num(inl.U_y),
        "__U_MAG__":   _format_num(inl.U_mag),
        "__K__":       _format_num(inl.k_inlet),
        "__OMEGA__":   _format_num(inl.omega_inlet),
        "__LIFT_X__":  _format_num(-math.sin(aoa_rad)),
        "__LIFT_Y__":  _format_num(math.cos(aoa_rad)),
        "__DRAG_X__":  _format_num(math.cos(aoa_rad)),
        "__DRAG_Y__":  _format_num(math.sin(aoa_rad)),
        "__END_TIME__":      str(end_time),
        "__WRITE_INTERVAL__": str(write_interval),
    }


def _coerce_numeric_keys(d: dict, keys: tuple[str, ...]):
    """Convert string-substituted numbers back to int for keys that must
    render without quotes (endTime, writeInterval)."""
    for k in keys:
        if k in d and isinstance(d[k], str) and d[k].lstrip("-").isdigit():
            d[k] = int(d[k])


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _write_field(zero_dir: Path, field: str, cls: str, body: dict) -> None:
    text = render_foam_dict(cls=cls, obj=field, body=body)
    (zero_dir / field).write_text(text)


def write_zero_dir(zero_dir: Path, spec: CaseSpec, cfg: dict, mapping: dict[str, str]) -> None:
    zero_dir.mkdir(parents=True, exist_ok=True)
    bc = _substitute(cfg["boundary_conditions"], mapping)
    field_classes = cfg["field_classes"]
    for field in ("U", "p", "k", "omega", "nut", "phi"):
        _write_field(zero_dir, field, field_classes[field], bc[field])


def write_constant_dir(constant_dir: Path, cfg: dict) -> None:
    constant_dir.mkdir(parents=True, exist_ok=True)
    (constant_dir / "momentumTransport").write_text(
        render_foam_dict("dictionary", "momentumTransport",
                         cfg["momentum_transport"])
    )
    nu = float(cfg["physical"]["nu"])
    (constant_dir / "physicalProperties").write_text(
        render_foam_dict("dictionary", "physicalProperties", {
            "viscosityModel": "constant",
            "nu": f"{nu:.6e}",
        })
    )


def write_system_dir(system_dir: Path, cfg: dict, mapping: dict[str, str],
                     end_time: int, write_interval: int) -> None:
    system_dir.mkdir(parents=True, exist_ok=True)

    cd = _substitute(cfg["control_dict"], mapping)
    _coerce_numeric_keys(cd, ("endTime", "writeInterval", "startTime", "deltaT",
                              "writePrecision", "timePrecision", "purgeWrite"))
    (system_dir / "controlDict").write_text(
        render_foam_dict("dictionary", "controlDict", cd)
    )

    (system_dir / "fvSchemes").write_text(
        render_foam_dict("dictionary", "fvSchemes", cfg["fv_schemes"])
    )
    (system_dir / "fvSolution").write_text(
        render_foam_dict("dictionary", "fvSolution", cfg["fv_solution"])
    )


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def setup_openfoam_case(of_case_dir: Path, spec: CaseSpec,
                        end_time: int = 5000, write_interval: int = 500) -> None:
    of_case_dir.mkdir(parents=True, exist_ok=True)

    # Guarantee at least one solution write at end_time, even for short runs.
    effective_interval = min(write_interval, end_time)
    if end_time % effective_interval != 0:
        effective_interval = end_time

    cfg = copy.deepcopy(_load_config())
    mapping = _per_case_mapping(spec, end_time, effective_interval)

    write_zero_dir(of_case_dir / "0", spec, cfg, mapping)
    write_constant_dir(of_case_dir / "constant", cfg)
    write_system_dir(of_case_dir / "system", cfg, mapping, end_time, effective_interval)
    LOG.info("[%s] OpenFOAM case files written under %s", spec.case_id, of_case_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write OpenFOAM case files for one case.")
    p.add_argument("case_id")
    p.add_argument("--of-case", required=True, type=Path)
    p.add_argument("--end-time", type=int, default=5000)
    p.add_argument("--split", default="train")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    parsed = parse_case_id(args.case_id)
    spec = CaseSpec.build(parsed["naca_code"], parsed["aoa_deg"], parsed["Re"], args.split)
    setup_openfoam_case(args.of_case, spec, end_time=args.end_time)


if __name__ == "__main__":
    main()
