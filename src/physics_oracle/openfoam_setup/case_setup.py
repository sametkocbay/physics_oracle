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

from physics_oracle.core.case_spec import CaseSpec, parse_case_id
from physics_oracle.core.logging import setup_logging
from physics_oracle.core.paths import OPENFOAM_CONFIG_PATH

from ._residual_functions import RESIDUAL_FO_BLOCK
from .of_writer import (
    render_foam_dict,
    render_nonuniform_scalar,
    render_nonuniform_vector,
)

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
    startup = _load_config().get("startup", {})
    u_limit = float(startup.get("u_limit_factor", 4.0)) * inl.U_mag
    omega_min = float(startup.get("omega_floor_factor", 1.0e-3)) * inl.omega_inlet
    return {
        "__U_LIM__":     _format_num(u_limit),
        "__OMEGA_MIN__": _format_num(omega_min),
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
        "__NU__":      _format_num(float(_load_config()["physical"]["nu"])),
    }


def _coerce_numeric_keys(d: dict, keys: tuple[str, ...]):
    """Convert string-substituted numbers back to int for keys that must
    render without quotes (endTime, writeInterval)."""
    for k in keys:
        if k in d and isinstance(d[k], str) and d[k].lstrip("-").isdigit():
            d[k] = int(d[k])


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge where `override` wins.  An *empty* dict override
    replaces the base value entirely (needed for `residualControl: {}` to
    clear the production thresholds); non-empty dicts merge key-wise."""
    out = dict(base)
    for k, v in override.items():
        if (isinstance(v, dict) and v
                and isinstance(out.get(k), dict)):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


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
# Two-stage startup continuation (configs/openfoam.yaml `startup:` section)
# ---------------------------------------------------------------------------

def startup_config() -> dict:
    """The `startup:` section of openfoam.yaml ({} if absent)."""
    return _load_config().get("startup", {}) or {}


def write_stage_system(of_case_dir: Path, spec: CaseSpec, stage: str,
                       stage_a_end: int, total_end: int,
                       write_interval: int = 500) -> None:
    """Rewrite system/{controlDict,fvSchemes,fvSolution,fvConstraints} for one
    stage of the two-stage startup continuation.

    Stage "A": first-order momentum, heavy under-relaxation, no
    residualControl exit, limitMag(U) + bound(omega) fvConstraints, runs
    [0, stage_a_end] with intermediate writes so a crash still leaves a
    restart point.

    Stage "B": production schemes/solution from the top-level config,
    startFrom latestTime, endTime `total_end`, and an *empty* fvConstraints
    (must overwrite the stage-A file, not merely delete it).
    """
    if stage not in ("A", "B"):
        raise ValueError(f"stage must be 'A' or 'B', got {stage!r}")
    system_dir = of_case_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(_load_config())
    startup = cfg.get("startup", {}) or {}
    stage_a = startup.get("stage_a", {}) or {}

    if stage == "A":
        end_time = int(stage_a.get("iterations", 400))
        # Intermediate writes so an early death still leaves a restart point;
        # purgeWrite in the production controlDict keeps only the latest.
        interval = max(1, min(100, end_time))
        fv_schemes = _deep_merge(cfg["fv_schemes"], stage_a.get("fv_schemes", {}))
        fv_solution = _deep_merge(cfg["fv_solution"], stage_a.get("fv_solution", {}))
        constraints = stage_a.get("fv_constraints", {}) or {}
        start_from = "startTime"
    else:
        end_time = int(total_end)
        interval = min(write_interval, end_time)
        if end_time % interval != 0:
            interval = end_time
        stage_b = startup.get("stage_b", {}) or {}
        fv_schemes = _deep_merge(cfg["fv_schemes"], stage_b.get("fv_schemes", {}))
        fv_solution = _deep_merge(cfg["fv_solution"], stage_b.get("fv_solution", {}))
        constraints = {}          # overwrite the stage-A constraints file
        start_from = "latestTime"

    mapping = _per_case_mapping(spec, end_time, interval)

    cd = _substitute(cfg["control_dict"], mapping)
    cd["startFrom"] = start_from
    _coerce_numeric_keys(cd, ("endTime", "writeInterval", "startTime", "deltaT",
                              "writePrecision", "timePrecision", "purgeWrite"))
    (system_dir / "controlDict").write_text(
        render_foam_dict("dictionary", "controlDict", cd))
    (system_dir / "fvSchemes").write_text(
        render_foam_dict("dictionary", "fvSchemes", fv_schemes))
    (system_dir / "fvSolution").write_text(
        render_foam_dict("dictionary", "fvSolution", _substitute(fv_solution, mapping)))
    (system_dir / "fvConstraints").write_text(
        render_foam_dict("dictionary", "fvConstraints",
                         _substitute(constraints, mapping)))
    LOG.info("[%s] stage-%s system dicts written (endTime=%d)",
             spec.case_id, stage, end_time)


# ---------------------------------------------------------------------------
# ML-warm-started variant — nonuniform initialFields + optional residual capture
# ---------------------------------------------------------------------------

def _patch_internal_field(bc_field: dict, rendered: str) -> None:
    """Replace the `internalField` entry in a BC tree with a pre-rendered
    nonuniform list string.  The OF dict renderer emits whatever string we
    give it verbatim after `internalField   `, so we can pass the whole
    `nonuniform List<scalar>\\n N\\n (\\n ... \\n)` blob as a single value."""
    bc_field["internalField"] = rendered


def write_zero_dir_with_fields(
    zero_dir: Path,
    spec: CaseSpec,
    cfg: dict,
    mapping: dict[str, str],
    fields: dict,
) -> None:
    """Like write_zero_dir, but `internalField` for each rendered field comes
    from `fields` (numpy arrays) instead of the per-case scalar in the YAML."""
    zero_dir.mkdir(parents=True, exist_ok=True)
    bc = _substitute(cfg["boundary_conditions"], mapping)
    field_classes = cfg["field_classes"]

    # Patch internalField for each ML-provided field.
    if "U" in fields:
        _patch_internal_field(bc["U"], render_nonuniform_vector(fields["U"]))
    if "p" in fields:
        _patch_internal_field(bc["p"], render_nonuniform_scalar(fields["p"]))
    if "k" in fields:
        _patch_internal_field(bc["k"], render_nonuniform_scalar(fields["k"]))
    if "omega" in fields:
        _patch_internal_field(bc["omega"], render_nonuniform_scalar(fields["omega"]))
    if "nut" in fields:
        _patch_internal_field(bc["nut"], render_nonuniform_scalar(fields["nut"]))
    # phi is intentionally left uniform 0 — OF rebuilds it from U.

    for field in ("U", "p", "k", "omega", "nut", "phi"):
        _write_field(zero_dir, field, field_classes[field], bc[field])


def setup_case_with_initial_fields(
    of_case_dir: Path,
    spec: CaseSpec,
    fields: dict,
    *,
    end_time: int,
    enable_residual_capture: bool,
    write_interval: int | None = None,
) -> None:
    """Write a complete OpenFOAM case with ML-provided initial fields.

    Parameters
    ----------
    of_case_dir
        Target directory.  Must already contain `constant/polyMesh/` if the
        caller plans to run foamRun afterwards.
    spec
        Case spec — drives the boundary-condition values (inlet U, k, omega).
    fields
        Dict with keys ``U`` (N, 3), ``p`` (N,), ``k`` (N,), ``omega`` (N,),
        ``nut`` (N,).  Any missing key falls back to the uniform inlet value.
    end_time
        Solver endTime (1 for residual-only mode, ~500 for a warm-start run).
    enable_residual_capture
        If True, replace controlDict's ``functions`` with the single
        ``residualFields`` coded function object (see ``_residual_functions``),
        which writes per-cell ``momentumResidual``, ``continuityResidual``,
        ``kResidual`` and ``omegaResidual`` volScalarFields.  The forceCoeffs /
        yPlus / residuals objects are dropped — they are not needed for a
        residual-only evaluation (and yPlus needs a constructed turbulence
        model).  Evaluate the case with ``runner.run_residual_postprocess``.
    write_interval
        controlDict writeInterval.  Defaults to `end_time` (write once at the
        end), which is what both residual-only and short warm-start runs want.
    """
    of_case_dir.mkdir(parents=True, exist_ok=True)
    write_interval = write_interval if write_interval is not None else end_time

    cfg = copy.deepcopy(_load_config())
    mapping = _per_case_mapping(spec, end_time, write_interval)

    if enable_residual_capture:
        # Residual-only evaluation: the coded FO is the sole function object.
        cfg["control_dict"]["functions"] = {
            "residualFields": copy.deepcopy(RESIDUAL_FO_BLOCK)
        }
    # else: keep the standard forceCoeffs / residuals / yPlus blocks from
    # configs/openfoam.yaml for a normal warm-start run.

    write_zero_dir_with_fields(of_case_dir / "0", spec, cfg, mapping, fields)
    write_constant_dir(of_case_dir / "constant", cfg)
    write_system_dir(of_case_dir / "system", cfg, mapping, end_time, write_interval)
    LOG.info("[%s] ML-warm-started case written under %s (residuals=%s, endTime=%d)",
             spec.case_id, of_case_dir, enable_residual_capture, end_time)


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
