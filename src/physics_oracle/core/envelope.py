"""§1 operating envelope and physical constants.

All values are read from configs/openfoam.yaml so the YAML is the single
source of truth for nu, chord, and turbulence parameters.
"""
from __future__ import annotations

import os
from functools import lru_cache

import yaml

from .paths import OPENFOAM_CONFIG_PATH


def _env_float(key: str, default: float) -> float:
    """Optional environment override for an OOD-envelope bound (float)."""
    val = os.environ.get(key)
    return float(val) if val not in (None, "") else default


ENVELOPE = {
    "aoa_min_deg": -5.0,
    "aoa_max_deg": 5.0,
    "re_min": 1.0e5,
    "re_max": 5.0e5,
    "camber_pct_min": 0.0,
    "camber_pct_max": 6.0,
    "camber_pos_pct_min": 20.0,
    "camber_pos_pct_max": 60.0,
    "thickness_pct_min": 8.0,
    "thickness_pct_max": 18.0,
}

OPENFOAM_VERSION = "v13"
MESH_VERSION = "v1"


# ---------------------------------------------------------------------------
# Out-of-distribution (OOD) probe envelope
#
# The OOD set is built from conditions the in-domain envelope above never sees,
# WITHOUT leaving the Reynolds band the C-mesh was validated for (first-cell
# height is fixed, so pushing Re higher would drive y+ past the wall-function
# limit and QC would reject the cases anyway).  OOD-ness therefore comes from:
#   * atypical *geometry* not present in the trained profiles
#       - thinner than 8 % chord,
#       - thicker than 18 % chord,
#       - more cambered than 6 % chord,
#   * higher |AoA| than the trained +/-5 deg, which together with the strong
#     camber yields higher |Cl| (lift AND downforce, via +/- AoA) than anything
#     in the in-domain set.
# Re is held inside ENVELOPE's [re_min, re_max] so the mesh stays valid.
# ---------------------------------------------------------------------------

OOD_ENVELOPE = {
    # Re kept inside the mesh-valid / trained band on purpose.  Overridable via
    # OOD_RE_MIN / OOD_RE_MAX, but pushing Re past the trained band drives y+
    # out of wall-function validity and QC will reject the cases.
    "re_min": _env_float("OOD_RE_MIN", ENVELOPE["re_min"]),
    "re_max": _env_float("OOD_RE_MAX", ENVELOPE["re_max"]),
    # |AoA| beyond the trained +/-5 deg (capped to keep steady RANS convergeable).
    # Override the band per run via OOD_AOA_ABS_MIN / OOD_AOA_ABS_MAX, e.g. a
    # stricter 12-15 deg probe:  OOD_AOA_ABS_MIN=12 OOD_AOA_ABS_MAX=15 ./generate_ood
    "aoa_abs_min_deg": _env_float("OOD_AOA_ABS_MIN", 6.0),
    "aoa_abs_max_deg": _env_float("OOD_AOA_ABS_MAX", 12.0),
    # Geometry ranges that lie outside the trained box.
    "thin_thickness_pct": (4.0, 7.0),     # below thickness_pct_min (8)
    "thick_thickness_pct": (19.0, 26.0),  # above thickness_pct_max (18)
    "high_camber_pct": (7.0, 9.0),        # above camber_pct_max (6)
    "camber_pos_pct": (20.0, 60.0),       # same position band as in-domain
}

# How the requested OOD count is split across atypical-geometry families.
# Weights are normalised at sampling time, so they need not sum to the target.
OOD_BUCKET_WEIGHTS = {
    "thin": 0.28,         # very thin sections at high AoA
    "thick": 0.36,        # thick sections at high AoA
    "high_camber": 0.36,  # strongly cambered -> highest |Cl|
}


@lru_cache(maxsize=1)
def _openfoam_config() -> dict:
    return yaml.safe_load(OPENFOAM_CONFIG_PATH.read_text())


def _physical() -> dict:
    return _openfoam_config()["physical"]


def _turbulence() -> dict:
    return _openfoam_config()["turbulence"]


def chord() -> float:
    return float(_physical()["chord"])


def nu() -> float:
    return float(_physical()["nu"])


def turb_intensity() -> float:
    return float(_turbulence()["intensity"])


def turb_length_fraction() -> float:
    return float(_turbulence()["length_fraction"])


def c_mu() -> float:
    return float(_turbulence()["c_mu"])


# Module-level constants for callers that prefer attribute access.  They are
# resolved lazily on first import (after configs/openfoam.yaml exists).
CHORD = chord()
NU = nu()
TURB_INTENSITY = turb_intensity()
TURB_LENGTH_FRAC = turb_length_fraction()
C_MU = c_mu()
