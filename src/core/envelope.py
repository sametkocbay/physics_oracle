"""§1 operating envelope and physical constants.

All values are read from configs/openfoam.yaml so the YAML is the single
source of truth for nu, chord, and turbulence parameters.
"""
from __future__ import annotations

from functools import lru_cache

import yaml

from .paths import OPENFOAM_CONFIG_PATH


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
