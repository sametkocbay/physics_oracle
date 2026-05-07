"""Shared utilities for the CFD dataset generation pipeline.

All implementation decisions trace back to ReadMe.md ("Dataset Generation Guide").
Section references in comments (e.g. §5.3) point at that document.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
CASES_DIR = DATASET_ROOT / "cases"
SPLITS_DIR = DATASET_ROOT / "splits"
SCRIPTS_DIR = DATASET_ROOT / "scripts"
TEMPLATE_DIR = PROJECT_ROOT / "case_template"
MANIFEST_PATH = DATASET_ROOT / "manifest.yaml"
REJECTION_LOG_PATH = DATASET_ROOT / "rejection_log.csv"


# ---------------------------------------------------------------------------
# §1 operating envelope
# ---------------------------------------------------------------------------

CHORD = 1.0                 # all cases are chord-normalized
NU = 1.0e-5                 # kinematic viscosity (constant; Re sets U)
TURB_INTENSITY = 0.01       # I=0.01, §5.3 k formula
TURB_LENGTH_FRAC = 0.07     # L = 0.07 * chord, §5.3 omega formula
C_MU = 0.09                 # k-omega SST closure constant

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
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("cfd_data_generator")


# ---------------------------------------------------------------------------
# §3 case naming
# ---------------------------------------------------------------------------

CASE_ID_RE = re.compile(
    r"^NACA(?P<naca>\d{4})_(?P<sign>[pn])(?P<aoa>\d+\.\d)_(?P<re>\d+\.\d+e[+-]?\d+)$"
)


def format_aoa(aoa_deg: float) -> str:
    """+5.0 -> 'p5.0'; -2.5 -> 'n2.5'; 0.0 -> 'p0.0'  (§3 format rules)."""
    sign = "n" if aoa_deg < 0 else "p"
    return f"{sign}{abs(aoa_deg):.1f}"


def format_re(re_value: float) -> str:
    """Re=1.5e6 -> '1.5e6' (§3 examples)."""
    mantissa, exp = f"{re_value:.1e}".split("e")
    return f"{float(mantissa):.1f}e{int(exp)}"


def make_case_id(naca_code: str, aoa_deg: float, re_value: float) -> str:
    return f"NACA{naca_code}_{format_aoa(aoa_deg)}_{format_re(re_value)}"


def parse_case_id(case_id: str) -> dict:
    m = CASE_ID_RE.match(case_id)
    if not m:
        raise ValueError(f"Invalid case id: {case_id!r}")
    sign = -1.0 if m["sign"] == "n" else 1.0
    return {
        "naca_code": m["naca"],
        "aoa_deg": sign * float(m["aoa"]),
        "Re": float(m["re"]),
    }


# ---------------------------------------------------------------------------
# NACA 4-digit airfoil
# ---------------------------------------------------------------------------

def naca4_code(camber_pct: float, position_pct: float, thickness_pct: float) -> str:
    """Round continuous (camber%, position%, thickness%) to a 4-digit code.

    NACA MPXX:  M = camber%  (0–9)
                P = position of max camber in tenths (0–9)
                XX = thickness%  (00–99)
    """
    m = int(round(np.clip(camber_pct, 0, 9)))
    p = int(round(np.clip(position_pct / 10.0, 0, 9)))
    if m == 0:                  # symmetric airfoil → P must be 0
        p = 0
    xx = int(round(np.clip(thickness_pct, 1, 99)))
    return f"{m}{p}{xx:02d}"


def naca4_params(naca_code: str) -> tuple[float, float, float]:
    """code -> (max camber [frac], camber position [frac], thickness [frac])."""
    if len(naca_code) != 4 or not naca_code.isdigit():
        raise ValueError(f"NACA code must be 4 digits, got {naca_code!r}")
    m = int(naca_code[0]) / 100.0
    p = int(naca_code[1]) / 10.0
    t = int(naca_code[2:]) / 100.0
    return m, p, t


def naca4_coordinates(naca_code: str, n_points: int = 200, chord: float = CHORD) -> np.ndarray:
    """NACA 4-digit airfoil profile, ordered from trailing edge clockwise (§6.3).

    Cosine spacing on the chord — densest near LE and TE, where curvature is
    largest. Returns (2*N - 1, 2): TE → upper → LE → lower → TE.
    """
    m, p, t = naca4_params(naca_code)
    beta = np.linspace(0.0, np.pi, n_points)
    x = 0.5 * (1.0 - np.cos(beta))                             # cosine-spaced [0, 1]

    # Open trailing edge (the standard NACA "open TE" form, last coefficient
    # -0.1015 → finite TE thickness of ≈ 0.252 % chord).  We deliberately do
    # NOT use the closed-TE form (-0.1036) because a cusped TE collides under
    # boundary-layer extrusion on cambered airfoils, producing degenerate
    # cells / `defaultFaces` after gmshToFoam.  The open TE is also closer to
    # what real airfoils have in practice.
    yt = 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
    )

    # camber line and slope
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
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Order: upper TE -> upper surface -> LE -> lower surface -> lower TE.
    # With the open TE, upper TE (xu[-1], yu[-1]) and lower TE (xl[-1], yl[-1])
    # are distinct points (yu[-1] ≈ +0.0013·thickness, yl[-1] ≈ -0.0013·thickness),
    # so we keep both.  The LE duplicate is dropped because xu[0]=xl[0]=0,
    # yu[0]=yl[0]=0 there.
    upper = np.column_stack([xu[::-1], yu[::-1]])              # TE_upper .. LE
    lower = np.column_stack([xl[1:], yl[1:]])                  # LE+1 .. TE_lower
    coords = np.vstack([upper, lower]) * chord
    return coords


# ---------------------------------------------------------------------------
# §5.3 inlet conditions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InletConditions:
    U_mag: float
    U_x: float
    U_y: float
    k_inlet: float
    omega_inlet: float
    nu: float


def compute_inlet_conditions(
    aoa_deg: float, re_value: float, chord: float = CHORD, nu: float = NU
) -> InletConditions:
    """U from Re=U·chord/nu; k and omega from §5.3 formulas."""
    U_mag = re_value * nu / chord
    aoa_rad = math.radians(aoa_deg)
    U_x = U_mag * math.cos(aoa_rad)
    U_y = U_mag * math.sin(aoa_rad)

    k_inlet = 1.5 * (U_mag * TURB_INTENSITY) ** 2
    L = TURB_LENGTH_FRAC * chord
    omega_inlet = math.sqrt(k_inlet) / (C_MU ** 0.25 * L)

    return InletConditions(
        U_mag=U_mag, U_x=U_x, U_y=U_y,
        k_inlet=k_inlet, omega_inlet=omega_inlet, nu=nu,
    )


# ---------------------------------------------------------------------------
# §9 reproducibility — solver settings hash
# ---------------------------------------------------------------------------

def md5_of_paths(paths: Iterable[Path]) -> str:
    h = hashlib.md5()
    for p in sorted(Path(x) for x in paths):
        h.update(p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Case spec
# ---------------------------------------------------------------------------

@dataclass
class CaseSpec:
    case_id: str
    naca_code: str
    aoa_deg: float
    Re: float
    split: str                               # 'train' | 'val' | 'test' | 'ood_probe'
    flags: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, naca_code: str, aoa_deg: float, re_value: float, split: str) -> "CaseSpec":
        return cls(
            case_id=make_case_id(naca_code, aoa_deg, re_value),
            naca_code=naca_code,
            aoa_deg=float(aoa_deg),
            Re=float(re_value),
            split=split,
        )

    @property
    def case_dir(self) -> Path:
        return CASES_DIR / self.case_id

    @property
    def of_case_dir(self) -> Path:
        return self.case_dir / "of_case"

    def inlet(self) -> InletConditions:
        return compute_inlet_conditions(self.aoa_deg, self.Re)
