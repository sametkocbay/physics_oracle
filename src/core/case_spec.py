"""§3 case naming, §5.3 inlet conditions, CaseSpec dataclass."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .envelope import C_MU, CHORD, NU, TURB_INTENSITY, TURB_LENGTH_FRAC
from .paths import CASES_DIR


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


@dataclass
class CaseSpec:
    case_id: str
    naca_code: str
    aoa_deg: float
    Re: float
    split: str
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
