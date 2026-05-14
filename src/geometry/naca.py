"""NACA 4-digit airfoil math."""
from __future__ import annotations

import numpy as np

from core.envelope import CHORD


def naca4_code(camber_pct: float, position_pct: float, thickness_pct: float) -> str:
    """Round continuous (camber%, position%, thickness%) to a 4-digit code.

    NACA MPXX:  M = camber%  (0–9)
                P = position of max camber in tenths (0–9)
                XX = thickness%  (00–99)
    """
    m = int(round(np.clip(camber_pct, 0, 9)))
    p = int(round(np.clip(position_pct / 10.0, 0, 9)))
    if m == 0:
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
    x = 0.5 * (1.0 - np.cos(beta))

    # Open trailing edge (the standard NACA "open TE" form, last coefficient
    # -0.1015 → finite TE thickness of ≈ 0.252 % chord).  We deliberately do
    # NOT use the closed-TE form (-0.1036) because a cusped TE collides under
    # boundary-layer extrusion on cambered airfoils, producing degenerate
    # cells / `defaultFaces` after gmshToFoam.
    yt = 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
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
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    upper = np.column_stack([xu[::-1], yu[::-1]])
    lower = np.column_stack([xl[1:], yl[1:]])
    coords = np.vstack([upper, lower]) * chord
    return coords
