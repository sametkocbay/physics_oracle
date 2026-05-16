"""Boundary face values for the differentiable residual.

The FVM operators (`operators.py`) need the field value on each boundary face.
This module evaluates those from the case's boundary-condition types
(`configs/openfoam.yaml` `boundary_conditions`) as a differentiable function of
the cell field — so autograd still flows through `zeroGradient`/`freestream`
patches where the boundary value depends on the interior field.

The BC set is fixed for this pipeline (NACA airfoil, kOmegaSST):

    patch         U            p            k             omega
    inlet         fixedValue   zeroGrad     fixedValue    fixedValue
    outlet        zeroGrad     fixedValue   zeroGrad      zeroGrad
    top/bottom    freestream   freestream   inletOutlet   inletOutlet
    airfoilWalls  noSlip(0)    zeroGrad     fixedValue(0) wallFunction*
    frontAndBack  empty (excluded from every operator)

`nut` is `calculated` (≈ owner value) and `omegaWallFunction` is treated as
zeroGradient — the airfoil-adjacent cells are masked in the residual anyway.
"""
from __future__ import annotations

import torch

from physics_oracle.residual_diff.geometry import MeshGeometry


class BoundaryConditions:
    """Evaluates per-patch boundary face values for U, p, k, omega, nut."""

    def __init__(self, geom: MeshGeometry, inlet, *,
                 dtype: torch.dtype = torch.float64,
                 device: str | torch.device = "cpu"):
        self.geom = geom
        self.dtype = dtype
        self.device = device
        self.U_inlet = torch.tensor([inlet.U_x, inlet.U_y, 0.0],
                                    dtype=dtype, device=device)
        self.k_inlet = float(inlet.k_inlet)
        self.omega_inlet = float(inlet.omega_inlet)
        # per-field, per-patch rule: ("fixed", val) | ("zg",) | ("freestream", val)
        #                            | ("inletOutlet", val)
        self._rules = {
            "U": {"inlet": ("fixed", self.U_inlet),
                  "outlet": ("zg",),
                  "top": ("freestream", self.U_inlet),
                  "bottom": ("freestream", self.U_inlet),
                  "airfoilWalls": ("fixed", torch.zeros(3, dtype=dtype, device=device))},
            "p": {"inlet": ("zg",), "outlet": ("fixed", 0.0),
                  "top": ("freestream", 0.0), "bottom": ("freestream", 0.0),
                  "airfoilWalls": ("zg",)},
            "k": {"inlet": ("fixed", self.k_inlet), "outlet": ("zg",),
                  "top": ("inletOutlet", self.k_inlet),
                  "bottom": ("inletOutlet", self.k_inlet),
                  "airfoilWalls": ("fixed", 0.0)},
            "omega": {"inlet": ("fixed", self.omega_inlet), "outlet": ("zg",),
                      "top": ("inletOutlet", self.omega_inlet),
                      "bottom": ("inletOutlet", self.omega_inlet),
                      "airfoilWalls": ("zg",)},
            "nut": {nm: ("zg",) for nm in
                    ("inlet", "outlet", "top", "bottom", "airfoilWalls")},
        }

    def _const(self, val, nb: int, vector: bool) -> torch.Tensor:
        if vector:
            v = val if torch.is_tensor(val) else torch.zeros(3, dtype=self.dtype,
                                                             device=self.device)
            return v.reshape(1, 3).expand(nb, 3).clone()
        return torch.full((nb,), float(val), dtype=self.dtype, device=self.device)

    def values(self, name: str, field: torch.Tensor,
               U: torch.Tensor) -> list[torch.Tensor]:
        """Boundary face values of `field` for every real patch (in `geom.patches`
        order).  `U` is the cell velocity, used for the boundary-flux sign that
        switches `freestream`/`inletOutlet` between inflow and outflow."""
        rules = self._rules[name]
        vector = field.dim() == 2
        out: list[torch.Tensor] = []
        for p in self.geom.patches:
            cells = p["cells"]
            nb = cells.numel()
            rule = rules.get(p["name"], ("zg",))
            owner_val = field[cells]
            kind = rule[0]
            if kind == "fixed":
                out.append(self._const(rule[1], nb, vector))
            elif kind == "zg":
                out.append(owner_val)
            else:  # freestream / inletOutlet — switch on outward face flux
                outflow = (U[cells] * p["Sf"]).sum(-1) > 0.0
                far = self._const(rule[1], nb, vector)
                mask = outflow.reshape((nb,) + (1,) * (field.dim() - 1))
                out.append(torch.where(mask, owner_val, far))
        return out
