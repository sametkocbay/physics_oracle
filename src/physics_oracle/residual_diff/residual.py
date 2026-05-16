"""Differentiable per-cell residual — the PyTorch twin of the coded FO.

`DifferentiableResidual` reproduces `openfoam_setup/_residual_functions.py`
(`_CODE_WRITE`) with the FVM operators in `operators.py`: it evaluates the
explicit residual of the momentum, continuity, k and omega equations at a
prediction and is differentiable end-to-end, so the residual can be used as a
physics-informed training loss.

The kOmegaSST source-term algebra is ported verbatim from the coded FO;
`div`/`grad`/`laplacian` use the same `fvSchemes` as OpenFOAM.
"""
from __future__ import annotations

import torch
from torch import nn

from physics_oracle.residual_diff.boundary import BoundaryConditions
from physics_oracle.residual_diff.geometry import MeshGeometry
from physics_oracle.residual_diff.operators import (
    cell_limited_grad_V,
    face_flux,
    gauss_div_advect,
    gauss_div_tensor,
    gauss_grad,
    gauss_laplacian,
    interp_linear_upwind_V,
    interp_upwind,
    surface_integrate,
)

# kOmegaSST coefficients (OF13 Foundation defaults — see kOmegaSSTBase.C).
_AK1, _AK2 = 0.85, 1.0
_AW1, _AW2 = 0.5, 0.856
_G1, _G2 = 5.0 / 9.0, 0.44
_B1, _B2 = 0.075, 0.0828
_BETA_STAR, _A1, _B1C, _C1 = 0.09, 0.31, 1.0, 10.0


# ---- tensor algebra (OF conventions; grad_ij = dU_j/dx_i) ------------------

def _T(A):           # transpose
    return A.transpose(-1, -2)


def _tr(A):          # trace -> (...,)
    return A[..., 0, 0] + A[..., 1, 1] + A[..., 2, 2]


def _symm(A):
    return 0.5 * (A + _T(A))


def _twoSymm(A):
    return A + _T(A)


def _dev(A):
    eye = torch.eye(3, dtype=A.dtype, device=A.device)
    return A - (1.0 / 3.0) * _tr(A)[..., None, None] * eye


def _dev2(A):
    eye = torch.eye(3, dtype=A.dtype, device=A.device)
    return A - (2.0 / 3.0) * _tr(A)[..., None, None] * eye


def _magSqr(A):      # sum of squared components -> (...,)
    return (A * A).sum((-2, -1))


def _ddot(A, B):     # double inner product A && B -> (...,)
    return (A * B).sum((-2, -1))


class DifferentiableResidual(nn.Module):
    """Per-cell residual of the 4 governing equations, differentiable in the
    prediction tensors.

    Parameters
    ----------
    geom : MeshGeometry
        Mesh geometry exported from OpenFOAM (`residual_diff.geometry`).
    nu : float
        Laminar kinematic viscosity.
    inlet : InletConditions
        Freestream / inlet values (drives boundary conditions).
    """

    def __init__(self, geom: MeshGeometry, nu: float, inlet, *,
                 dtype: torch.dtype = torch.float64,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.geom = geom.to(device, dtype)
        self.nu = float(nu)
        self.dtype = dtype
        self.device = device
        self.bc = BoundaryConditions(self.geom, inlet, dtype=dtype, device=device)
        # airfoil wall-adjacent cells — masked, as in the coded FO.
        wall = next((p for p in self.geom.patches
                     if p["name"] == "airfoilWalls"), None)
        self._wall_cells = wall["cells"] if wall is not None else None

    def _owner_bnd(self, field):
        """Boundary face values = owner-cell values (for diffusivity fields)."""
        return [field[p["cells"]] for p in self.geom.patches]

    def forward(self, U: torch.Tensor, p: torch.Tensor, k: torch.Tensor,
                omega: torch.Tensor, nut: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``{momentumResidual, continuityResidual, kResidual,
        omegaResidual}`` — per-cell ``(Nc,)`` tensors."""
        geom, bc, nu = self.geom, self.bc, self.nu

        # ---- boundary values ----
        U_b = bc.values("U", U, U)
        p_b = bc.values("p", p, U)
        k_b = bc.values("k", k, U)
        w_b = bc.values("omega", omega, U)

        # ---- flux & gradients ----
        phi_i, phi_b = face_flux(U, U_b, geom)
        continuity = surface_integrate(phi_i, phi_b, geom)          # div(phi)
        gradU = cell_limited_grad_V(U, U_b, geom)                   # (Nc,3,3)
        gradp = gauss_grad(p, p_b, geom)                            # (Nc,3)
        gradk = gauss_grad(k, k_b, geom)                            # (Nc,3)
        gradw = gauss_grad(omega, w_b, geom)

        # ---- momentum ----
        nuEff = nu + nut
        nuEff_b = self._owner_bnd(nuEff)
        # bounded Gauss linearUpwindV div(phi,U)
        U_face = interp_linear_upwind_V(U, phi_i, gradU, geom)
        adv_U = (gauss_div_advect(phi_i, phi_b, U_face, U_b, geom)
                 - continuity.unsqueeze(-1) * U)
        lap_U = gauss_laplacian(nuEff, nuEff_b, U, U_b, gradU, geom)
        # stress term  div(nuEff * dev2(T(grad U)))   [Gauss linear]
        X = nuEff[:, None, None] * _dev2(_T(gradU))
        div_X = gauss_div_tensor(X, self._owner_bnd(X), geom)
        mom = adv_U + gradp - lap_U - div_X
        # eps-guarded norm: plain vector_norm has a 0/0 gradient at mom=0.
        momentumResidual = torch.sqrt((mom * mom).sum(-1) + 1.0e-30)

        # ---- kOmegaSST blending & production ----
        y = geom.y
        # +eps under the sqrt: sqrt(S2) has an infinite gradient at S2=0
        # (uniform far-field cells), which autograd turns into 0*inf = NaN.
        S2 = 2.0 * _magSqr(_symm(gradU)) + 1.0e-30
        GbyNu = _ddot(_dev(_twoSymm(gradU)), gradU)
        G = nut * GbyNu

        CDkOmega = (2.0 * _AW2) * (gradk * gradw).sum(-1) / omega
        CDkOmegaPlus = torch.clamp(CDkOmega, min=1.0e-10)
        arg1 = torch.clamp(
            torch.minimum(
                torch.maximum(
                    (1.0 / _BETA_STAR) * torch.sqrt(k) / (omega * y),
                    500.0 * nu / (y * y * omega),
                ),
                (4.0 * _AW2) * k / (CDkOmegaPlus * y * y),
            ),
            max=10.0,
        )
        F1 = torch.tanh(arg1 ** 4)
        arg2 = torch.clamp(
            torch.maximum(
                (2.0 / _BETA_STAR) * torch.sqrt(k) / (omega * y),
                500.0 * nu / (y * y * omega),
            ),
            max=100.0,
        )
        F23 = torch.tanh(arg2 ** 2)

        alphaK = F1 * (_AK1 - _AK2) + _AK2
        alphaOmega = F1 * (_AW1 - _AW2) + _AW2
        gamma = F1 * (_G1 - _G2) + _G2
        beta = F1 * (_B1 - _B2) + _B2
        DkEff = alphaK * nut + nu
        DomegaEff = alphaOmega * nut + nu

        # ---- k equation ----
        k_face = interp_upwind(k, phi_i, geom)
        adv_k = (gauss_div_advect(phi_i, phi_b, k_face, k_b, geom)
                 - continuity * k)
        lap_k = gauss_laplacian(DkEff, self._owner_bnd(DkEff), k, k_b, gradk, geom)
        Pk = torch.minimum(G, (_C1 * _BETA_STAR) * k * omega)
        kResidual = torch.abs(
            adv_k - lap_k - Pk
            + (2.0 / 3.0) * continuity * k
            + _BETA_STAR * omega * k
        )

        # ---- omega equation ----
        w_face = interp_upwind(omega, phi_i, geom)
        adv_w = (gauss_div_advect(phi_i, phi_b, w_face, w_b, geom)
                 - continuity * omega)
        lap_w = gauss_laplacian(DomegaEff, self._owner_bnd(DomegaEff),
                                omega, w_b, gradw, geom)
        omega_prod = gamma * torch.minimum(
            GbyNu,
            (_C1 / _A1) * _BETA_STAR * omega
            * torch.maximum(_A1 * omega, _B1C * F23 * torch.sqrt(S2)),
        )
        omegaResidual = torch.abs(
            adv_w - lap_w - omega_prod
            + (2.0 / 3.0) * gamma * continuity * omega
            + beta * omega * omega
            + (F1 - 1.0) * CDkOmega
        )

        # NB: the coded FO writes continuityResidual *signed* (div(phi)); only
        # momentum/k/omega are magnitudes.  Keep continuity signed to match.
        out = {
            "momentumResidual": momentumResidual,
            "continuityResidual": continuity,
            "kResidual": kResidual,
            "omegaResidual": omegaResidual,
        }
        # mask airfoil wall-function cells (matches the coded FO).
        if self._wall_cells is not None:
            for key in out:
                out[key] = out[key].clone()
                out[key][self._wall_cells] = 0.0
        return out
