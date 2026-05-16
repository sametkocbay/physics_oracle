"""Differentiable finite-volume operators in PyTorch.

These reproduce the OpenFOAM `fvc::` operators for the schemes used by the case
(`configs/openfoam.yaml` `fvSchemes`): `linear`/`upwind`/`linearUpwindV`
interpolation, `Gauss` gradient (plain and `cellLimited`), `Gauss` divergence
with the `bounded` modifier, and `Gauss linear corrected` laplacian.

Every operator is a scatter-add over mesh faces keyed by `owner`/`neighbour`
(`torch.Tensor.index_add_`, autograd-friendly).  Geometry comes from
`MeshGeometry` (constants); the field tensors carry the gradient.

Field shapes: scalar `(Nc,)`, vector `(Nc, 3)`, tensor `(Nc, 3, 3)`.
Boundary values are passed per operator as a list of per-patch tensors
(`bvals[i]` aligns with `geom.patches[i]`).
"""
from __future__ import annotations

import torch

from physics_oracle.residual_diff.geometry import MeshGeometry

_SMALL = 1.0e-6      # OF `small`
_VSMALL = 1.0e-300   # OF `vSmall`


# ---------------------------------------------------------------------------
# Scatter helper
# ---------------------------------------------------------------------------

def _scatter(internal: torch.Tensor, bnd: list[torch.Tensor | None],
             geom: MeshGeometry) -> torch.Tensor:
    """Sum face contributions to cells: +owner / -neighbour for internal faces,
    +owner for boundary faces.  `internal` is (Fi, ...), `bnd[i]` is (nb_i, ...).

    Uses out-of-place `index_add` so the chain stays autograd-safe.
    """
    out = torch.zeros((geom.n_cells, *internal.shape[1:]),
                      dtype=internal.dtype, device=internal.device)
    out = out.index_add(0, geom.owner, internal)
    out = out.index_add(0, geom.neigh, -internal)
    for patch, contrib in zip(geom.patches, bnd):
        if contrib is not None:
            out = out.index_add(0, patch["cells"], contrib)
    return out


def _w(geom: MeshGeometry, ndim_extra: int) -> torch.Tensor:
    """Owner interpolation weight, shaped to broadcast over `ndim_extra` axes."""
    return geom.w.reshape(geom.w.shape + (1,) * ndim_extra)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interp_linear(field: torch.Tensor, geom: MeshGeometry) -> torch.Tensor:
    """Linear interpolation to internal faces: w·owner + (1-w)·neighbour."""
    w = _w(geom, field.dim() - 1)
    return w * field[geom.owner] + (1.0 - w) * field[geom.neigh]


def face_flux(U: torch.Tensor, U_bnd: list[torch.Tensor],
              geom: MeshGeometry) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Volumetric face flux phi = U_f · Sf  (internal + per-patch boundary)."""
    phi_int = (interp_linear(U, geom) * geom.Sf).sum(-1)
    phi_bnd = [(ub * p["Sf"]).sum(-1) for ub, p in zip(U_bnd, geom.patches)]
    return phi_int, phi_bnd


def interp_upwind(field: torch.Tensor, phi_int: torch.Tensor,
                  geom: MeshGeometry) -> torch.Tensor:
    """Upwind interpolation to internal faces (owner if phi≥0 else neighbour)."""
    pos = (phi_int >= 0).reshape(phi_int.shape + (1,) * (field.dim() - 1))
    return torch.where(pos, field[geom.owner], field[geom.neigh])


def interp_linear_upwind_V(U: torch.Tensor, phi_int: torch.Tensor,
                           gradU: torch.Tensor, geom: MeshGeometry) -> torch.Tensor:
    """`linearUpwindV` interpolation of a vector field to internal faces.

    Upwind value + the limited gradient correction
    `(Cf − C[upwind]) & grad(U)[upwind]` (see OF `linearUpwindVTemplates.C`).
    """
    pos = phi_int >= 0
    up = torch.where(pos, geom.owner, geom.neigh)               # (Fi,)
    upwind_val = U[up]                                          # (Fi,3)

    d = geom.Cf - geom.C[up]                                    # (Fi,3)
    corr = torch.einsum("fi,fij->fj", d, gradU[up])             # (Fi,3)

    # V-limiter: cap the correction against the linear-interpolation jump.
    dU = U[geom.neigh] - U[geom.owner]
    maxCorr = torch.where(pos.unsqueeze(-1),
                          (1.0 - geom.w).unsqueeze(-1) * dU,
                          geom.w.unsqueeze(-1) * (-dU))
    sfCorrs = (corr * corr).sum(-1)                             # |corr|^2
    maxCorrs = (corr * maxCorr).sum(-1)                         # corr · maxCorr
    scale = torch.where(
        maxCorrs < 0,
        torch.zeros_like(sfCorrs),
        torch.where(sfCorrs > maxCorrs,
                    maxCorrs / (sfCorrs + _VSMALL),
                    torch.ones_like(sfCorrs)),
    )
    scale = torch.where(sfCorrs > 0, scale, torch.ones_like(sfCorrs))
    return upwind_val + scale.unsqueeze(-1) * corr


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------

def gauss_grad(field: torch.Tensor, field_bnd: list[torch.Tensor],
               geom: MeshGeometry) -> torch.Tensor:
    """`Gauss linear` gradient.  scalar→(Nc,3) vector, vector→(Nc,3,3) tensor
    with convention grad_ij = dU_j/dx_i (OF `Sf ⊗ U_f`)."""
    f_face = interp_linear(field, geom)
    if field.dim() == 1:                                   # scalar
        internal = f_face.unsqueeze(-1) * geom.Sf          # (Fi,3)
        bnd = [fb.unsqueeze(-1) * p["Sf"]
               for fb, p in zip(field_bnd, geom.patches)]
    else:                                                  # vector
        internal = geom.Sf.unsqueeze(-1) * f_face.unsqueeze(-2)   # (Fi,3,3)
        bnd = [p["Sf"].unsqueeze(-1) * fb.unsqueeze(-2)
               for fb, p in zip(field_bnd, geom.patches)]
    V = geom.V.reshape((geom.n_cells,) + (1,) * (internal.dim() - 1))
    return _scatter(internal, bnd, geom) / V


def cell_limited_grad_V(U: torch.Tensor, U_bnd: list[torch.Tensor],
                        geom: MeshGeometry) -> torch.Tensor:
    """`cellLimited Gauss linear 1` gradient of a vector field (minmod limiter,
    k=1) — see OF `cellLimitedGrad.C`.  Returns the (Nc,3,3) tensor gradient."""
    g = gauss_grad(U, U_bnd, geom)                          # (Nc,3,3) unlimited
    own, nei = geom.owner, geom.neigh

    # max/min of U over each cell and its face-neighbours (component-wise);
    # out-of-place index_reduce keeps the chain autograd-safe.
    maxV, minV = U, U
    for idx, src in ((own, U[nei]), (nei, U[own])):
        maxV = maxV.index_reduce(0, idx, src, "amax", include_self=True)
        minV = minV.index_reduce(0, idx, src, "amin", include_self=True)
    for ub, p in zip(U_bnd, geom.patches):
        maxV = maxV.index_reduce(0, p["cells"], ub, "amax", include_self=True)
        minV = minV.index_reduce(0, p["cells"], ub, "amin", include_self=True)
    maxV = maxV - U                                         # deltas (k=1)
    minV = minV - U

    # per-cell, per-component limiter = min over faces of minmod(r).
    arange3 = torch.arange(3, device=U.device)
    limiter = torch.ones(geom.n_cells * 3, dtype=U.dtype, device=U.device)

    def _apply(limiter: torch.Tensor, cell_idx: torch.Tensor,
               Cf: torch.Tensor) -> torch.Tensor:
        d = Cf - geom.C[cell_idx]                           # (F,3)
        extrap = torch.einsum("fi,fij->fj", d, g[cell_idx]) # (F,3) per component
        big = extrap > _SMALL
        small = extrap < -_SMALL
        denom = torch.where(big | small, extrap, torch.ones_like(extrap))
        r = torch.where(big, maxV[cell_idx] / denom,
                        torch.where(small, minV[cell_idx] / denom,
                                    torch.ones_like(extrap)))
        face_lim = torch.clamp(r, max=1.0)                  # minmod: min(r,1)
        idx = (cell_idx.unsqueeze(-1) * 3 + arange3).reshape(-1)
        return limiter.index_reduce(0, idx, face_lim.reshape(-1),
                                    "amin", include_self=True)

    limiter = _apply(limiter, own, geom.Cf)
    limiter = _apply(limiter, nei, geom.Cf)
    for p in geom.patches:
        limiter = _apply(limiter, p["cells"], p["Cf"])

    # scale each component's gradient row: g_ij *= limiter_j
    return g * limiter.reshape(U.shape).unsqueeze(-2)


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------

def surface_integrate(phi_int: torch.Tensor, phi_bnd: list[torch.Tensor],
                      geom: MeshGeometry) -> torch.Tensor:
    """`fvc::div(phi)` of a face-flux field → (Nc,) cell field."""
    return _scatter(phi_int, list(phi_bnd), geom) / geom.V


def gauss_div_advect(phi_int: torch.Tensor, phi_bnd: list[torch.Tensor],
                     psi_face: torch.Tensor, psi_bnd: list[torch.Tensor],
                     geom: MeshGeometry) -> torch.Tensor:
    """`Gauss` `div(phi, psi)` = (1/V) Σ ±phi·psi_face.  psi may be scalar/vector."""
    extra = psi_face.dim() - 1
    internal = phi_int.reshape(phi_int.shape + (1,) * extra) * psi_face
    bnd = [pb.reshape(pb.shape + (1,) * extra) * sb
           for pb, sb in zip(phi_bnd, psi_bnd)]
    V = geom.V.reshape((geom.n_cells,) + (1,) * extra)
    return _scatter(internal, bnd, geom) / V


def gauss_div_tensor(X: torch.Tensor, X_bnd: list[torch.Tensor],
                     geom: MeshGeometry) -> torch.Tensor:
    """`Gauss linear` `div` of a tensor field → (Nc,3) vector: (1/V) Σ ±(Sf & X_f)."""
    X_face = interp_linear(X, geom)                          # (Fi,3,3)
    internal = torch.einsum("fi,fij->fj", geom.Sf, X_face)   # (Fi,3)
    bnd = [torch.einsum("fi,fij->fj", p["Sf"], xb)
           for xb, p in zip(X_bnd, geom.patches)]
    return _scatter(internal, bnd, geom) / geom.V.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------

def gauss_laplacian(gamma: torch.Tensor, gamma_bnd: list[torch.Tensor],
                    field: torch.Tensor, field_bnd: list[torch.Tensor],
                    cell_grad: torch.Tensor, geom: MeshGeometry) -> torch.Tensor:
    """`Gauss linear corrected` laplacian(gamma, field).

    snGrad = nonOrthDeltaCoeffs·(field_N − field_P)
             + nonOrthCorrectionVector · interp(cell_grad).
    `cell_grad` is the cell gradient OF's `correctedSnGrad` uses for the
    non-orthogonal correction — the **named** grad scheme of the field, i.e.
    `cellLimited` `grad(U)` for laplacian(nuEff,U) and `Gauss linear`
    `grad(k)` / `grad(omega)` for the turbulence laplacians.  Using the
    *limited* gradient is essential: the cellLimited limiter zeroes the
    wall-adjacent gradient of a uniform field, so the correction vanishes
    there as it does in OpenFOAM.
    Boundary snGrad = patch deltaCoeffs·(field_b − field_P) (no correction).
    scalar field → (Nc,) ; vector field → (Nc,3).
    """
    grad_f = interp_linear(cell_grad, geom)                           # face grad
    gamma_face = interp_linear(gamma, geom)                           # (Fi,)
    fo, fn = field[geom.owner], field[geom.neigh]

    if field.dim() == 1:                                    # scalar
        corr = (geom.ncorr * grad_f).sum(-1)
        sn = geom.ndc * (fn - fo) + corr
        internal = gamma_face * geom.magSf * sn
    else:                                                   # vector
        corr = torch.einsum("fi,fij->fj", geom.ncorr, grad_f)
        sn = geom.ndc.unsqueeze(-1) * (fn - fo) + corr
        internal = (gamma_face * geom.magSf).unsqueeze(-1) * sn

    bnd = []
    for fb, gb, p in zip(field_bnd, gamma_bnd, geom.patches):
        fo_p = field[p["cells"]]
        if field.dim() == 1:
            sn_b = p["ndc"] * (fb - fo_p)
            bnd.append(gb * p["magSf"] * sn_b)
        else:
            sn_b = p["ndc"].unsqueeze(-1) * (fb - fo_p)
            bnd.append((gb * p["magSf"]).unsqueeze(-1) * sn_b)

    V = geom.V.reshape((geom.n_cells,) + (1,) * (internal.dim() - 1))
    return _scatter(internal, bnd, geom) / V
