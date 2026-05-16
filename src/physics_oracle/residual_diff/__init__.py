"""Differentiable PyTorch reimplementation of the OpenFOAM residual.

`openfoam_setup/_residual_functions.py` computes the per-cell residual inside a
separate OpenFOAM process — accurate, but a black box to autograd.  This package
reimplements the same finite-volume residual in PyTorch so gradients flow from a
prediction tensor back to the residual (usable as a physics-informed loss).

The FVM operators use mesh geometry exported once from OpenFOAM (see
`geometry.export_mesh_geometry`), so they are exact on the geometry side.
"""
from physics_oracle.residual_diff.geometry import (
    MeshGeometry,
    export_mesh_geometry,
)
from physics_oracle.residual_diff.residual import DifferentiableResidual

__all__ = ["MeshGeometry", "export_mesh_geometry", "DifferentiableResidual"]
