"""Optional function-object blocks injected when residual capture is enabled.

OpenFOAM 13 Foundation does NOT have ``solverInfo writeResidualFields`` —
that's an ESI/OpenCFD fork feature.  Foundation's ``residuals`` function
object only writes scalar (per-iteration) initial residuals to
``postProcessing/residuals/0/residuals.dat``; it does not emit volScalarFields.

For now ``setup_case_with_initial_fields(enable_residual_capture=True)`` is a
no-op on the controlDict (it already has a ``residuals`` block that captures
the scalar residual signal).  Per-cell spatial residuals will require either
ESI OF or a custom function-object — out of scope for the first round-trip.
"""
from __future__ import annotations

# Intentionally empty — kept as a hook for future spatial-residual capture.
SOLVER_INFO_BLOCK: dict = {}
