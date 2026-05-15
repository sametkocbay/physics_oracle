"""physics_oracle — OpenFOAM-backed CFD oracle for NACA airfoils.

Dataset generation, ML warm-start, and residual evaluation.
"""
from physics_oracle.core import (
    ENVELOPE,
    OPENFOAM_CONFIG_PATH,
    POSTPROCESS_CONFIG_PATH,
    CaseSpec,
    setup_logging,
)
from physics_oracle.core.case_spec import compute_inlet_conditions
from physics_oracle.geometry.naca import naca4_coordinates
from physics_oracle.openfoam_setup.case_setup import (
    setup_case_with_initial_fields,
    setup_openfoam_case,
)
from physics_oracle.openfoam_setup.mesh_h5_to_polymesh import write_polymesh_from_h5
from physics_oracle.openfoam_setup.runner import run_simple_foam
from physics_oracle.cli.run_ml_initialized_step import (
    StepResult,
    build_full_fields,
    load_prediction,
    match_cells,
    run_step,
)

__all__ = [
    "CaseSpec",
    "ENVELOPE",
    "setup_logging",
    "OPENFOAM_CONFIG_PATH",
    "POSTPROCESS_CONFIG_PATH",
    "compute_inlet_conditions",
    "naca4_coordinates",
    "setup_openfoam_case",
    "setup_case_with_initial_fields",
    "write_polymesh_from_h5",
    "run_simple_foam",
    "run_step",
    "StepResult",
    "load_prediction",
    "match_cells",
    "build_full_fields",
]
