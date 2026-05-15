"""NACA airfoil math + LHS sampling."""
from .naca import naca4_code, naca4_coordinates, naca4_params
from .sampling import (
    assign_splits,
    sample_cases,
    sample_fill_cases,
    sample_naca_profiles,
    sample_ood_cases,
)
