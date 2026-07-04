"""Shared core utilities — paths, envelope, case spec, logging."""
from .case_spec import (
    CaseSpec,
    InletConditions,
    compute_inlet_conditions,
    format_aoa,
    format_re,
    make_case_id,
    parse_case_id,
)
from .envelope import (
    CHORD,
    C_MU,
    ENVELOPE,
    MESH_VERSION,
    NU,
    OOD_BUCKET_WEIGHTS,
    OOD_ENVELOPE,
    OPENFOAM_VERSION,
    TURB_INTENSITY,
    TURB_LENGTH_FRAC,
)
from .logging import setup_logging
from .paths import (
    CASES_DIR,
    CONFIGS_DIR,
    DATASET_ROOT,
    MANIFEST_PATH,
    ML_DATASET_DIR,
    OPENFOAM_CONFIG_PATH,
    POSTPROCESS_CONFIG_PATH,
    REJECTION_LOG_PATH,
    SPLITS_DIR,
)
from .repro import md5_of_paths
