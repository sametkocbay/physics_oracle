"""Filesystem paths for the pipeline.

Two distinct anchors:

* ``CONFIGS_DIR`` is *package data* — it lives next to this file inside the
  installed package, so it resolves correctly for source, editable, and wheel
  installs.
* ``DATASET_ROOT`` is *runtime output* and must never land inside
  ``site-packages``.  It defaults to ``<cwd>/dataset`` and can be overridden
  with the ``PHYSICS_ORACLE_DATASET_ROOT`` environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

# .../physics_oracle/  — the package directory (parent of core/).
_PACKAGE_DIR = Path(__file__).resolve().parent.parent

CONFIGS_DIR = _PACKAGE_DIR / "configs"
OPENFOAM_CONFIG_PATH = CONFIGS_DIR / "openfoam.yaml"
POSTPROCESS_CONFIG_PATH = CONFIGS_DIR / "postprocess.yaml"

# Runtime output root — caller-controlled, never inside the installed package.
DATASET_ROOT = Path(
    os.environ.get("PHYSICS_ORACLE_DATASET_ROOT", Path.cwd() / "dataset")
)
CASES_DIR = DATASET_ROOT / "cases"
SPLITS_DIR = DATASET_ROOT / "splits"
ML_DATASET_DIR = DATASET_ROOT / "ML_dataset"
MANIFEST_PATH = DATASET_ROOT / "manifest.yaml"
REJECTION_LOG_PATH = DATASET_ROOT / "rejection_log.csv"
