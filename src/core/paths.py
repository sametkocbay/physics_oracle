"""Filesystem paths for the dataset pipeline.

Resolved relative to the project root (the parent of `src/`).
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATASET_ROOT = PROJECT_ROOT / "dataset"
CASES_DIR = DATASET_ROOT / "cases"
SPLITS_DIR = DATASET_ROOT / "splits"
ML_DATASET_DIR = DATASET_ROOT / "ML_dataset"
MANIFEST_PATH = DATASET_ROOT / "manifest.yaml"
REJECTION_LOG_PATH = DATASET_ROOT / "rejection_log.csv"

OPENFOAM_CONFIG_PATH = CONFIGS_DIR / "openfoam.yaml"
POSTPROCESS_CONFIG_PATH = CONFIGS_DIR / "postprocess.yaml"
