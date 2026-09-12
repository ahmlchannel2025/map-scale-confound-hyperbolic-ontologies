"""Locations of data, results and checkpoints.

Every notebook in this project used to hard-code a Google-Drive path. All code
now resolves paths from one root, in this order:

1. the ``HME_ROOT`` environment variable, if set;
2. the repository root (the directory containing ``src/``), when running from a
   checkout;
3. the current working directory.

On Colab, mount Drive and ``export HME_ROOT=/content/drive/MyDrive/<repo>``.
"""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("HME_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "src").is_dir() and (p / "README.md").exists():
            return p
    return Path.cwd().resolve()


ROOT = project_root()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"

# Checkpoints written by the temperature / paper experiments (July 2026 onward).
# Historically ``/content/drive/MyDrive/dim_runs``; now inside the repo.
DIM_RUNS = RESULTS / "dim_runs"

NODES_CSV = DATA_PROCESSED / "icd10_nodes.csv"
TREE_PICKLE = DATA_PROCESSED / "icd10_tree_with_features.pkl"

ICD10_URL = "https://www.cms.gov/files/zip/2025-code-tables-tabular-and-index.zip"
