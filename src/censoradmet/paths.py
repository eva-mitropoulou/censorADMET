"""Repository-local paths, with optional user-provided data locations."""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CENSORADMET_DATA_DIR", ROOT / "data"))
RESULTS_DIR = ROOT / "results"
CACHE_DIR = Path(os.environ.get("CENSORADMET_CACHE_DIR", ROOT / ".cache"))
MEASUREMENT_PARQUET = DATA_DIR / "measurement_records.parquet"
