"""
Configuration settings for Module 1.

Centralizes paths, parameters, and constants.
"""

import numpy as np
from pathlib import Path

import os

# ---------------------------------------------------------
# Phase 3: STAC Configuration
# ---------------------------------------------------------
COPERNICUS_STAC_URL = os.getenv("COPERNICUS_STAC_URL", "https://sh.dataspace.copernicus.eu/api/catalogs/stac")
SENTINEL_2_L2A_COLLECTION = "sentinel-2-l2a"

# ---------------------------------------------------------
# Phase 5: Normalization Constants
# ---------------------------------------------------------
import numpy as np
ZSCORE_DTYPE = np.float32
ZSCORE_STAT_DTYPE = np.float64
ZSCORE_DDOF = 0
NORMALIZATION_FILL_VALUE = 0.0
DEFAULT_CLOUD_COVER = 20.0
STAC_TIMEOUT_SEC = 30

# ---------------------------------------------------------
# Phase 6: Wavelet Constants
# ---------------------------------------------------------
WAVELET_NAME = "db1"
WAVELET_LEVEL = 1
WAVELET_MODE = "symmetric"

# Important processing constants
TILE_SIZE = 256
NUM_BANDS = 13

# L2A Spectral Bands
# Note: Sentinel-2 L2A provides 12 surface-reflectance spectral bands (B10 is excluded).
# The project interface currently requires 13 channels (NUM_BANDS = 13).
# The mapping of the 13th channel is pending project-level specification clarification.
SENTINEL_2_L2A_SPECTRAL_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", 
    "B07", "B08", "B8A", "B09", "B11", "B12"
]
OUTPUT_DTYPE = np.float32
EXPECTED_SHAPE = (256, 256, 13)

# Project base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Relative data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
EVAL_DIR = DATA_DIR / "evaluation"
SERBIA_EVAL_DIR = EVAL_DIR / "serbia"
BALKANS_EVAL_DIR = EVAL_DIR / "western_balkans"
