"""
Configuration settings for Module 1.

Centralizes paths, parameters, and constants.
"""

import numpy as np
from pathlib import Path

import os

# STAC Configuration
COPERNICUS_STAC_URL = os.getenv("COPERNICUS_STAC_URL", "https://stac.dataspace.copernicus.eu/v1/")
SENTINEL_2_L2A_COLLECTION = "sentinel-2-l2a"
DEFAULT_CLOUD_COVER = 20.0
STAC_TIMEOUT_SEC = 30

# Important processing constants
TILE_SIZE = 256
NUM_BANDS = 13
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
