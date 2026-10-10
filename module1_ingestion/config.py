"""
Configuration settings for Module 1.

Centralizes paths, parameters, and constants.
"""

import numpy as np
from pathlib import Path

import os

# ---------------------------------------------------------
# Phase 3: STAC Configuration & Provider Settings
# ---------------------------------------------------------
DEFAULT_STAC_PROVIDER = os.getenv("STAC_PROVIDER", "copernicus")

COPERNICUS_STAC_URL = os.getenv("COPERNICUS_STAC_URL", "https://sh.dataspace.copernicus.eu/api/catalogs/stac")
SENTINEL_2_L2A_COLLECTION = "sentinel-2-l2a"

EARTH_SEARCH_STAC_URL = os.getenv("EARTH_SEARCH_STAC_URL", "https://earth-search.aws.element84.com/v1")
EARTH_SEARCH_COLLECTION = "sentinel-2-l2a"

# Band-to-Asset mappings for each STAC provider
COPERNICUS_BAND_TO_ASSET = {
    "B01": "B01",
    "B02": "B02",
    "B03": "B03",
    "B04": "B04",
    "B05": "B05",
    "B06": "B06",
    "B07": "B07",
    "B08": "B08",
    "B8A": "B8A",
    "B09": "B09",
    "B11": "B11",
    "B12": "B12"
}

EARTH_SEARCH_BAND_TO_ASSET = {
    "B01": "coastal",
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B09": "nir09",
    "B11": "swir16",
    "B12": "swir22"
}

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

# ---------------------------------------------------------
# Phase 7: Albedo / Reflection / Atmospheric Filtering
# ---------------------------------------------------------
ALBEDO_FILTER_ENABLED = True
GLARE_Z_THRESHOLD = 3.0           
SHADOW_Z_THRESHOLD = -1.5         
ATMOSPHERIC_Z_THRESHOLD = 2.5     

GLARE_REFLECTANCE_THRESHOLD = 4000.0
SHADOW_REFLECTANCE_THRESHOLD = 500.0
ATMOSPHERIC_REFLECTANCE_THRESHOLD = 2000.0

MIN_VALID_PIXEL_FRACTION = 0.05
FILTER_FILL_VALUE = 0.0

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
REAL_SAMPLE_DIR = DATA_DIR / "real_sample"
REAL_SAMPLE_TILES_DIR = REAL_SAMPLE_DIR / "tiles"
REAL_SAMPLE_INDEX_CSV = REAL_SAMPLE_DIR / "tiles_index.csv"
