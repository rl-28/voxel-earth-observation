"""
Configuration constants and parameters for Module 2: Metaheuristic Optimization Core.
"""

from typing import Dict, List, Any
import numpy as np

# Total spectral bands in Sentinel-2 Level-2A standardized interface
TOTAL_BANDS = 13

# 0-Indexed Band Mapping
BAND_NAMES: List[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B10", "B11", "B12"
]

BAND_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(BAND_NAMES)}
INDEX_TO_BAND: Dict[int, str] = {i: name for i, name in enumerate(BAND_NAMES)}

# Physics-Guided Spectral Domain Partitions
SPECTRUM_DOMAINS: Dict[str, Dict[str, Any]] = {
    "VISIBLE": {
        "bands": ["B02", "B03", "B04"],
        "indices": [1, 2, 3],
        "description": "Blue, Green, Red bands for true-color and urban surface features"
    },
    "NIR_RED_EDGE": {
        "bands": ["B05", "B06", "B07", "B08", "B8A"],
        "indices": [4, 5, 6, 7, 8],
        "description": "Vegetation red edge and near-infrared bands for chlorophyll and biomass"
    },
    "SWIR": {
        "bands": ["B11", "B12"],
        "indices": [11, 12],
        "description": "Short-wave infrared bands for moisture, water penetration, and soil penetration"
    },
    "AUXILIARY": {
        "bands": ["B01", "B09", "B10"],
        "indices": [0, 9, 10],
        "description": "Coastal aerosol, water vapour, and cirrus atmospheric absorption bands"
    }
}

# Domain Name Aliases
DOMAIN_NAME_ALIASES: Dict[str, str] = {
    "VISIBLE": "VISIBLE",
    "RED_EDGE": "NIR_RED_EDGE",
    "NIR": "NIR_RED_EDGE",
    "NIR_RED_EDGE": "NIR_RED_EDGE",
    "SWIR": "SWIR",
    "AUXILIARY": "AUXILIARY"
}

# Expected Spatial-Spectral Tensor Contract
EXPECTED_HEIGHT = 256
EXPECTED_WIDTH = 256
EXPECTED_CHANNELS = 13
EXPECTED_DTYPE = np.float32

# Multi-Objective Fitness Scorecard Defaults
DEFAULT_SCORECARD_WEIGHTS: Dict[str, float] = {
    "accuracy": 0.65,
    "jm_separability": 0.25,
    "band_penalty": 0.10
}

# Hardware Resource & System Constraints
RAM_CEILING_GB = 16.0
HARD_RAM_PENALTY = 100.0  # Large step penalty applied when memory ceiling is exceeded

# Metaheuristic & Surrogate Defaults
DEFAULT_TARGET_BAND_COUNT = 8
DEFAULT_WOA_POP_SIZE = 20
DEFAULT_WOA_MAX_ITER = 50
DEFAULT_WOA_SPIRAL_B = 1.0
DEFAULT_PATIENCE = 10
DEFAULT_PSEUDO_LABEL_CLUSTERS = 6
DEFAULT_SUBSAMPLE_PIXELS = 1000
DEFAULT_SEED = 42
EPSILON_COV = 1e-6
