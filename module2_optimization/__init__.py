"""
Module 2: Metaheuristic Optimization Core for Sentinel-2 Level-2A Spectral Band Selection.
"""

from .config import (
    BAND_NAMES,
    BAND_TO_INDEX,
    INDEX_TO_BAND,
    SPECTRUM_DOMAINS,
    TOTAL_BANDS,
    DEFAULT_SCORECARD_WEIGHTS,
    RAM_CEILING_GB,
    HARD_RAM_PENALTY
)
from .physics_partitioning import (
    validate_input_tensor,
    partition_spectrum
)

__all__ = [
    "BAND_NAMES",
    "BAND_TO_INDEX",
    "INDEX_TO_BAND",
    "SPECTRUM_DOMAINS",
    "TOTAL_BANDS",
    "DEFAULT_SCORECARD_WEIGHTS",
    "RAM_CEILING_GB",
    "HARD_RAM_PENALTY",
    "validate_input_tensor",
    "partition_spectrum"
]
