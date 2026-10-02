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
    HARD_RAM_PENALTY,
    DEFAULT_TARGET_BAND_COUNT,
    EMPTY_MASK_PENALTY,
    OVER_BUDGET_PENALTY_WEIGHT
)
from .physics_partitioning import (
    validate_input_tensor,
    partition_spectrum
)
from .fitness_scorecard import (
    FitnessEvaluator,
    generate_pseudo_labels,
    compute_jm_distance,
    fast_knn_accuracy
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
    "DEFAULT_TARGET_BAND_COUNT",
    "EMPTY_MASK_PENALTY",
    "OVER_BUDGET_PENALTY_WEIGHT",
    "validate_input_tensor",
    "partition_spectrum",
    "FitnessEvaluator",
    "generate_pseudo_labels",
    "compute_jm_distance",
    "fast_knn_accuracy"
]
