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
    OVER_BUDGET_PENALTY_WEIGHT,
    DEFAULT_WOA_POP_SIZE,
    DEFAULT_WOA_MAX_ITER,
    DEFAULT_WOA_SPIRAL_B,
    DEFAULT_PATIENCE,
    DEFAULT_SEED
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
from .chaotic_seeding import (
    chebyshev_map_sequence,
    repair_mask,
    repair_population,
    generate_chaotic_population
)
from .bio_search import (
    sigmoid_transfer_function,
    binary_whale_optimization_search
)
from .orchestrator import (
    apply_domain_balance_guard,
    run_module2_optimization,
    save_band_mask
)
from .integration import (
    apply_module2_band_selection
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
    "DEFAULT_WOA_POP_SIZE",
    "DEFAULT_WOA_MAX_ITER",
    "DEFAULT_WOA_SPIRAL_B",
    "DEFAULT_PATIENCE",
    "DEFAULT_SEED",
    "validate_input_tensor",
    "partition_spectrum",
    "FitnessEvaluator",
    "generate_pseudo_labels",
    "compute_jm_distance",
    "fast_knn_accuracy",
    "chebyshev_map_sequence",
    "repair_mask",
    "repair_population",
    "generate_chaotic_population",
    "sigmoid_transfer_function",
    "binary_whale_optimization_search",
    "apply_domain_balance_guard",
    "run_module2_optimization",
    "save_band_mask",
    "apply_module2_band_selection"
]
