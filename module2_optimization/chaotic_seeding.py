"""
module2_optimization/chaotic_seeding.py

Chebyshev chaotic population initialization with Opposition-Based Learning (OBL)
and constraint-enforced mask repair for metaheuristic spectral band selection.

Mathematical Formulation:
- Chebyshev Chaotic Map: x_{k+1} = cos(d · arccos(x_k)), x ∈ [-1, 1]
- Normalization: y = (x + 1) / 2 ∈ [0, 1]
- Binarization: Bit = 1 if y >= 0.5 else 0
- Opposition-Based Learning: X_opp = 1 - X
- Repair Step: Ensures 1 <= ||X||_1 <= target_band_count
"""

import logging
from typing import Optional, List, Tuple
import numpy as np

from .config import (
    TOTAL_BANDS,
    DEFAULT_WOA_POP_SIZE,
    DEFAULT_TARGET_BAND_COUNT,
    DEFAULT_SEED
)
from .fitness_scorecard import FitnessEvaluator

logger = logging.getLogger(__name__)


def chebyshev_map_sequence(
    length: int,
    x0: Optional[float] = None,
    degree: int = 4,
    seed: int = DEFAULT_SEED
) -> np.ndarray:
    """
    Generates a sequence of chaotic values using the Chebyshev map.
    x_{k+1} = cos(degree * arccos(x_k))

    Args:
        length: Total number of sequential values to generate.
        x0: Initial seed state in (-1, 1) excluding fixed points {0, -1, 1}.
        degree: Degree of the Chebyshev polynomial (default: 4).
        seed: Random seed used if x0 is not explicitly provided.

    Returns:
        np.ndarray: Array of values in [-1, 1] of shape (length,).
    """
    if length <= 0:
        return np.empty(0, dtype=np.float64)

    if x0 is None:
        rng = np.random.default_rng(seed)
        # Avoid fixed points {0, -1, 1, ±0.5}
        x0 = float(rng.uniform(0.15, 0.85))
    else:
        x0 = float(np.clip(x0, -0.9999, 0.9999))
        if abs(x0) < 1e-4:
            x0 = 0.37

    sequence = np.zeros(length, dtype=np.float64)
    x = x0

    for i in range(length):
        x = np.clip(x, -1.0, 1.0)
        x = np.cos(degree * np.arccos(x))
        sequence[i] = x

    return sequence


def repair_mask(
    mask: np.ndarray,
    target_band_count: int,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Enforces the spectral budget constraint: 1 <= ||mask||_1 <= target_band_count.

    Args:
        mask: 1D binary array of length 13.
        target_band_count: Maximum permissible active bands.
        rng: Optional NumPy random generator for deterministic bit-flipping.

    Returns:
        np.ndarray: Repaired binary mask of length 13 (dtype: np.int32).
    """
    repaired = np.asarray(mask, dtype=np.int32).copy().ravel()
    if rng is None:
        rng = np.random.default_rng()

    active_indices = np.where(repaired == 1)[0]
    k_active = len(active_indices)
    max_bands = max(1, min(int(target_band_count), TOTAL_BANDS))

    # Case 1: Empty mask -> Activate at least 1 band randomly
    if k_active < 1:
        chosen_idx = rng.choice(TOTAL_BANDS)
        repaired[chosen_idx] = 1

    # Case 2: Over-budget -> Randomly deactivate excess bands
    elif k_active > max_bands:
        n_excess = k_active - max_bands
        deactivate_indices = rng.choice(active_indices, size=n_excess, replace=False)
        repaired[deactivate_indices] = 0

    return repaired


def repair_population(
    population: np.ndarray,
    target_band_count: int,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Applies the repair step row-by-row across an entire population matrix.

    Args:
        population: 2D binary array of shape (num_agents, 13).
        target_band_count: Maximum permissible active bands.
        rng: Optional random generator.

    Returns:
        np.ndarray: Repaired 2D binary population of shape (num_agents, 13).
    """
    repaired_pop = np.zeros_like(population, dtype=np.int32)
    for i in range(len(population)):
        repaired_pop[i] = repair_mask(population[i], target_band_count, rng=rng)
    return repaired_pop


def generate_chaotic_population(
    num_agents: int = DEFAULT_WOA_POP_SIZE,
    target_band_count: int = DEFAULT_TARGET_BAND_COUNT,
    evaluator: Optional[FitnessEvaluator] = None,
    seed: int = DEFAULT_SEED
) -> np.ndarray:
    """
    Generates a diverse initial binary population using Chebyshev chaotic mapping,
    Opposition-Based Learning (OBL) selection, and budget repair.

    Args:
        num_agents: Number of population agents to generate (default: 20).
        target_band_count: Target spectral band budget (e.g. 4, 6, 5, 8).
        evaluator: Optional FitnessEvaluator instance for OBL fitness comparison.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: Binary population array of shape (num_agents, 13) with dtype int32.
    """
    rng = np.random.default_rng(seed)
    total_elements = num_agents * TOTAL_BANDS

    # 1. Generate continuous Chebyshev chaotic sequence
    chaotic_seq = chebyshev_map_sequence(
        length=total_elements,
        seed=seed
    )

    # 2. Map from [-1, 1] to [0, 1] and reshape to (num_agents, 13)
    unit_mapped = (chaotic_seq + 1.0) / 2.0
    continuous_pop = unit_mapped.reshape(num_agents, TOTAL_BANDS)

    # 3. Binarize using 0.5 threshold
    binary_pop = (continuous_pop >= 0.5).astype(np.int32)

    # 4. Enforce repair on initial chaotic population
    repaired_pop = repair_population(binary_pop, target_band_count, rng=rng)

    # 5. Opposition-Based Learning (OBL) Enhancement
    if evaluator is not None:
        # Generate opposite population: X_opp = 1 - X
        opp_pop = 1 - repaired_pop
        repaired_opp_pop = repair_population(opp_pop, target_band_count, rng=rng)

        # Evaluate both candidate sets: 2 * num_agents
        combined_pop = np.vstack([repaired_pop, repaired_opp_pop])
        scores = np.array([evaluator.get_score(ind) for ind in combined_pop])

        # Rank and select top N individuals
        top_indices = np.argsort(scores)[::-1][:num_agents]
        selected_population = combined_pop[top_indices]
        return selected_population.astype(np.int32)

    return repaired_pop.astype(np.int32)
