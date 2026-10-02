"""
module2_optimization/orchestrator.py

Main execution orchestrator for Module 2: Metaheuristic Optimization Core.

Execution Flow:
1. Input Tensor Validation & Dimension Normalization
2. Physics-Guided Spectral Partitioning
3. Multi-Objective Green-AI Fitness Evaluator Setup
4. Chebyshev Chaotic Seeding with Opposition-Based Learning (OBL)
5. Binary Whale Optimization Algorithm (BWOA)
6. Domain-Balance Guard (Ensuring critical physics domains: VISIBLE, NIR, SWIR are represented)
7. Final Metrics & Computation Savings Estimation
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .config import (
    TOTAL_BANDS,
    INDEX_TO_BAND,
    SPECTRUM_DOMAINS,
    DEFAULT_TARGET_BAND_COUNT,
    DEFAULT_WOA_POP_SIZE,
    DEFAULT_WOA_MAX_ITER,
    DEFAULT_SCORECARD_WEIGHTS
)
from .physics_partitioning import validate_input_tensor, partition_spectrum
from .fitness_scorecard import FitnessEvaluator
from .chaotic_seeding import generate_chaotic_population, repair_mask
from .bio_search import binary_whale_optimization_search

logger = logging.getLogger(__name__)

# Essential physics domains required for balanced remote sensing analysis
CRITICAL_DOMAINS = ["VISIBLE", "NIR_RED_EDGE", "SWIR"]
REPRESENTATIVE_DOMAIN_BANDS = {
    "VISIBLE": 3,      # B04 (Red)
    "NIR_RED_EDGE": 7, # B08 (NIR)
    "SWIR": 11         # B11 (SWIR-1)
}


def apply_domain_balance_guard(
    mask: np.ndarray,
    target_band_count: int,
    evaluator: FitnessEvaluator,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Guarantees balanced spectral representation across critical electromagnetic domains:
    VISIBLE, NIR_RED_EDGE, and SWIR.

    If a critical domain has 0 active bands, replaces a redundant band from an over-represented
    domain with the primary band of the unrepresented domain.

    Args:
        mask: 1D binary array of shape (13,).
        target_band_count: Maximum permissible active bands.
        evaluator: FitnessEvaluator to compare and validate candidate replacements.
        rng: Optional random generator for tie-breaking.

    Returns:
        np.ndarray: Domain-balanced binary mask of length 13.
    """
    balanced_mask = np.asarray(mask, dtype=np.int32).copy().ravel()
    if rng is None:
        rng = np.random.default_rng()

    # Domain representation count
    for domain in CRITICAL_DOMAINS:
        domain_indices = SPECTRUM_DOMAINS[domain]["indices"]
        active_in_domain = [idx for idx in domain_indices if balanced_mask[idx] == 1]

        if len(active_in_domain) == 0:
            rep_band = REPRESENTATIVE_DOMAIN_BANDS[domain]
            current_active = np.where(balanced_mask == 1)[0]

            if len(current_active) < target_band_count:
                balanced_mask[rep_band] = 1
            else:
                # Find an over-represented domain with > 1 active bands
                over_represented_candidates = []
                for other_dom in SPECTRUM_DOMAINS.keys():
                    other_indices = SPECTRUM_DOMAINS[other_dom]["indices"]
                    other_active = [idx for idx in other_indices if balanced_mask[idx] == 1]
                    if len(other_active) > 1:
                        over_represented_candidates.extend(other_active)

                if over_represented_candidates:
                    # Remove one candidate and insert missing domain's band
                    victim_band = rng.choice(over_represented_candidates)
                    test_mask = balanced_mask.copy()
                    test_mask[victim_band] = 0
                    test_mask[rep_band] = 1

                    # Only accept if valid and within budget
                    if np.sum(test_mask) <= target_band_count:
                        balanced_mask = test_mask

    return repair_mask(balanced_mask, target_band_count, rng=rng)


def run_module2_optimization(
    input_tensor: np.ndarray,
    target_count: int = DEFAULT_TARGET_BAND_COUNT,
    weights: Optional[Dict[str, float]] = None,
    labels: Optional[np.ndarray] = None,
    enforce_domain_balance: bool = True,
    pop_size: int = DEFAULT_WOA_POP_SIZE,
    max_iter: int = DEFAULT_WOA_MAX_ITER,
    seed: int = 0
) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    """
    End-to-end execution pipeline for Module 2 Metaheuristic Band Selection.

    FLOP Savings Formula:
    ---------------------
    In modern hybrid convolutional and vision transformer (CNN-ViT) architectures,
    early patch-embedding and initial 2D/3D convolution layers scale linearly with the
    number of input spectral channels C_in.
    The estimated reduction in early computation is computed as:
        FLOP Savings (%) = ((13 - k_selected) / 13) * 100%
        Data Volume Reduction (%) = ((13 - k_selected) / 13) * 100%

    Args:
        input_tensor: 3D (256, 256, 13) or 4D (N, 256, 256, 13) float32 tensor.
        target_count: Requested number of spectral bands (clamped to [4, 13]).
        weights: Multi-objective fitness scorecard weights.
        labels: Optional (256, 256) or (N, 256, 256) land cover class labels.
        enforce_domain_balance: If True, activates physics domain representation guard.
        pop_size: Population size for Whale Optimization (default: 20).
        max_iter: Maximum search iterations (default: 50).
        seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple[np.ndarray, List[int], Dict[str, Any]]:
            - optimal_band_mask: 1D binary array of length 13 (dtype: int32).
            - selected_indices: Sorted list of 0-based integer band indices.
            - optimization_metrics: Detailed dictionary with classification accuracy,
              Jeffries-Matusita separability, FLOP savings, data reduction percentage,
              fitness score, and convergence history.
    """
    logger.info(f"Initializing Module 2 Metaheuristic Band Selection (seed={seed})...")

    # 1. Validate Input Tensor
    tensor = validate_input_tensor(input_tensor)

    # 2. Boundary Check on target_count
    if target_count < 4 or target_count > TOTAL_BANDS:
        clamped_count = max(4, min(int(target_count), TOTAL_BANDS))
        logger.warning(
            f"Requested target_band_count ({target_count}) is outside the feasible range [4, 13]. "
            f"Clamped to {clamped_count}."
        )
        effective_target = clamped_count
    else:
        effective_target = int(target_count)

    # 3. Physics Partitioning
    partitions = partition_spectrum(tensor)
    logger.debug(f"Input spectrum successfully partitioned into {len(partitions)} physical domains.")

    # 4. Multi-Objective Fitness Evaluator
    evaluator = FitnessEvaluator(
        tensor=tensor,
        target_band_count=effective_target,
        weights=weights or DEFAULT_SCORECARD_WEIGHTS,
        labels=labels,
        seed=seed
    )

    # 5. Chaotic Population Seeding with OBL
    initial_population = generate_chaotic_population(
        num_agents=pop_size,
        target_band_count=effective_target,
        evaluator=evaluator,
        seed=seed
    )

    # 6. Binary Whale Optimization Algorithm Search
    best_mask, best_fitness, history = binary_whale_optimization_search(
        evaluator=evaluator,
        target_band_count=effective_target,
        pop_size=pop_size,
        max_iter=max_iter,
        initial_population=initial_population,
        seed=seed
    )

    # 7. Physics Domain Balance Guard
    if enforce_domain_balance:
        rng = np.random.default_rng(seed)
        best_mask = apply_domain_balance_guard(
            mask=best_mask,
            target_band_count=effective_target,
            evaluator=evaluator,
            rng=rng
        )

    # 8. Extract Final Metrics
    final_fitness, detailed_metrics = evaluator.evaluate(best_mask)
    selected_indices = sorted(list(np.where(best_mask == 1)[0]))
    k_selected = len(selected_indices)
    selected_bands = [INDEX_TO_BAND[idx] for idx in selected_indices]

    data_reduction_pct = float(((TOTAL_BANDS - k_selected) / TOTAL_BANDS) * 100.0)
    flop_savings_pct = float(((TOTAL_BANDS - k_selected) / TOTAL_BANDS) * 100.0)

    domain_distribution: Dict[str, int] = {}
    for d_name, d_info in SPECTRUM_DOMAINS.items():
        domain_distribution[d_name] = sum(1 for idx in selected_indices if idx in d_info["indices"])

    optimization_metrics: Dict[str, Any] = {
        "accuracy": detailed_metrics["accuracy"],
        "jm_distance": detailed_metrics["jm_distance"],
        "jm_normalized": detailed_metrics["jm_normalized"],
        "selected_bands": selected_bands,
        "selected_indices": selected_indices,
        "selected_band_count": k_selected,
        "target_band_count": effective_target,
        "data_reduction_pct": data_reduction_pct,
        "flop_savings_pct": flop_savings_pct,
        "fitness": final_fitness,
        "convergence_history": history,
        "domain_distribution": domain_distribution,
        "ram_penalty": detailed_metrics["ram_penalty"],
        "band_penalty": detailed_metrics["total_band_penalty"]
    }

    logger.info(
        f"Module 2 Optimization complete: selected {k_selected} bands {selected_bands} "
        f"with {data_reduction_pct:.1f}% data reduction and fitness {final_fitness:.4f}."
    )

    return best_mask, selected_indices, optimization_metrics
