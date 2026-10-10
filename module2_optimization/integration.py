"""
module2_optimization/integration.py

Integration pipeline connecting Stage 0 (Execution Contract), Module 1 (Ingestion Tensor),
and Module 2 (Whale Optimization Band Selection) to prepare reduced tensors for Module 3.
"""

import logging
from typing import Tuple, List, Dict, Optional, Any, Union
import numpy as np

from .config import (
    TOTAL_BANDS,
    DEFAULT_TARGET_BAND_COUNT,
    DEFAULT_SCORECARD_WEIGHTS,
    DEFAULT_WOA_POP_SIZE,
    DEFAULT_WOA_MAX_ITER,
    DEFAULT_SEED
)
from .physics_partitioning import validate_input_tensor
from .orchestrator import run_module2_optimization

try:
    from utils.llm_parser import VoxelExecutionContract
    HAS_STAGE0 = True
except ImportError:
    VoxelExecutionContract = None
    HAS_STAGE0 = False

logger = logging.getLogger(__name__)


def apply_module2_band_selection(
    tensor_13_bands: np.ndarray,
    contract: Optional[Any] = None,
    target_band_count: Optional[int] = None,
    scorecard_weights: Optional[Dict[str, float]] = None,
    labels: Optional[np.ndarray] = None,
    enable_module2: bool = True,
    enforce_domain_balance: bool = True,
    pop_size: int = DEFAULT_WOA_POP_SIZE,
    max_iter: int = DEFAULT_WOA_MAX_ITER,
    seed: int = DEFAULT_SEED
) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    """
    Executes Module 2 band selection after Module 1 ingestion.

    Reads parameters from VoxelExecutionContract (if provided), optimizes spectral band
    allocation via Binary Whale Optimization, and applies the optimal mask to produce a
    reduced spatial-spectral tensor of shape (256, 256, k) or (N, 256, 256, k) for Module 3.

    Args:
        tensor_13_bands: Validated Module 1 output tensor (256, 256, 13) or (N, 256, 256, 13).
        contract: Optional VoxelExecutionContract from Stage 0 parser.
        target_band_count: Optional explicit band budget (overrides contract if specified).
        scorecard_weights: Optional scorecard weights dictionary.
        labels: Optional (256, 256) or (N, 256, 256) land cover class labels.
        enable_module2: Feature flag (if False, bypasses optimization and returns original tensor).
        enforce_domain_balance: If True, activates physics domain representation guard.
        pop_size: Whale Optimization population size.
        max_iter: Whale Optimization maximum search iterations.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple[np.ndarray, List[int], Dict[str, Any]]:
            - reduced_tensor: Sliced tensor of shape (..., k) containing only optimal bands.
            - selected_indices: Sorted list of active 0-based band indices.
            - optimization_metrics: Scorecard and performance metrics dictionary.
    """
    validated_tensor = validate_input_tensor(tensor_13_bands)

    # Feature Flag Check: Bypass Module 2 if disabled
    if not enable_module2:
        logger.info("Module 2 band selection disabled via feature flag. Passing full 13-band tensor.")
        all_indices = list(range(TOTAL_BANDS))
        bypass_metrics = {
            "module2_enabled": False,
            "selected_band_count": TOTAL_BANDS,
            "selected_indices": all_indices,
            "data_reduction_pct": 0.0,
            "flop_savings_pct": 0.0
        }
        return validated_tensor, all_indices, bypass_metrics

    # Resolve Target Band Count & Weights from Stage 0 Contract or Arguments
    effective_target = DEFAULT_TARGET_BAND_COUNT
    effective_weights = DEFAULT_SCORECARD_WEIGHTS.copy()

    if contract is not None:
        if hasattr(contract, "target_band_count") and contract.target_band_count is not None:
            effective_target = contract.target_band_count
        if hasattr(contract, "scorecard_weights") and contract.scorecard_weights is not None:
            effective_weights = contract.scorecard_weights

    if target_band_count is not None:
        effective_target = target_band_count
    if scorecard_weights is not None:
        effective_weights = scorecard_weights

    # Run Module 2 Metaheuristic Optimization
    best_mask, selected_indices, metrics = run_module2_optimization(
        input_tensor=validated_tensor,
        target_count=effective_target,
        weights=effective_weights,
        labels=labels,
        enforce_domain_balance=enforce_domain_balance,
        pop_size=pop_size,
        max_iter=max_iter,
        seed=seed
    )

    # Output Hand-Off: Slice input tensor along the spectral channel dimension
    # (256, 256, 13) -> (256, 256, k) or (N, 256, 256, 13) -> (N, 256, 256, k)
    reduced_tensor = validated_tensor[..., selected_indices]
    metrics["module2_enabled"] = True
    metrics["output_shape"] = reduced_tensor.shape

    logger.info(
        f"Module 2 Handoff: Tensor reduced from {validated_tensor.shape} to {reduced_tensor.shape} "
        f"with {len(selected_indices)} bands ({metrics['data_reduction_pct']:.1f}% reduction)."
    )

    return reduced_tensor, selected_indices, metrics
