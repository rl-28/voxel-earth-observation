"""
module2_optimization/bio_search.py

Binary Whale Optimization Algorithm (BWOA) for Sentinel-2 spectral band selection.

Key Mechanisms:
- Sigmoid Transfer Function: S(v) = 1 / (1 + e^(-v)) for mapping continuous velocities to bit probabilities.
- Three Biological Search Behaviors:
  1. Encircling Prey: D = |C · X* - X|, X_new = X* - A · D
  2. Exploration Phase: D = |C · X_rand - X|, X_new = X_rand - A · D (when |A| >= 1)
  3. Logarithmic Bubble-Net Spiral: X_new = D' · e^(b·l) · cos(2πl) + X*
- Parameter 'a' decays linearly from 2.0 to 0.0.
- Strict constraint repair after every agent position update.
- Convergence history tracking and optional patience-based early stopping.
"""

import logging
from typing import Tuple, List, Optional
import numpy as np

from .config import (
    TOTAL_BANDS,
    DEFAULT_WOA_POP_SIZE,
    DEFAULT_WOA_MAX_ITER,
    DEFAULT_WOA_SPIRAL_B,
    DEFAULT_PATIENCE,
    DEFAULT_TARGET_BAND_COUNT,
    DEFAULT_SEED
)
from .fitness_scorecard import FitnessEvaluator
from .chaotic_seeding import generate_chaotic_population, repair_mask

logger = logging.getLogger(__name__)


def sigmoid_transfer_function(v: np.ndarray) -> np.ndarray:
    """
    Sigmoid S-shaped transfer function: S(v) = 1 / (1 + exp(-v)).
    Clipped to prevent numerical overflow in exp.
    """
    v_clipped = np.clip(v, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-v_clipped))


def binary_whale_optimization_search(
    evaluator: FitnessEvaluator,
    target_band_count: int = DEFAULT_TARGET_BAND_COUNT,
    pop_size: int = DEFAULT_WOA_POP_SIZE,
    max_iter: int = DEFAULT_WOA_MAX_ITER,
    spiral_b: float = DEFAULT_WOA_SPIRAL_B,
    patience: Optional[int] = DEFAULT_PATIENCE,
    initial_population: Optional[np.ndarray] = None,
    seed: int = DEFAULT_SEED
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Executes Binary Whale Optimization Algorithm (BWOA) to find optimal spectral band mask.

    Args:
        evaluator: FitnessEvaluator instance for multi-objective scorecard computation.
        target_band_count: Maximum permissible selected bands (e.g. 4, 6, 5, 8).
        pop_size: Number of whale search agents (default: 20).
        max_iter: Maximum search iterations (default: 50).
        spiral_b: Logarithmic spiral shape constant (default: 1.0).
        patience: Number of iterations without fitness improvement before early stopping.
        initial_population: Optional pre-seeded binary population of shape (pop_size, 13).
        seed: Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, float, List[float]]:
            - best_mask: 1D binary array of shape (13,) with selected bands as 1s.
            - best_fitness: Scalar fitness score of the best mask.
            - history: List of best fitness scores at each iteration (non-decreasing).
    """
    rng = np.random.default_rng(seed)
    target_count = max(1, min(int(target_band_count), TOTAL_BANDS))

    # 1. Population Initialization
    if initial_population is not None:
        population = np.asarray(initial_population, dtype=np.int32).copy()
        if population.shape != (pop_size, TOTAL_BANDS):
            raise ValueError(
                f"Initial population shape must be ({pop_size}, {TOTAL_BANDS}), got {population.shape}"
            )
    else:
        population = generate_chaotic_population(
            num_agents=pop_size,
            target_band_count=target_count,
            evaluator=evaluator,
            seed=seed
        )

    # 2. Initial Fitness Evaluation
    scores = np.array([evaluator.get_score(agent) for agent in population], dtype=np.float64)
    best_idx = int(np.argmax(scores))
    best_mask = population[best_idx].copy()
    best_fitness = float(scores[best_idx])

    history: List[float] = [best_fitness]
    no_improvement_count = 0

    # 3. Main Optimization Loop
    for t in range(max_iter):
        # Linearly decreasing parameter 'a' from 2 to 0
        a = 2.0 - 2.0 * (t / max_iter)

        for i in range(pop_size):
            r1 = rng.uniform(0.0, 1.0, size=TOTAL_BANDS)
            r2 = rng.uniform(0.0, 1.0, size=TOTAL_BANDS)

            A = 2.0 * a * r1 - a  # Vector in [-a, a]
            C = 2.0 * r2          # Vector in [0, 2]

            p = rng.uniform(0.0, 1.0)
            current_x = population[i].astype(np.float64)

            if p < 0.5:
                # Norm or mean of A vector
                a_norm = np.max(np.abs(A))
                if a_norm < 1.0:
                    # Encircling Prey
                    D = np.abs(C * best_mask.astype(np.float64) - current_x)
                    continuous_v = best_mask.astype(np.float64) - A * D
                else:
                    # Exploration: Search for Prey via Random Agent
                    rand_idx = rng.choice(pop_size)
                    x_rand = population[rand_idx].astype(np.float64)
                    D = np.abs(C * x_rand - current_x)
                    continuous_v = x_rand - A * D
            else:
                # Logarithmic Bubble-Net Spiral Attack
                D_prime = np.abs(best_mask.astype(np.float64) - current_x)
                l = rng.uniform(-1.0, 1.0)
                spiral_factor = np.exp(spiral_b * l) * np.cos(2.0 * np.pi * l)
                continuous_v = D_prime * spiral_factor + best_mask.astype(np.float64)

            # Sigmoid Transfer & Probabilistic Binarization
            prob = sigmoid_transfer_function(continuous_v)
            rand_bits = rng.uniform(0.0, 1.0, size=TOTAL_BANDS)
            new_binary = (rand_bits < prob).astype(np.int32)

            # Apply Repair Step
            repaired_binary = repair_mask(new_binary, target_count, rng=rng)
            population[i] = repaired_binary

        # Evaluate Updated Population
        for i in range(pop_size):
            score_i = evaluator.get_score(population[i])
            scores[i] = score_i
            if score_i > best_fitness:
                best_fitness = float(score_i)
                best_mask = population[i].copy()

        history.append(best_fitness)

        # Early Stopping Check
        if len(history) > 1 and (history[-1] - history[-2]) < 1e-6:
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        if patience is not None and no_improvement_count >= patience:
            logger.debug(
                f"Binary WOA converged early at iteration {t + 1}/{max_iter} (patience={patience})."
            )
            break

    # Guarantee monotonic non-decreasing history record
    for idx in range(1, len(history)):
        if history[idx] < history[idx - 1]:
            history[idx] = history[idx - 1]

    # Ensure best mask respects target budget
    best_mask = repair_mask(best_mask, target_count, rng=rng)
    final_fitness = float(evaluator.get_score(best_mask))

    return best_mask, final_fitness, history
