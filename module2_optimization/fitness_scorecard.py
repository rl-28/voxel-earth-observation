"""
module2_optimization/fitness_scorecard.py

Green-AI multi-objective fitness scorecard for spectral band selection:
Fitness(X) = w1·Accuracy + w2·JM_norm − w3·(‖X‖₁/13) − w4·RAM_penalty

Features:
- Fast k-NN / surrogate classification accuracy
- Regularized Jeffries-Matusita (JM) statistical separability distance
- Steep penalty for exceeding target band budget and empty masks
- Hard step penalty when hardware RAM ceiling (16.0 GB) is breached
- Pseudo-label generation via K-Means when ground truth is omitted
- Exact evaluation caching keyed by binary mask
"""

import logging
from typing import Dict, Tuple, Optional, Any, List
import numpy as np

from .config import (
    TOTAL_BANDS,
    DEFAULT_SCORECARD_WEIGHTS,
    RAM_CEILING_GB,
    HARD_RAM_PENALTY,
    DEFAULT_TARGET_BAND_COUNT,
    DEFAULT_PSEUDO_LABEL_CLUSTERS,
    DEFAULT_SUBSAMPLE_PIXELS,
    DEFAULT_SEED,
    EPSILON_COV,
    EMPTY_MASK_PENALTY,
    OVER_BUDGET_PENALTY_WEIGHT
)
from .physics_partitioning import validate_input_tensor

logger = logging.getLogger(__name__)


def generate_pseudo_labels(
    pixel_features: np.ndarray,
    n_clusters: int = DEFAULT_PSEUDO_LABEL_CLUSTERS,
    seed: int = DEFAULT_SEED,
    max_iter: int = 100
) -> np.ndarray:
    """
    Generates pseudo-ground truth land-cover labels using vectorised K-Means clustering.

    Args:
        pixel_features: 2D array of shape (N_pixels, 13)
        n_clusters: Number of pseudo-classes (default: 6)
        seed: Random seed for deterministic centroid initialization
        max_iter: Maximum EM iterations

    Returns:
        np.ndarray: Integer class labels of shape (N_pixels,) in range [0, n_clusters-1]
    """
    logger.warning(
        f"No ground truth labels provided; generated pseudo-labels using K-Means (k={n_clusters}) surrogate."
    )
    rng = np.random.default_rng(seed)
    n_samples = len(pixel_features)

    # Handle edge case where samples < clusters
    if n_samples < n_clusters:
        return np.zeros(n_samples, dtype=np.int32)

    # Initialize centroids via Random Partition / Forgy
    init_indices = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = pixel_features[init_indices].copy()

    labels = np.zeros(n_samples, dtype=np.int32)
    for _ in range(max_iter):
        # Compute squared Euclidean distances: (N_samples, k)
        # ||x - c||^2 = ||x||^2 - 2 x·c + ||c||^2
        dist_sq = (
            np.sum(pixel_features ** 2, axis=1, keepdims=True)
            - 2 * np.dot(pixel_features, centroids.T)
            + np.sum(centroids ** 2, axis=1, keepdims=True).T
        )
        new_labels = np.argmin(dist_sq, axis=1).astype(np.int32)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update centroids
        for k in range(n_clusters):
            mask_k = (labels == k)
            if np.any(mask_k):
                centroids[k] = pixel_features[mask_k].mean(axis=0)

    return labels


def compute_jm_distance(
    features: np.ndarray,
    labels: np.ndarray,
    epsilon: float = EPSILON_COV
) -> Tuple[float, float]:
    """
    Computes regularized Jeffries-Matusita (JM) distance across all class pairs.

    JM_ij = 2(1 - e^(-B_ij))
    B_ij = 1/8 (mu_i - mu_j)^T Sigma^(-1) (mu_i - mu_j) + 1/2 ln(det(Sigma) / sqrt(det(Sigma_i) det(Sigma_j)))

    Args:
        features: Subsampled feature matrix of shape (N_samples, n_selected_bands)
        labels: Class labels of shape (N_samples,)
        epsilon: Diagonal regularization coefficient for covariance stability

    Returns:
        Tuple[float, float]: (mean_jm_distance in [0, 2], normalized_jm in [0, 1])
    """
    classes = np.unique(labels)
    n_classes = len(classes)
    n_features = features.shape[1]

    if n_classes < 2 or n_features == 0:
        return 0.0, 0.0

    class_stats = {}
    eye = np.eye(n_features, dtype=np.float64)

    for c in classes:
        x_c = features[labels == c].astype(np.float64)
        if len(x_c) < 2:
            continue
        mu = np.mean(x_c, axis=0)
        cov = np.cov(x_c, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        cov_reg = cov + epsilon * eye
        class_stats[c] = (mu, cov_reg)

    valid_classes = list(class_stats.keys())
    if len(valid_classes) < 2:
        return 0.0, 0.0

    jm_list = []
    for i in range(len(valid_classes)):
        c1 = valid_classes[i]
        mu1, cov1 = class_stats[c1]
        for j in range(i + 1, len(valid_classes)):
            c2 = valid_classes[j]
            mu2, cov2 = class_stats[c2]

            cov_avg = 0.5 * (cov1 + cov2)
            diff_mu = mu1 - mu2

            # Matrix inversion with pinv safeguard
            try:
                inv_cov_avg = np.linalg.inv(cov_avg)
            except np.linalg.LinAlgError:
                inv_cov_avg = np.linalg.pinv(cov_avg)

            # Term 1: Mahalanobis component
            t1 = 0.125 * float(np.dot(diff_mu.T, np.dot(inv_cov_avg, diff_mu)))

            # Term 2: Determinant ratio with logdet safeguard
            sign_avg, logdet_avg = np.linalg.slogdet(cov_avg)
            sign1, logdet1 = np.linalg.slogdet(cov1)
            sign2, logdet2 = np.linalg.slogdet(cov2)

            if sign_avg > 0 and sign1 > 0 and sign2 > 0:
                t2 = 0.5 * (logdet_avg - 0.5 * (logdet1 + logdet2))
            else:
                t2 = 0.0

            bhattacharyya = max(0.0, t1 + t2)
            jm_ij = 2.0 * (1.0 - np.exp(-bhattacharyya))
            # Strict boundary clamping to [0, 2]
            jm_ij = min(2.0, max(0.0, jm_ij))
            jm_list.append(jm_ij)

    if not jm_list:
        return 0.0, 0.0

    mean_jm = float(np.mean(jm_list))
    normalized_jm = float(mean_jm / 2.0)
    return mean_jm, normalized_jm


def fast_knn_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    k_neighbors: int = 3,
    train_fraction: float = 0.7,
    seed: int = DEFAULT_SEED
) -> float:
    """
    Computes fast classification accuracy using k-Nearest Neighbors on train-test split.

    Args:
        features: Feature array of shape (N_samples, n_selected_bands)
        labels: Class labels of shape (N_samples,)
        k_neighbors: Number of nearest neighbors
        train_fraction: Fraction of data for training
        seed: Random seed for deterministic train/test partition

    Returns:
        float: Accuracy score in range [0.0, 1.0]
    """
    n_samples = len(features)
    if n_samples == 0 or features.shape[1] == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_train = max(1, int(n_samples * train_fraction))

    train_idx, test_idx = indices[:n_train], indices[n_train:]
    if len(test_idx) == 0:
        test_idx = train_idx

    X_train, y_train = features[train_idx], labels[train_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    # Vectorized pairwise squared Euclidean distance: (N_test, N_train)
    # ||test - train||^2 = ||test||^2 - 2 test·train^T + ||train||^2
    dists = (
        np.sum(X_test ** 2, axis=1, keepdims=True)
        - 2 * np.dot(X_test, X_train.T)
        + np.sum(X_train ** 2, axis=1, keepdims=True).T
    )

    k_eff = min(k_neighbors, len(X_train))
    top_k_indices = np.argpartition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
    top_k_labels = y_train[top_k_indices]  # (N_test, k_eff)

    # Majority vote across top k
    predictions = []
    for row in top_k_labels:
        vals, counts = np.unique(row, return_counts=True)
        predictions.append(vals[np.argmax(counts)])

    predictions = np.array(predictions)
    accuracy = float(np.mean(predictions == y_test))
    return accuracy


class FitnessEvaluator:
    """
    Multi-objective Green-AI fitness scorecard evaluator with memoization.
    """

    def __init__(
        self,
        tensor: np.ndarray,
        target_band_count: int = DEFAULT_TARGET_BAND_COUNT,
        weights: Optional[Dict[str, float]] = None,
        labels: Optional[np.ndarray] = None,
        seed: int = DEFAULT_SEED,
        n_subsample: int = DEFAULT_SUBSAMPLE_PIXELS
    ):
        self.tensor = validate_input_tensor(tensor)
        self.target_band_count = max(1, min(int(target_band_count), TOTAL_BANDS))
        self.weights = weights or DEFAULT_SCORECARD_WEIGHTS.copy()
        self.seed = seed
        self.n_subsample = n_subsample

        # Normalization of weights
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.w_acc = self.weights.get("accuracy", 0.65) / total_w
            self.w_jm = self.weights.get("jm_separability", 0.25) / total_w
            self.w_band = self.weights.get("band_penalty", 0.10) / total_w
        else:
            self.w_acc, self.w_jm, self.w_band = 0.65, 0.25, 0.10

        # Memory footprint estimation
        self.estimated_ram_gb = self._estimate_ram_footprint(self.tensor)
        self.hard_ram_penalty = (
            HARD_RAM_PENALTY if self.estimated_ram_gb > RAM_CEILING_GB else 0.0
        )

        # Prepare 2D pixel samples and labels
        self._prepare_dataset(labels)

        # Cache evaluations keyed by byte string of binary mask
        self._cache: Dict[bytes, Tuple[float, Dict[str, Any]]] = {}

    def _estimate_ram_footprint(self, tensor: np.ndarray) -> float:
        """Estimates peak memory consumption in gigabytes."""
        bytes_per_elem = 4  # float32
        tensor_bytes = tensor.size * bytes_per_elem
        # Optimization overhead factor (population, distance matrices, surrogate models)
        overhead_factor = 2.5
        total_estimated_gb = (tensor_bytes * overhead_factor) / (1024 ** 3)
        return float(total_estimated_gb)

    def _prepare_dataset(self, labels: Optional[np.ndarray]) -> None:
        """Flattens spatial dimensions and extracts deterministic subsamples."""
        # Reshape (..., 256, 256, 13) to (N_pixels, 13)
        flat_features = self.tensor.reshape(-1, TOTAL_BANDS)
        n_total_pixels = len(flat_features)

        rng = np.random.default_rng(self.seed)
        subsample_size = min(self.n_subsample, n_total_pixels)
        sample_indices = rng.choice(n_total_pixels, size=subsample_size, replace=False)

        self.sampled_features = flat_features[sample_indices]

        if labels is not None:
            flat_labels = np.asarray(labels).reshape(-1)
            if len(flat_labels) != n_total_pixels:
                logger.warning(
                    f"Labels size ({len(flat_labels)}) does not match pixel count ({n_total_pixels}). Generating pseudo-labels."
                )
                self.sampled_labels = generate_pseudo_labels(
                    self.sampled_features, seed=self.seed
                )
            else:
                self.sampled_labels = flat_labels[sample_indices]
        else:
            self.sampled_labels = generate_pseudo_labels(
                self.sampled_features, seed=self.seed
            )

    def evaluate(self, mask: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluates the multi-objective fitness score of a binary candidate mask.

        Args:
            mask: 1D binary array of length 13 (values 0 or 1).

        Returns:
            Tuple[float, Dict[str, Any]]: (fitness_score, detailed_metrics_dict)
        """
        mask_arr = np.asarray(mask, dtype=np.int32).ravel()
        if len(mask_arr) != TOTAL_BANDS:
            raise ValueError(f"Mask must have length {TOTAL_BANDS}, got {len(mask_arr)}")

        mask_key = mask_arr.tobytes()
        if mask_key in self._cache:
            return self._cache[mask_key]

        selected_indices = np.where(mask_arr == 1)[0]
        k_selected = len(selected_indices)

        # 1. Empty mask penalty
        if k_selected == 0:
            metrics = {
                "accuracy": 0.0,
                "jm_distance": 0.0,
                "jm_normalized": 0.0,
                "band_count": 0,
                "band_penalty": float(EMPTY_MASK_PENALTY),
                "ram_penalty": self.hard_ram_penalty,
                "fitness": -float(EMPTY_MASK_PENALTY) - self.hard_ram_penalty
            }
            res = (metrics["fitness"], metrics)
            self._cache[mask_key] = res
            return res

        # Sliced features
        sub_features = self.sampled_features[:, selected_indices]

        # 2. Surrogate Accuracy
        acc = fast_knn_accuracy(sub_features, self.sampled_labels, seed=self.seed)

        # 3. Jeffries-Matusita Distance
        jm_dist, jm_norm = compute_jm_distance(sub_features, self.sampled_labels)

        # 4. Band Penalties
        linear_band_penalty = self.w_band * (k_selected / TOTAL_BANDS)
        over_budget_excess = max(0, k_selected - self.target_band_count)
        quadratic_over_budget_penalty = OVER_BUDGET_PENALTY_WEIGHT * (
            (over_budget_excess / TOTAL_BANDS) ** 2
        )
        total_band_penalty = linear_band_penalty + quadratic_over_budget_penalty

        # 5. Composite Fitness Function
        fitness = (
            (self.w_acc * acc)
            + (self.w_jm * jm_norm)
            - total_band_penalty
            - self.hard_ram_penalty
        )

        metrics = {
            "accuracy": float(acc),
            "jm_distance": float(jm_dist),
            "jm_normalized": float(jm_norm),
            "band_count": int(k_selected),
            "linear_band_penalty": float(linear_band_penalty),
            "over_budget_penalty": float(quadratic_over_budget_penalty),
            "total_band_penalty": float(total_band_penalty),
            "ram_penalty": float(self.hard_ram_penalty),
            "fitness": float(fitness)
        }

        res = (float(fitness), metrics)
        self._cache[mask_key] = res
        return res

    def get_score(self, mask: np.ndarray) -> float:
        """Helper method returning scalar fitness score."""
        return self.evaluate(mask)[0]
