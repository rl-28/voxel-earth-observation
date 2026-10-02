"""
Unit and Integration Tests for Module 2: Metaheuristic Optimization Core.
"""

import unittest
import numpy as np

from module2_optimization.config import (
    TOTAL_BANDS,
    BAND_NAMES,
    BAND_TO_INDEX,
    INDEX_TO_BAND,
    SPECTRUM_DOMAINS,
    HARD_RAM_PENALTY,
    EMPTY_MASK_PENALTY,
    DEFAULT_WOA_POP_SIZE,
    DEFAULT_TARGET_BAND_COUNT
)
from module2_optimization.physics_partitioning import (
    validate_input_tensor,
    partition_spectrum
)
from module2_optimization.fitness_scorecard import (
    FitnessEvaluator,
    compute_jm_distance,
    fast_knn_accuracy,
    generate_pseudo_labels
)
from module2_optimization.chaotic_seeding import (
    chebyshev_map_sequence,
    repair_mask,
    repair_population,
    generate_chaotic_population
)
from module2_optimization.bio_search import (
    sigmoid_transfer_function,
    binary_whale_optimization_search
)


class TestPhysicsPartitioning(unittest.TestCase):
    def setUp(self):
        # Deterministic 3D synthetic tensor: (256, 256, 13)
        np.random.seed(42)
        self.tensor_3d = np.random.randn(256, 256, 13).astype(np.float32)
        # Deterministic 4D synthetic batch: (4, 256, 256, 13)
        self.tensor_4d = np.random.randn(4, 256, 256, 13).astype(np.float32)

    def test_band_configuration_and_indices(self):
        """Validates that 13 Sentinel-2 bands are mapped with correct 0-based indices."""
        self.assertEqual(TOTAL_BANDS, 13)
        self.assertEqual(len(BAND_NAMES), 13)
        self.assertEqual(BAND_NAMES[0], "B01")
        self.assertEqual(BAND_NAMES[1], "B02")
        self.assertEqual(BAND_NAMES[7], "B08")
        self.assertEqual(BAND_NAMES[8], "B8A")
        self.assertEqual(BAND_NAMES[10], "B10")
        self.assertEqual(BAND_NAMES[12], "B12")

        # Bidirectional index mappings
        for i, name in enumerate(BAND_NAMES):
            self.assertEqual(BAND_TO_INDEX[name], i)
            self.assertEqual(INDEX_TO_BAND[i], name)

    def test_domain_groups_coverage(self):
        """Ensures all 13 bands belong to exactly one physics spectrum domain."""
        all_indices = []
        for domain, info in SPECTRUM_DOMAINS.items():
            all_indices.extend(info["indices"])
            for b_name in info["bands"]:
                self.assertIn(b_name, BAND_NAMES)

        self.assertEqual(sorted(all_indices), list(range(13)))
        self.assertEqual(SPECTRUM_DOMAINS["VISIBLE"]["indices"], [1, 2, 3])
        self.assertEqual(SPECTRUM_DOMAINS["NIR_RED_EDGE"]["indices"], [4, 5, 6, 7, 8])
        self.assertEqual(SPECTRUM_DOMAINS["SWIR"]["indices"], [11, 12])
        self.assertEqual(SPECTRUM_DOMAINS["AUXILIARY"]["indices"], [0, 9, 10])

    def test_partition_spectrum_3d(self):
        """Tests spectrum partitioning on a single 3D scene tile (256, 256, 13)."""
        partitions = partition_spectrum(self.tensor_3d)

        self.assertIn("VISIBLE", partitions)
        self.assertIn("NIR_RED_EDGE", partitions)
        self.assertIn("SWIR", partitions)
        self.assertIn("AUXILIARY", partitions)

        # Check shapes
        self.assertEqual(partitions["VISIBLE"]["data"].shape, (256, 256, 3))
        self.assertEqual(partitions["NIR_RED_EDGE"]["data"].shape, (256, 256, 5))
        self.assertEqual(partitions["SWIR"]["data"].shape, (256, 256, 2))
        self.assertEqual(partitions["AUXILIARY"]["data"].shape, (256, 256, 3))

        # Check content fidelity
        np.testing.assert_array_equal(
            partitions["VISIBLE"]["data"],
            self.tensor_3d[..., [1, 2, 3]]
        )
        np.testing.assert_array_equal(
            partitions["SWIR"]["data"],
            self.tensor_3d[..., [11, 12]]
        )

    def test_partition_spectrum_4d(self):
        """Tests spectrum partitioning on a 4D batch (N, 256, 256, 13)."""
        partitions = partition_spectrum(self.tensor_4d)

        self.assertEqual(partitions["VISIBLE"]["data"].shape, (4, 256, 256, 3))
        self.assertEqual(partitions["NIR_RED_EDGE"]["data"].shape, (4, 256, 256, 5))
        self.assertEqual(partitions["SWIR"]["data"].shape, (4, 256, 256, 2))
        self.assertEqual(partitions["AUXILIARY"]["data"].shape, (4, 256, 256, 3))

    def test_validation_errors(self):
        """Verifies that invalid shapes, dtypes, or non-finite values raise proper exceptions."""
        with self.assertRaises(TypeError):
            validate_input_tensor([[1, 2, 3]])

        with self.assertRaises(ValueError):
            validate_input_tensor(np.zeros((256, 256), dtype=np.float32))

        with self.assertRaises(ValueError):
            validate_input_tensor(np.zeros((128, 128, 13), dtype=np.float32))

        with self.assertRaises(ValueError):
            validate_input_tensor(np.zeros((256, 256), dtype=np.float32))

        inf_tensor = self.tensor_3d.copy()
        inf_tensor[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_input_tensor(inf_tensor)


class TestFitnessScorecard(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.tensor = np.random.randn(256, 256, 13).astype(np.float32)
        self.tensor[128:, :, 1] += 3.0  # B02
        self.tensor[:, 128:, 7] += 4.0  # B08

        self.evaluator = FitnessEvaluator(
            tensor=self.tensor,
            target_band_count=4,
            seed=42,
            n_subsample=500
        )

    def test_empty_mask_penalized(self):
        """Verifies that an empty mask (all 0s) receives heavy penalty."""
        empty_mask = np.zeros(13, dtype=np.int32)
        fitness, metrics = self.evaluator.evaluate(empty_mask)
        self.assertLessEqual(fitness, -EMPTY_MASK_PENALTY)
        self.assertEqual(metrics["band_count"], 0)

    def test_over_budget_penalty(self):
        """Validates that masks exceeding target_band_count are penalized."""
        within_budget_mask = np.zeros(13, dtype=np.int32)
        within_budget_mask[[1, 2, 7, 11]] = 1

        over_budget_mask = np.ones(13, dtype=np.int32)

        score_within = self.evaluator.get_score(within_budget_mask)
        score_over = self.evaluator.get_score(over_budget_mask)

        self.assertGreater(score_within, score_over)
        _, over_metrics = self.evaluator.evaluate(over_budget_mask)
        self.assertGreater(over_metrics["over_budget_penalty"], 0.0)

    def test_jm_distance_bounds_and_properties(self):
        """Verifies that JM distance is in [0, 2] and normalized JM in [0, 1]."""
        mask = np.zeros(13, dtype=np.int32)
        mask[[1, 3, 7, 11]] = 1

        _, metrics = self.evaluator.evaluate(mask)
        self.assertGreaterEqual(metrics["jm_distance"], 0.0)
        self.assertLessEqual(metrics["jm_distance"], 2.0)
        self.assertGreaterEqual(metrics["jm_normalized"], 0.0)
        self.assertLessEqual(metrics["jm_normalized"], 1.0)

    def test_determinism_and_caching(self):
        """Verifies deterministic evaluation and caching efficiency."""
        mask = np.zeros(13, dtype=np.int32)
        mask[[1, 4, 7, 11]] = 1

        score1, m1 = self.evaluator.evaluate(mask)
        score2, m2 = self.evaluator.evaluate(mask)

        self.assertEqual(score1, score2)
        self.assertEqual(m1["accuracy"], m2["accuracy"])
        self.assertEqual(m1["jm_distance"], m2["jm_distance"])
        self.assertIn(mask.tobytes(), self.evaluator._cache)

    def test_ram_penalty_trigger(self):
        """Verifies that hard RAM penalty triggers when estimated footprint exceeds RAM_CEILING_GB."""
        evaluator_ram = FitnessEvaluator(
            tensor=self.tensor,
            target_band_count=4,
            seed=42
        )
        evaluator_ram.hard_ram_penalty = HARD_RAM_PENALTY
        mask = np.zeros(13, dtype=np.int32)
        mask[[1, 2, 7]] = 1

        fitness, metrics = evaluator_ram.evaluate(mask)
        self.assertEqual(metrics["ram_penalty"], HARD_RAM_PENALTY)
        self.assertLess(fitness, 0.0)

    def test_with_explicit_ground_truth_labels(self):
        """Verifies evaluation with externally provided labels."""
        explicit_labels = np.random.randint(0, 4, size=(256, 256))
        evaluator_labeled = FitnessEvaluator(
            tensor=self.tensor,
            target_band_count=4,
            labels=explicit_labels,
            seed=42,
            n_subsample=500
        )
        mask = np.zeros(13, dtype=np.int32)
        mask[[1, 2, 7, 11]] = 1
        fitness, metrics = evaluator_labeled.evaluate(mask)
        self.assertIsInstance(fitness, float)
        self.assertGreater(metrics["accuracy"], 0.0)


class TestChaoticSeedingAndOBL(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.tensor = np.random.randn(256, 256, 13).astype(np.float32)
        self.evaluator = FitnessEvaluator(
            tensor=self.tensor,
            target_band_count=5,
            seed=42,
            n_subsample=500
        )

    def test_chebyshev_map_bounds(self):
        """Validates that Chebyshev map outputs bounded values in [-1, 1]."""
        seq = chebyshev_map_sequence(length=100, seed=42)
        self.assertEqual(len(seq), 100)
        self.assertTrue(np.all(seq >= -1.0))
        self.assertTrue(np.all(seq <= 1.0))

    def test_repair_mask_enforcement(self):
        """Verifies that repair_mask enforces 1 <= active_bands <= target_band_count."""
        empty_mask = np.zeros(13, dtype=np.int32)
        repaired_empty = repair_mask(empty_mask, target_band_count=4)
        self.assertGreaterEqual(np.sum(repaired_empty), 1)
        self.assertLessEqual(np.sum(repaired_empty), 4)

        over_mask = np.ones(13, dtype=np.int32)
        repaired_over = repair_mask(over_mask, target_band_count=4)
        self.assertEqual(np.sum(repaired_over), 4)

        valid_mask = np.zeros(13, dtype=np.int32)
        valid_mask[[1, 3, 7]] = 1
        repaired_valid = repair_mask(valid_mask, target_band_count=5)
        self.assertEqual(np.sum(repaired_valid), 3)

    def test_generate_chaotic_population_shape_and_values(self):
        """Validates output shape (num_agents, 13) and binary values {0, 1}."""
        pop = generate_chaotic_population(
            num_agents=15,
            target_band_count=6,
            seed=42
        )
        self.assertEqual(pop.shape, (15, 13))
        self.assertTrue(np.all(np.isin(pop, [0, 1])))
        sums = np.sum(pop, axis=1)
        self.assertTrue(np.all(sums >= 1))
        self.assertTrue(np.all(sums <= 6))

    def test_reproducibility_with_seed(self):
        """Verifies that fixed seed produces identical populations."""
        pop1 = generate_chaotic_population(num_agents=10, target_band_count=4, seed=123)
        pop2 = generate_chaotic_population(num_agents=10, target_band_count=4, seed=123)
        np.testing.assert_array_equal(pop1, pop2)

    def test_obl_population_selection(self):
        """Validates Opposition-Based Learning integration with FitnessEvaluator."""
        pop_obl = generate_chaotic_population(
            num_agents=10,
            target_band_count=5,
            evaluator=self.evaluator,
            seed=42
        )
        self.assertEqual(pop_obl.shape, (10, 13))
        for ind in pop_obl:
            self.assertGreaterEqual(np.sum(ind), 1)
            self.assertLessEqual(np.sum(ind), 5)


class TestBinaryWhaleOptimization(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.tensor = np.random.randn(256, 256, 13).astype(np.float32)
        self.tensor[128:, :, 1] += 3.0  # B02
        self.tensor[:, 128:, 7] += 4.0  # B08
        self.evaluator = FitnessEvaluator(
            tensor=self.tensor,
            target_band_count=4,
            seed=42,
            n_subsample=400
        )

    def test_sigmoid_transfer_function(self):
        """Validates sigmoid transfer function properties and bounds."""
        v = np.array([-100.0, 0.0, 100.0])
        s = sigmoid_transfer_function(v)
        self.assertAlmostEqual(s[1], 0.5)
        self.assertGreaterEqual(s[0], 0.0)
        self.assertLessEqual(s[2], 1.0)

    def test_bwoa_search_contract_and_budget(self):
        """Verifies that BWOA returns 13-bit mask respecting target budget."""
        best_mask, best_fitness, history = binary_whale_optimization_search(
            evaluator=self.evaluator,
            target_band_count=4,
            pop_size=10,
            max_iter=15,
            seed=42
        )
        self.assertEqual(len(best_mask), 13)
        self.assertTrue(np.all(np.isin(best_mask, [0, 1])))
        self.assertGreaterEqual(np.sum(best_mask), 1)
        self.assertLessEqual(np.sum(best_mask), 4)
        self.assertIsInstance(best_fitness, float)

    def test_bwoa_monotonic_history(self):
        """Verifies that best fitness history is non-decreasing across iterations."""
        _, _, history = binary_whale_optimization_search(
            evaluator=self.evaluator,
            target_band_count=4,
            pop_size=10,
            max_iter=15,
            seed=42
        )
        self.assertGreater(len(history), 1)
        for t in range(len(history) - 1):
            self.assertGreaterEqual(history[t + 1], history[t])

    def test_bwoa_result_superiority_or_equality(self):
        """Verifies that optimized mask fitness is >= initial best seeded fitness."""
        # Initial seed best
        initial_pop = generate_chaotic_population(
            num_agents=10,
            target_band_count=4,
            evaluator=self.evaluator,
            seed=42
        )
        initial_best_score = max(self.evaluator.get_score(ind) for ind in initial_pop)

        best_mask, best_fitness, history = binary_whale_optimization_search(
            evaluator=self.evaluator,
            target_band_count=4,
            pop_size=10,
            max_iter=15,
            initial_population=initial_pop,
            seed=42
        )
        self.assertGreaterEqual(best_fitness, initial_best_score)
        self.assertEqual(history[0], initial_best_score)


if __name__ == "__main__":
    unittest.main()
