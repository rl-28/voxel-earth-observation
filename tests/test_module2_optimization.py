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
    SPECTRUM_DOMAINS
)
from module2_optimization.physics_partitioning import (
    validate_input_tensor,
    partition_spectrum
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
        # Non-numpy input
        with self.assertRaises(TypeError):
            validate_input_tensor([[1, 2, 3]])

        # Invalid dimensions (2D)
        with self.assertRaises(ValueError):
            validate_input_tensor(np.zeros((256, 256), dtype=np.float32))

        # Invalid spatial resolution
        with self.assertRaises(ValueError):
            validate_input_tensor(np.zeros((128, 128, 13), dtype=np.float32))

        # Invalid channel count (e.g. 12 instead of 13)
        with self.assertRaises(ValueError):
            validate_input_tensor(np.zeros((256, 256, 12), dtype=np.float32))

        # Non-finite values
        inf_tensor = self.tensor_3d.copy()
        inf_tensor[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_input_tensor(inf_tensor)


if __name__ == "__main__":
    unittest.main()
