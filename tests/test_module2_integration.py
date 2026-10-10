"""
Integration Tests: Module 1 -> Stage 0 Contract -> Module 2 Pipeline Handoff.
"""

import unittest
import numpy as np

from utils.llm_parser import VoxelExecutionContract
from module2_optimization.integration import apply_module2_band_selection
from module2_optimization.config import TOTAL_BANDS


class TestModule2Integration(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 3D test tile: (256, 256, 13)
        self.tensor_3d = np.random.randn(256, 256, 13).astype(np.float32)
        self.tensor_3d[128:, :, 1] += 3.0  # Blue
        self.tensor_3d[:, 128:, 7] += 4.0  # NIR

        # 4D test batch: (2, 256, 256, 13)
        self.tensor_4d = np.random.randn(2, 256, 256, 13).astype(np.float32)

        # Stage 0 Contracts
        self.contract_water = VoxelExecutionContract(
            prompt="Flood risk in South Mumbai",
            location_name="South Mumbai, India",
            bbox=[72.8075, 18.9375, 72.8325, 18.9625],
            task_type="WATER_MOISTURE",
            target_band_count=4,
            priority_spectrum_domains=["NIR", "SWIR"],
            scorecard_weights={"accuracy": 0.65, "jm_separability": 0.25, "band_penalty": 0.10},
            user_summary="Sensor configured for water detection."
        )

        self.contract_crop = VoxelExecutionContract(
            prompt="Crop health in Punjab",
            location_name="Punjab, India",
            bbox=[75.8, 30.9, 76.0, 31.1],
            task_type="CROP_HEALTH",
            target_band_count=6,
            priority_spectrum_domains=["RED_EDGE", "NIR"],
            scorecard_weights={"accuracy": 0.70, "jm_separability": 0.20, "band_penalty": 0.10},
            user_summary="Sensor configured for vegetation index."
        )

    def test_pipeline_integration_3d_water(self):
        """Tests end-to-end integration: 3D (256, 256, 13) -> Contract (4 bands) -> (256, 256, 4)."""
        reduced_tensor, selected_indices, metrics = apply_module2_band_selection(
            tensor_13_bands=self.tensor_3d,
            contract=self.contract_water,
            pop_size=6,
            max_iter=8,
            seed=42
        )

        self.assertLessEqual(reduced_tensor.shape[2], 4)
        self.assertEqual(reduced_tensor.shape[0], 256)
        self.assertEqual(reduced_tensor.shape[1], 256)
        self.assertEqual(reduced_tensor.shape[2], len(selected_indices))
        self.assertEqual(metrics["output_shape"], reduced_tensor.shape)
        self.assertTrue(metrics["module2_enabled"])
        self.assertGreater(metrics["data_reduction_pct"], 60.0)

    def test_pipeline_integration_4d_crop(self):
        """Tests end-to-end integration: 4D (2, 256, 256, 13) -> Contract (6 bands) -> (2, 256, 256, 6)."""
        reduced_batch, selected_indices, metrics = apply_module2_band_selection(
            tensor_13_bands=self.tensor_4d,
            contract=self.contract_crop,
            pop_size=6,
            max_iter=8,
            seed=42
        )

        self.assertEqual(reduced_batch.shape[0], 2)
        self.assertEqual(reduced_batch.shape[1], 256)
        self.assertEqual(reduced_batch.shape[2], 256)
        self.assertLessEqual(reduced_batch.shape[3], 6)
        self.assertEqual(reduced_batch.shape[3], len(selected_indices))

    def test_feature_flag_bypass(self):
        """Verifies that setting enable_module2=False bypasses reduction and returns full 13-band tensor."""
        output_tensor, selected_indices, metrics = apply_module2_band_selection(
            tensor_13_bands=self.tensor_3d,
            contract=self.contract_water,
            enable_module2=False
        )

        self.assertEqual(output_tensor.shape, (256, 256, 13))
        self.assertEqual(len(selected_indices), TOTAL_BANDS)
        self.assertFalse(metrics["module2_enabled"])
        self.assertEqual(metrics["data_reduction_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
