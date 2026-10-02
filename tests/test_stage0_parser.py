"""
Unit and Integration Tests for Stage 0 Natural Language Query Parser & Execution Contract.
"""

import unittest
from unittest.mock import MagicMock, patch
import json

from utils.llm_parser import (
    VoxelExecutionContract,
    VoxelStageZeroParser,
    VoxelQueryCompiler,
    DEFAULT_LOCATION_NAME,
    DEFAULT_LAT,
    DEFAULT_LON,
    BBOX_OFFSET,
    TASK_WATER,
    TASK_CROP,
    TASK_URBAN,
    TASK_GENERAL
)


class TestVoxelExecutionContract(unittest.TestCase):
    def test_contract_dataclass_and_serialization(self):
        contract = VoxelExecutionContract(
            prompt="Detect flood in South Mumbai",
            location_name="South Mumbai, India",
            bbox=[72.8075, 18.9375, 72.8325, 18.9625],
            task_type="WATER_MOISTURE",
            target_band_count=4,
            priority_spectrum_domains=["NIR", "SWIR"],
            scorecard_weights={"accuracy": 0.65, "jm_separability": 0.25, "band_penalty": 0.10},
            user_summary="Sensor setup summary."
        )
        self.assertEqual(contract.prompt, "Detect flood in South Mumbai")
        self.assertEqual(contract.location_name, "South Mumbai, India")
        self.assertEqual(len(contract.bbox), 4)
        self.assertEqual(contract.target_band_count, 4)
        self.assertAlmostEqual(sum(contract.scorecard_weights.values()), 1.0)
        
        d = contract.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["task_type"], "WATER_MOISTURE")
        self.assertEqual(d["target_band_count"], 4)


class TestStageZeroParser(unittest.TestCase):
    def setUp(self):
        self.parser = VoxelStageZeroParser()

    def test_alias_equivalence(self):
        self.assertIs(VoxelQueryCompiler, VoxelStageZeroParser)

    def test_default_geocoding_bbox_calculation(self):
        bbox = self.parser._geocode_to_bbox("Unknown Remote Location 12345XYZ")
        expected_bbox = [
            round(DEFAULT_LON - BBOX_OFFSET, 4),
            round(DEFAULT_LAT - BBOX_OFFSET, 4),
            round(DEFAULT_LON + BBOX_OFFSET, 4),
            round(DEFAULT_LAT + BBOX_OFFSET, 4),
        ]
        self.assertEqual(bbox, expected_bbox)
        # Verify ~2.56 km tile dimension (delta lon and delta lat equal 2 * offset)
        self.assertAlmostEqual(bbox[2] - bbox[0], 2 * BBOX_OFFSET, places=4)
        self.assertAlmostEqual(bbox[3] - bbox[1], 2 * BBOX_OFFSET, places=4)

    def test_rule_based_fallback_water_moisture(self):
        res = self.parser._rule_based_fallback("Assess waterlogging and river flood risk in South Mumbai")
        self.assertEqual(res["task_type"], TASK_WATER)
        self.assertEqual(res["target_band_count"], 4)
        self.assertEqual(res["priority_spectrum_domains"], ["NIR", "SWIR"])
        self.assertAlmostEqual(sum(res["scorecard_weights"].values()), 1.0)

    def test_rule_based_fallback_crop_health(self):
        res = self.parser._rule_based_fallback("Monitor agricultural crop health and NDVI vegetation in Punjab")
        self.assertEqual(res["task_type"], TASK_CROP)
        self.assertEqual(res["target_band_count"], 6)
        self.assertEqual(res["priority_spectrum_domains"], ["RED_EDGE", "NIR"])
        self.assertIn("Punjab", res["location_query"])

    def test_rule_based_fallback_urban_infrastructure(self):
        res = self.parser._rule_based_fallback("Analyze concrete building density and urban infrastructure in Tokyo")
        self.assertEqual(res["task_type"], TASK_URBAN)
        self.assertEqual(res["target_band_count"], 5)
        self.assertEqual(res["priority_spectrum_domains"], ["VISIBLE", "SWIR"])

    def test_rule_based_fallback_general_landcover(self):
        res = self.parser._rule_based_fallback("General terrain mapping and land cover classification")
        self.assertEqual(res["task_type"], TASK_GENERAL)
        self.assertEqual(res["target_band_count"], 8)
        self.assertEqual(res["priority_spectrum_domains"], ["VISIBLE", "NIR", "SWIR", "RED_EDGE"])

    def test_mocked_llm_json_generation(self):
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "location_query": "Belgrade, Serbia",
            "task_type": "WATER_MOISTURE",
            "target_band_count": 4,
            "priority_spectrum_domains": ["NIR", "SWIR"],
            "scorecard_weights": {"accuracy": 0.70, "jm_separability": 0.20, "band_penalty": 0.10},
            "user_summary": "Configured for Danube river flood analysis."
        })
        mock_model.generate_content.return_value = mock_response

        parser = VoxelStageZeroParser()
        parser.model = mock_model

        contract = parser.parse("Analyze Danube flooding in Belgrade")
        self.assertEqual(contract.location_name, "Belgrade, Serbia")
        self.assertEqual(contract.task_type, "WATER_MOISTURE")
        self.assertEqual(contract.target_band_count, 4)
        self.assertEqual(contract.user_summary, "Configured for Danube river flood analysis.")

    def test_mocked_llm_with_markdown_backticks(self):
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "```json\n" + json.dumps({
            "location_query": "Berlin, Germany",
            "task_type": "URBAN_INFRASTRUCTURE",
            "target_band_count": 5,
            "priority_spectrum_domains": ["VISIBLE", "SWIR"],
            "scorecard_weights": {"accuracy": 0.65, "jm_separability": 0.25, "band_penalty": 0.10},
            "user_summary": "Configured for urban building density analysis."
        }) + "\n```"
        mock_model.generate_content.return_value = mock_response

        parser = VoxelStageZeroParser()
        parser.model = mock_model

        contract = parser.parse("Inspect building density in Berlin")
        self.assertEqual(contract.location_name, "Berlin, Germany")
        self.assertEqual(contract.task_type, "URBAN_INFRASTRUCTURE")
        self.assertEqual(contract.target_band_count, 5)


if __name__ == "__main__":
    unittest.main()
