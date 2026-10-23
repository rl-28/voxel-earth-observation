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

    def test_unknown_location_raises_error(self):
        # Verification of Fix 2: Never silently substitute South Mumbai or a preset
        with self.assertRaises(ValueError) as ctx:
            self.parser._geocode_to_bbox("Unknown Remote Location 12345XYZNonExistent")
        self.assertIn("could not be resolved by available geocoders", str(ctx.exception))

    def test_numeric_coordinate_pair_geocoding(self):
        # Direct lat, lon coordinates
        bbox, provider = self.parser._geocode_to_bbox("19.0760, 72.8777")
        self.assertEqual(provider, "Direct Numeric Coordinate Input")
        self.assertEqual(len(bbox), 4)
        # Check center is near 19.0760, 72.8777
        center_lon = (bbox[0] + bbox[2]) / 2.0
        center_lat = (bbox[1] + bbox[3]) / 2.0
        self.assertAlmostEqual(center_lat, 19.0760, places=3)
        self.assertAlmostEqual(center_lon, 72.8777, places=3)
        self.assertAlmostEqual(bbox[2] - bbox[0], 2 * BBOX_OFFSET, places=4)

    def test_reversed_coordinates_geocoding(self):
        # Explicit labels in reverse order
        bbox, provider = self.parser._geocode_to_bbox("lon: 83.8700, lat: 21.5700")
        self.assertEqual(provider, "Direct Numeric Coordinate Input")
        center_lon = (bbox[0] + bbox[2]) / 2.0
        center_lat = (bbox[1] + bbox[3]) / 2.0
        self.assertAlmostEqual(center_lat, 21.5700, places=3)
        self.assertAlmostEqual(center_lon, 83.8700, places=3)

        # Longitude-first where abs(lon) > 90 cannot be latitude
        bbox2, _ = self.parser._geocode_to_bbox("105.2000, 21.5700")
        center_lon2 = (bbox2[0] + bbox2[2]) / 2.0
        center_lat2 = (bbox2[1] + bbox2[3]) / 2.0
        self.assertAlmostEqual(center_lat2, 21.5700, places=3)
        self.assertAlmostEqual(center_lon2, 105.2000, places=3)

    def test_explicit_bounding_box_input(self):
        # Explicit 4-element bounding box
        bbox, provider = self.parser._geocode_to_bbox("[83.8000, 21.5000, 83.9000, 21.6000]")
        self.assertEqual(provider, "Direct Bounding Box Input")
        self.assertEqual(bbox, [83.8000, 21.5000, 83.9000, 21.6000])

    def test_invalid_coordinate_ranges(self):
        # Physical boundary violations must raise ValueError
        with self.assertRaises(ValueError):
            self.parser._geocode_to_bbox("95.0, 200.0")
        with self.assertRaises(ValueError):
            self.parser._geocode_to_bbox("lat: 95.0, lon: 75.0")
        with self.assertRaises(ValueError):
            self.parser._geocode_to_bbox("lat: 21.0, lon: 195.0")

    def test_distinct_inputs_produce_distinct_aois(self):
        # Ensure different geographic inputs produce strictly different AOIs
        bbox_mumbai, _ = self.parser._geocode_to_bbox("18.9500, 72.8200")
        bbox_hirakud, _ = self.parser._geocode_to_bbox("21.5700, 83.8700")
        self.assertNotEqual(bbox_mumbai, bbox_hirakud)

    def test_rule_based_fallback_with_numeric_coordinates(self):
        res = self.parser._rule_based_fallback("Assess waterlogging risk at 21.57, 83.87")
        self.assertEqual(res["task_type"], TASK_WATER)
        self.assertIn("21.57", res["location_query"])
        self.assertIn("83.87", res["location_query"])

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
