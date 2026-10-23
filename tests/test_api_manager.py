"""
Unit and Integration Tests for Centralized APIManager & Environment Security.
"""

import unittest
import os
from unittest.mock import patch

from utils.api_manager import APIManager, get_api_manager, _is_valid_key, _mask_key
from utils.llm_parser import VoxelStageZeroParser


class TestAPIManager(unittest.TestCase):
    def setUp(self):
        # Reset singleton instance between tests
        APIManager._instance = None
        import utils.api_manager
        utils.api_manager._global_api_manager = None

    def test_singleton_instance(self):
        """Verifies that APIManager follows the singleton pattern."""
        mgr1 = get_api_manager()
        mgr2 = get_api_manager()
        self.assertIs(mgr1, mgr2)

    def test_is_valid_key_filtering(self):
        """Verifies that empty strings and template placeholders are identified as invalid."""
        self.assertFalse(_is_valid_key(""))
        self.assertFalse(_is_valid_key("   "))
        self.assertFalse(_is_valid_key(None))
        self.assertFalse(_is_valid_key("your_gemini_api_key_here"))
        self.assertFalse(_is_valid_key("YOUR_OPENAI_API_KEY_HERE"))
        self.assertFalse(_is_valid_key("none"))
        self.assertFalse(_is_valid_key("null"))
        # Valid key
        self.assertTrue(_is_valid_key("AIzaSyD-valid_real_key_12345"))

    def test_mask_key_security(self):
        """Verifies that secret keys are safely masked for logging & diagnostics."""
        self.assertEqual(_mask_key(None), "Not Set")
        self.assertEqual(_mask_key("your_gemini_api_key_here"), "Not Set")
        self.assertEqual(_mask_key("1234"), "****")
        self.assertEqual(_mask_key("AIzaSyD-valid_real_key_12345"), "AIza...2345")

    @patch.dict(os.environ, {
        "GEMINI_API_KEY": "AIzaSyD-test-gemini-key",
        "MAPBOX_API_KEY": "pk.eyJ1-test-mapbox-token",
        "WANDB_API_KEY": "test-wandb-key"
    }, clear=True)
    def test_key_getters_and_capabilities(self):
        """Verifies key getters and boolean capability checkers."""
        mgr = APIManager(auto_load=False)
        self.assertTrue(mgr.has_gemini())
        self.assertEqual(mgr.get_gemini_key(), "AIzaSyD-test-gemini-key")
        self.assertTrue(mgr.has_mapbox())
        self.assertTrue(mgr.has_wandb())
        self.assertFalse(mgr.has_openai())
        self.assertFalse(mgr.has_groq())
        self.assertEqual(mgr.get_active_llm_provider(), "gemini")

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-proj-test-openai-key"
    }, clear=True)
    def test_llm_provider_fallback_order(self):
        """Verifies fallback to OpenAI when Gemini is not configured."""
        mgr = APIManager(auto_load=False)
        self.assertFalse(mgr.has_gemini())
        self.assertTrue(mgr.has_openai())
        self.assertEqual(mgr.get_active_llm_provider(), "openai")

    @patch.dict(os.environ, {}, clear=True)
    def test_offline_fallback_provider(self):
        """Verifies fallback to rule-based engine when all API keys are absent."""
        mgr = APIManager(auto_load=False)
        self.assertFalse(mgr.has_llm_provider())
        self.assertEqual(mgr.get_active_llm_provider(), "rule_based_fallback")

    def test_health_check_structure(self):
        """Verifies comprehensive masked diagnostic output structure."""
        mgr = get_api_manager()
        report = mgr.health_check()

        self.assertIn("environment", report)
        self.assertIn("active_llm_provider", report)
        self.assertIn("stage_0_llm", report)
        self.assertIn("stage_1_ingestion", report)
        self.assertIn("visualization", report)
        self.assertIn("mlops_tracking", report)
        self.assertTrue(report["stage_0_llm"]["offline_fallback_ready"])
        self.assertTrue(report["stage_1_ingestion"]["nominatim_geocoding_ready"])

    @patch("utils.llm_parser._geocode_location_live", return_value=(18.95, 72.82, None, "Mock Nominatim"))
    @patch.dict(os.environ, {}, clear=True)
    def test_parser_integration_without_api_keys(self, mock_geocode):
        """Verifies Stage 0 query compilation works seamlessly in offline mode."""
        mgr = APIManager(auto_load=False)
        parser = VoxelStageZeroParser(api_manager=mgr)
        contract = parser.parse("Predict flood risk in South Mumbai")

        self.assertIn("South Mumbai", contract.location_name)
        self.assertEqual(contract.task_type, "WATER_MOISTURE")
        self.assertEqual(contract.target_band_count, 4)
        self.assertEqual(len(contract.bbox), 4)


if __name__ == "__main__":
    unittest.main()
