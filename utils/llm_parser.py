"""
utils/llm_parser.py - Voxel Stage 0 Query Parser & Natural Language Geospatial Compiler.

Translates freeform human text into a structured VoxelExecutionContract for:
- Stage 1 (STAC Ingestion BBOX)
- Stage 2 (Whale Optimization Algorithm Spectral Band Budget & Scorecard Weights)
- Stage 3 (Neural Fusion Task Profile & Inference)
"""

import os
import json
import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

from .api_manager import get_api_manager

# Optional third-party imports with fallback handling
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

try:
    from geopy.geocoders import Nominatim
    HAS_GEOPY = True
except ImportError:
    Nominatim = None
    HAS_GEOPY = False


# Default geospatial anchor: South Mumbai, India
DEFAULT_LOCATION_NAME = "South Mumbai, India"
DEFAULT_LAT = 18.95
DEFAULT_LON = 72.82
BBOX_OFFSET = 0.0125  # ~1.28 km radius offset matching 256x256 pixel tile footprint at 10m/pixel

# Task Profile Constants
TASK_WATER = "WATER_MOISTURE"
TASK_CROP = "CROP_HEALTH"
TASK_URBAN = "URBAN_INFRASTRUCTURE"
TASK_GENERAL = "GENERAL_LANDCOVER"

TASK_BAND_BUDGETS = {
    TASK_WATER: 4,
    TASK_CROP: 6,
    TASK_URBAN: 5,
    TASK_GENERAL: 8,
}

TASK_SPECTRUM_DOMAINS = {
    TASK_WATER: ["NIR", "SWIR"],
    TASK_CROP: ["RED_EDGE", "NIR"],
    TASK_URBAN: ["VISIBLE", "SWIR"],
    TASK_GENERAL: ["VISIBLE", "NIR", "SWIR", "RED_EDGE"],
}

DEFAULT_SCORECARD_WEIGHTS = {
    "accuracy": 0.65,
    "jm_separability": 0.25,
    "band_penalty": 0.10,
}


@dataclass
class VoxelExecutionContract:
    """
    Stage 0 Execution Contract configuring downstream Voxel pipeline stages.
    """
    prompt: str
    location_name: str
    bbox: List[float]                    # [min_lon, min_lat, max_lon, max_lat] -> For Module 1 STAC
    task_type: str                       # Task Profile -> For Module 3 & Dashboard UI
    target_band_count: int               # Band Budget -> For Module 2 Whale Optimization
    priority_spectrum_domains: List[str] # Physical Domains -> For Spectrum Partitioning
    scorecard_weights: Dict[str, float]  # Weights -> For Green AI Scorecard
    user_summary: str                    # Friendly summary for UI display

    def to_dict(self) -> Dict[str, Any]:
        """Serializes contract to a standard Python dictionary."""
        return {
            "prompt": self.prompt,
            "location_name": self.location_name,
            "bbox": self.bbox,
            "task_type": self.task_type,
            "target_band_count": self.target_band_count,
            "priority_spectrum_domains": self.priority_spectrum_domains,
            "scorecard_weights": self.scorecard_weights,
            "user_summary": self.user_summary,
        }


class VoxelStageZeroParser:
    """
    Stage 0 Natural Language Query Compiler for the Voxel Earth Observation platform.
    Uses centralized APIManager for credentials, with Google Gemini, OpenAI, Groq,
    and deterministic offline rule-based fallbacks.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        api_manager: Optional[Any] = None
    ):
        self.api_mgr = api_manager or get_api_manager()
        self.api_key = api_key or self.api_mgr.get_gemini_key()
        self.model_name = model_name
        self.model = None

        if HAS_GENAI and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini model: {e}")
                self.model = None

        if HAS_GEOPY and Nominatim is not None:
            try:
                self.geolocator = Nominatim(user_agent="voxel_stage0_parser")
            except Exception as e:
                logger.warning(f"Failed to initialize Nominatim geolocator: {e}")
                self.geolocator = None
        else:
            self.geolocator = None

    def _geocode_to_bbox(self, location_query: str) -> List[float]:
        """
        Converts place name into a ~2.56 km x 2.56 km geospatial BBOX:
        [min_lon, min_lat, max_lon, max_lat].
        Falls back to South Mumbai [72.8075, 18.9375, 72.8325, 18.9625] on error or timeout.
        """
        lat, lon = DEFAULT_LAT, DEFAULT_LON

        if self.geolocator and location_query and location_query.strip():
            try:
                loc = self.geolocator.geocode(location_query, timeout=5)
                if loc:
                    lat, lon = loc.latitude, loc.longitude
            except Exception as e:
                logger.debug(f"Geocoding exception for '{location_query}', falling back to default: {e}")
                lat, lon = DEFAULT_LAT, DEFAULT_LON

        offset = BBOX_OFFSET
        return [
            round(lon - offset, 4),  # min_lon
            round(lat - offset, 4),  # min_lat
            round(lon + offset, 4),  # max_lon
            round(lat + offset, 4)   # max_lat
        ]

    def _rule_based_fallback(self, user_prompt: str) -> Dict[str, Any]:
        """
        Deterministic, offline rule-based semantic parser when LLM is unavailable or fails.
        """
        prompt_lower = user_prompt.lower()

        def has_any_word(words: List[str]) -> bool:
            pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")\b"
            return bool(re.search(pattern, prompt_lower))

        # 1. Classify Task Type with exact word boundaries
        if has_any_word(["rain", "rainfall", "flood", "flooding", "water", "waterlogging", "moisture", "drainage", "river", "lake", "soil moisture"]):
            task_type = TASK_WATER
        elif has_any_word(["crop", "crops", "agriculture", "agricultural", "vegetation", "forest", "deforestation", "farm", "farming", "plant", "ndvi"]):
            task_type = TASK_CROP
        elif has_any_word(["urban", "building", "buildings", "infrastructure", "concrete", "road", "roads", "settlement", "heat island"]):
            task_type = TASK_URBAN
        else:
            task_type = TASK_GENERAL

        # 2. Extract Location heuristics
        location_query = DEFAULT_LOCATION_NAME
        loc_patterns = [
            r"\b(?:in|around|near|across|over|for)\s+([A-Z][a-zA-Z0-9\s,\-\.]+?)(?:\s+(?:for|to|with|using|and|\.|\?|$)|$)",
            r"\b(?:in|around|near|across|over|for)\s+([a-zA-Z\s,\-\.]+?)(?:\s+(?:for|to|with|using|and|\.|\?|$)|$)"
        ]
        for pat in loc_patterns:
            match = re.search(pat, user_prompt)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 2 and extracted.lower() not in ["the", "a", "an", "all", "our", "general", "each"]:
                    location_query = extracted
                    break

        target_band_count = TASK_BAND_BUDGETS[task_type]
        priority_spectrum_domains = TASK_SPECTRUM_DOMAINS[task_type]
        scorecard_weights = DEFAULT_SCORECARD_WEIGHTS.copy()

        user_summary = (
            f"Configured Voxel satellite pipeline for {task_type.replace('_', ' ').lower()} "
            f"analysis across {location_query} using a {target_band_count}-band optimization profile."
        )

        return {
            "location_query": location_query,
            "task_type": task_type,
            "target_band_count": target_band_count,
            "priority_spectrum_domains": priority_spectrum_domains,
            "scorecard_weights": scorecard_weights,
            "user_summary": user_summary,
        }

    def parse(self, user_prompt: str) -> VoxelExecutionContract:
        """
        Sends prompt to LLM and returns structured Execution Contract.
        Falls back to rule-based parser if API key is not configured or generation fails.
        """
        raw = None

        if self.model is not None:
            system_prompt = f"""You are the Stage 0 Natural Language Query Compiler for "Voxel", an Earth Observation platform using Sentinel-2 Level-2A satellite imagery, multi-objective Whale Optimization Algorithm (WOA) band selection, and hybrid CNN-ViT neural fusion.

USER PROMPT:
"{user_prompt}"

COMPILATION RULES:
1. "location_query": Extract city/region. If unspecified, default to "South Mumbai, India".
2. "task_type": Categorize into ONE of: "WATER_MOISTURE", "CROP_HEALTH", "URBAN_INFRASTRUCTURE", "GENERAL_LANDCOVER".
3. "target_band_count": Integer budget for WOA (4 for water/flood, 6 for agriculture, 5 for urban, 8 for general).
4. "priority_spectrum_domains": Array of light domains (e.g. ["NIR", "SWIR"]).
5. "scorecard_weights": Object with "accuracy", "jm_separability", "band_penalty" summing to 1.0.
6. "user_summary": Single concise sentence explaining sensor setup.

Return ONLY a valid raw JSON object matching this schema with NO markdown backticks:
{{
  "location_query": "string",
  "task_type": "WATER_MOISTURE" | "CROP_HEALTH" | "URBAN_INFRASTRUCTURE" | "GENERAL_LANDCOVER",
  "target_band_count": int,
  "priority_spectrum_domains": ["string"],
  "scorecard_weights": {{"accuracy": float, "jm_separability": float, "band_penalty": float}},
  "user_summary": "string"
}}"""
            try:
                response = self.model.generate_content(
                    system_prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                text = response.text.strip()
                # Clean possible markdown backticks if returned
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\n?", "", text)
                    text = re.sub(r"\n?```$", "", text)
                raw = json.loads(text)
            except Exception as e:
                logger.info(f"LLM generation failed or unavailable ({e}), using fallback parser.")
                raw = None

        if raw is None or not isinstance(raw, dict):
            raw = self._rule_based_fallback(user_prompt)

        # Sanitize and validate fields
        location_query = raw.get("location_query") or DEFAULT_LOCATION_NAME
        task_type = raw.get("task_type")
        if task_type not in TASK_BAND_BUDGETS:
            task_type = TASK_GENERAL

        target_band_count = raw.get("target_band_count", TASK_BAND_BUDGETS[task_type])
        priority_domains = raw.get("priority_spectrum_domains", TASK_SPECTRUM_DOMAINS[task_type])
        scorecard_weights = raw.get("scorecard_weights", DEFAULT_SCORECARD_WEIGHTS)
        user_summary = raw.get("user_summary") or (
            f"Configured Voxel satellite pipeline for {task_type.lower()} analysis across {location_query}."
        )

        # Convert place string to exact BBOX coordinates
        bbox = self._geocode_to_bbox(location_query)

        return VoxelExecutionContract(
            prompt=user_prompt,
            location_name=location_query,
            bbox=bbox,
            task_type=task_type,
            target_band_count=int(target_band_count),
            priority_spectrum_domains=priority_domains,
            scorecard_weights=scorecard_weights,
            user_summary=user_summary
        )


# Alias for unified nomenclature
VoxelQueryCompiler = VoxelStageZeroParser


# Direct script execution test
if __name__ == "__main__":
    parser = VoxelStageZeroParser()
    contract = parser.parse("Predict where highest rainfall accumulation will occur in South Mumbai")
    print("\n=== STAGE 0 EXECUTION CONTRACT ===")
    print(f"Location    : {contract.location_name}")
    print(f"BBOX        : {contract.bbox}")
    print(f"Task Type   : {contract.task_type}")
    print(f"Target Bands: {contract.target_band_count}")
    print(f"Summary     : {contract.user_summary}")
