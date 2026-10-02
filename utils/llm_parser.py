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
import urllib.request
import urllib.parse
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple

try:
    from .api_manager import get_api_manager
except ImportError:
    from utils.api_manager import get_api_manager

logger = logging.getLogger(__name__)

# Optional third-party imports
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
    llm_provider: str = "gemini"         # Active LLM provider used
    geocoder: str = "nominatim"          # Geocoding provider used

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
            "llm_provider": self.llm_provider,
            "geocoder": self.geocoder,
        }


def _call_gemini_rest_api(api_key: str, system_prompt: str, model_name: str = "gemini-1.5-flash") -> Optional[Dict[str, Any]]:
    """
    Direct REST API client for Google Gemini without requiring third-party SDK dependencies.
    """
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [{"text": system_prompt}]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            result = json.loads(response.read().decode("utf-8"))
            candidates = result.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    raw_text = parts[0].get("text", "").strip()
                    # Strip any possible markdown backticks
                    if raw_text.startswith("```"):
                        raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
                        raw_text = re.sub(r"\n?```$", "", raw_text)
                    return json.loads(raw_text)
    except Exception as e:
        logger.debug(f"Gemini REST call failed: {e}")
        return None
    return None


def _geocode_location_live(location_query: str, google_maps_key: Optional[str] = None) -> Tuple[float, float, str]:
    """
    Geocodes a location query to (latitude, longitude, provider_name).
    Tries:
    1. Google Maps Geocoding API (if key present)
    2. OpenStreetMap Nominatim live REST HTTP with progressive relaxation
    3. geopy Nominatim (if installed)
    4. Fallback to South Mumbai (18.95, 72.82) on network failure or unfound queries.
    """
    if not location_query or not location_query.strip():
        return DEFAULT_LAT, DEFAULT_LON, "Pre-cached Fallback"

    raw_query = location_query.strip()
    # Generate candidate queries for complex compounds (e.g. "Amazon Basin, Para, Brazil" -> ["Amazon Basin, Para, Brazil", "Para, Brazil", "Amazon Basin", "Brazil"])
    candidates = [raw_query]
    if "," in raw_query:
        parts = [p.strip() for p in raw_query.split(",") if p.strip()]
        if len(parts) >= 2:
            candidates.append(", ".join(parts[1:]))
            candidates.append(parts[0])
            candidates.append(parts[-1])

    # 1. Try Google Maps Geocoding API if key configured
    if google_maps_key and google_maps_key.strip():
        for q in candidates:
            try:
                params = urllib.parse.urlencode({"address": q, "key": google_maps_key.strip()})
                url = f"https://maps.googleapis.com/maps/api/geocode/json?{params}"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("status") == "OK" and len(data.get("results", [])) > 0:
                        loc = data["results"][0]["geometry"]["location"]
                        return float(loc["lat"]), float(loc["lng"]), "Google Maps Geocoding API"
            except Exception as e:
                logger.debug(f"Google Maps geocoding failed for '{q}': {e}")

    # 2. Try direct HTTP Nominatim API (OpenStreetMap)
    for q in candidates:
        try:
            params = urllib.parse.urlencode({"q": q, "format": "json", "limit": 1})
            url = f"https://nominatim.openstreetmap.org/search?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "VoxelEarthObservation/1.0 (voxel@earth.ai)"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data and len(data) > 0:
                    lat = float(data[0]["lat"])
                    lon = float(data[0]["lon"])
                    return lat, lon, "OpenStreetMap Nominatim (Live REST)"
        except Exception as e:
            logger.debug(f"HTTP Nominatim lookup failed for '{q}': {e}")

    # 3. Try geopy if installed
    if HAS_GEOPY and Nominatim is not None:
        try:
            geolocator = Nominatim(user_agent="voxel_stage0_parser")
            for q in candidates:
                loc = geolocator.geocode(q, timeout=5)
                if loc:
                    return float(loc.latitude), float(loc.longitude), "geopy Nominatim"
        except Exception as e:
            logger.debug(f"geopy Nominatim lookup failed: {e}")

    return DEFAULT_LAT, DEFAULT_LON, "Pre-cached Fallback (South Mumbai)"


class VoxelStageZeroParser:
    """
    Stage 0 Natural Language Query Compiler for the Voxel Earth Observation platform.
    Uses centralized APIManager for credentials, with Google Gemini (SDK or REST),
    live OpenStreetMap Nominatim geocoding, and deterministic offline rule-based fallbacks.
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
                logger.warning(f"Failed to initialize Gemini SDK model: {e}")
                self.model = None

    def _geocode_to_bbox(self, location_query: str) -> Tuple[List[float], str]:
        """
        Converts place name into a ~2.56 km x 2.56 km geospatial BBOX:
        [min_lon, min_lat, max_lon, max_lat] and returns geocoder name.
        """
        gmaps_key = self.api_mgr.get_google_maps_key() if self.api_mgr else None
        lat, lon, provider = _geocode_location_live(location_query, google_maps_key=gmaps_key)
        offset = BBOX_OFFSET
        bbox = [
            round(lon - offset, 4),  # min_lon
            round(lat - offset, 4),  # min_lat
            round(lon + offset, 4),  # max_lon
            round(lat + offset, 4)   # max_lat
        ]
        return bbox, provider

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
        Uses Gemini SDK -> Gemini REST -> Rule-based fallback.
        """
        raw = None
        llm_provider_used = "rule_based_fallback"
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

        # 1. Try Gemini SDK if initialized
        if self.model is not None:
            try:
                response = self.model.generate_content(
                    system_prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                text = response.text.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\n?", "", text)
                    text = re.sub(r"\n?```$", "", text)
                raw = json.loads(text)
                llm_provider_used = f"Google Gemini SDK ({self.model_name})"
            except Exception as e:
                logger.debug(f"Gemini SDK generation failed: {e}")
                raw = None

        # 2. Try direct Gemini REST API if SDK not available or failed
        if raw is None and self.api_key:
            raw = _call_gemini_rest_api(self.api_key, system_prompt, self.model_name)
            if raw is not None:
                llm_provider_used = f"Google Gemini REST API ({self.model_name})"

        # 3. Fallback to deterministic rule-based parser
        if raw is None or not isinstance(raw, dict):
            raw = self._rule_based_fallback(user_prompt)
            llm_provider_used = "Deterministic Offline Rule Engine"

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

        # Geocode to real BBOX coordinates
        bbox, geocoder_used = self._geocode_to_bbox(location_query)

        return VoxelExecutionContract(
            prompt=user_prompt,
            location_name=location_query,
            bbox=bbox,
            task_type=task_type,
            target_band_count=int(target_band_count),
            priority_spectrum_domains=priority_domains,
            scorecard_weights=scorecard_weights,
            user_summary=user_summary,
            llm_provider=llm_provider_used,
            geocoder=geocoder_used
        )


# Alias for unified nomenclature
VoxelQueryCompiler = VoxelStageZeroParser


if __name__ == "__main__":
    parser = VoxelStageZeroParser()
    contract = parser.parse("Analyze agricultural crop health and vegetation in Punjab")
    print("\n=== STAGE 0 EXECUTION CONTRACT ===")
    print(f"Location    : {contract.location_name}")
    print(f"BBOX        : {contract.bbox}")
    print(f"Task Type   : {contract.task_type}")
    print(f"Target Bands: {contract.target_band_count}")
    print(f"Summary     : {contract.user_summary}")
