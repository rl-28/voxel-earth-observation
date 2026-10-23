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
import math
import numpy as np
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


def calculate_geographic_span(bbox: List[float]) -> Tuple[float, float]:
    """
    Computes physical ground span (width_km, height_km) of a WGS84 bounding box [min_lon, min_lat, max_lon, max_lat].
    Uses precise WGS84 ellipsoidal distance formulas.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2.0
    lat_rad = np.radians(mid_lat)
    m_per_deg_lat = 111132.95 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    width_km = (max_lon - min_lon) * m_per_deg_lon / 1000.0
    height_km = (max_lat - min_lat) * m_per_deg_lat / 1000.0
    return float(width_km), float(height_km)


def compute_canonical_tile_bbox(lat: float, lon: float, tile_size_m: float = 2560.0) -> List[float]:
    """
    Computes exact WGS84 bounding box [min_lon, min_lat, max_lon, max_lat] centered at (lat, lon)
    matching tile_size_m x tile_size_m (canonical 2560m = 256 px at 10m Sentinel-2 GSD).
    Uses inbound rounding to ensure the physical ground footprint fits strictly within the single-tile limit
    (<= 2560.0m) without discretization overflow into adjacent raster cells.
    """
    lat_rad = np.radians(lat)
    m_per_deg_lat = 111132.95 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    half_m = tile_size_m / 2.0
    d_lat = half_m / m_per_deg_lat
    d_lon = half_m / m_per_deg_lon
    min_lon = round(math.ceil((lon - d_lon) * 1e6) / 1e6, 6)
    min_lat = round(math.ceil((lat - d_lat) * 1e6) / 1e6, 6)
    max_lon = round(math.floor((lon + d_lon) * 1e6) / 1e6, 6)
    max_lat = round(math.floor((lat + d_lat) * 1e6) / 1e6, 6)
    return [min_lon, min_lat, max_lon, max_lat]

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


def _call_gemini_rest_api(api_key: str, system_prompt: str, model_name: str = "gemini-3.5-flash") -> Optional[Dict[str, Any]]:
    """
    Direct REST API client for Google Gemini without requiring third-party SDK dependencies.
    Tries candidate active models (gemini-3.5-flash, gemini-3.7-flash, gemini-3.8-flash, gemini-flash-latest).
    """
    candidate_models = [model_name]
    for alt in ["gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-flash-latest"]:
        if alt not in candidate_models:
            candidate_models.append(alt)

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

    for m in candidate_models:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
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
            logger.debug(f"Gemini REST call failed for model {m}: {e}")
            continue

    return None


def parse_geographic_coordinates(text: str) -> Optional[Tuple[float, float, Optional[List[float]], str]]:
    """
    Detects and validates numeric coordinates (decimal lat/lon pair or 4-element BBOX)
    before sending text to external geocoders.
    
    Supports:
    - Decimal coordinate pairs: "19.0760, 72.8777", "19.0760 72.8777", "lat: 19.076, lon: 72.877"
    - Explicit Bounding Boxes: "[72.80, 18.93, 72.85, 18.97]", "72.80, 18.93, 72.85, 18.97"
    - Coordinate ordering and hemisphere range validation (lat in [-90, 90], lon in [-180, 180]).
    
    Returns:
        Tuple of (latitude, longitude, optional_explicit_bbox, provider_name) or None if not numeric.
    """
    if not text or not text.strip():
        return None

    clean_text = text.strip()

    # 1. Check for 4-element Bounding Box: [min_lon, min_lat, max_lon, max_lat]
    bbox_match = re.search(
        r"\[?\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*\]?",
        clean_text
    )
    if bbox_match:
        v1, v2, v3, v4 = [float(bbox_match.group(i)) for i in range(1, 5)]
        # Case A: Standard GIS [min_lon, min_lat, max_lon, max_lat]
        if -180.0 <= v1 <= 180.0 and -180.0 <= v3 <= 180.0 and -90.0 <= v2 <= 90.0 and -90.0 <= v4 <= 90.0 and v1 < v3 and v2 < v4:
            lat = (v2 + v4) / 2.0
            lon = (v1 + v3) / 2.0
            return lat, lon, [round(v1, 4), round(v2, 4), round(v3, 4), round(v4, 4)], "Direct Bounding Box Input"
        # Case B: [min_lat, min_lon, max_lat, max_lon]
        elif -90.0 <= v1 <= 90.0 and -90.0 <= v3 <= 90.0 and -180.0 <= v2 <= 180.0 and -180.0 <= v4 <= 180.0 and v1 < v3 and v2 < v4:
            lat = (v1 + v3) / 2.0
            lon = (v2 + v4) / 2.0
            return lat, lon, [round(v2, 4), round(v1, 4), round(v4, 4), round(v3, 4)], "Direct Bounding Box Input"

    # 2. Check for labeled coordinates (e.g., lat: 21.57, lon: 83.87 or lon: 83.87, lat: 21.57)
    lat_match = re.search(r"\b(?:lat|latitude)\s*[:=]?\s*([-+]?\d+\.?\d*)", clean_text, re.IGNORECASE)
    lon_match = re.search(r"\b(?:lon|lng|longitude)\s*[:=]?\s*([-+]?\d+\.?\d*)", clean_text, re.IGNORECASE)
    if lat_match and lon_match:
        lat = float(lat_match.group(1))
        lon = float(lon_match.group(1))
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Latitude out of physical bounds [-90, 90]: {lat}")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Longitude out of physical bounds [-180, 180]: {lon}")
        return lat, lon, None, "Direct Numeric Coordinate Input"

    # 3. Check for unlabeled 2-element Coordinate Pair: n1, n2
    coord_match = re.search(
        r"([-+]?\d+\.?\d*)\s*(?:[,\s/]+)\s*([-+]?\d+\.?\d*)",
        clean_text,
        re.IGNORECASE
    )
    if coord_match:
        n1 = float(coord_match.group(1))
        n2 = float(coord_match.group(2))

        # Check for longitude-first ordering e.g. 105.2, 21.57 (n1 > 90 cannot be latitude)
        if abs(n1) > 90.0 and abs(n2) <= 90.0:
            lat, lon = n2, n1
        elif abs(n1) <= 90.0 and abs(n2) <= 180.0:
            lat, lon = n1, n2
        else:
            raise ValueError(
                f"Invalid coordinate range: ({n1}, {n2}). Latitude must be between -90 and 90, longitude between -180 and 180."
            )

        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError(
                f"Coordinates out of physical bounds: lat={lat} (must be [-90, 90]), lon={lon} (must be [-180, 180])."
            )

        return lat, lon, None, "Direct Numeric Coordinate Input"

    return None


def _geocode_location_live(
    location_query: str, 
    google_maps_key: Optional[str] = None
) -> Tuple[float, float, Optional[List[float]], str]:
    """
    Geocodes a location query to (latitude, longitude, explicit_bbox, provider_name).
    
    Order of operations:
    1. Direct numeric coordinate / BBOX detection and range validation (bypasses network).
    2. Google Maps Geocoding API (if key present).
    3. OpenStreetMap Nominatim live REST HTTP with progressive relaxation.
    4. geopy Nominatim (if installed).
    5. Raises a clear ValueError on failure (never silently substitutes South Mumbai).
    """
    if not location_query or not location_query.strip():
        raise ValueError("Location query cannot be empty. Please provide a place name or decimal coordinates.")

    raw_query = location_query.strip()

    # Step 1: Detect numeric coordinates / bounding boxes before network calls
    num_result = parse_geographic_coordinates(raw_query)
    if num_result is not None:
        return num_result

    # Step 2: Natural-language candidate queries for complex compounds
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
                        return float(loc["lat"]), float(loc["lng"]), None, "Google Maps Geocoding API"
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
                    return lat, lon, None, "OpenStreetMap Nominatim (Live REST)"
        except Exception as e:
            logger.debug(f"HTTP Nominatim lookup failed for '{q}': {e}")

    # 3. Try geopy if installed
    if HAS_GEOPY and Nominatim is not None:
        try:
            geolocator = Nominatim(user_agent="voxel_stage0_parser")
            for q in candidates:
                loc = geolocator.geocode(q, timeout=5)
                if loc:
                    return float(loc.latitude), float(loc.longitude), None, "geopy Nominatim"
        except Exception as e:
            logger.debug(f"geopy Nominatim lookup failed: {e}")

    # CRITICAL: Never silently substitute South Mumbai or preset on geocoding failure.
    raise ValueError(
        f"Location resolution failed: '{raw_query}' could not be resolved by available geocoders. "
        "Please check spelling or provide numeric latitude/longitude coordinates (e.g. '19.0760, 72.8777')."
    )


class VoxelStageZeroParser:
    """
    Stage 0 Natural Language Query Compiler for the Voxel Earth Observation platform.
    Uses centralized APIManager for credentials, with Google Gemini (SDK or REST),
    live OpenStreetMap Nominatim geocoding, and deterministic offline rule-based fallbacks.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.5-flash",
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
        Converts place name or numeric coordinates into a ~2.56 km x 2.56 km geospatial BBOX:
        [min_lon, min_lat, max_lon, max_lat] and returns geocoder name.
        """
        gmaps_key = self.api_mgr.get_google_maps_key() if self.api_mgr else None
        lat, lon, explicit_bbox, provider = _geocode_location_live(location_query, google_maps_key=gmaps_key)
        
        if explicit_bbox is not None:
            return explicit_bbox, provider

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
        location_query = None
        # Check if coordinates exist directly in prompt first
        num_coords = parse_geographic_coordinates(user_prompt)
        if num_coords is not None:
            lat, lon, explicit_bbox, _ = num_coords
            if explicit_bbox is not None:
                location_query = f"{explicit_bbox[0]}, {explicit_bbox[1]}, {explicit_bbox[2]}, {explicit_bbox[3]}"
            else:
                location_query = f"{lat:.4f}, {lon:.4f}"
        else:
            loc_patterns = [
                r"\b(?:in|around|near|across|over|for|at)\s+([A-Z][a-zA-Z0-9\s,\-\.]+?)(?:\s+(?:for|to|with|using|and|\.|\?|$)|$)",
                r"\b(?:in|around|near|across|over|for|at)\s+([a-zA-Z0-9\s,\-\.]+?)(?:\s+(?:for|to|with|using|and|\.|\?|$)|$)"
            ]
            for pat in loc_patterns:
                match = re.search(pat, user_prompt)
                if match:
                    extracted = match.group(1).strip()
                    if len(extracted) > 2 and extracted.lower() not in ["the", "a", "an", "all", "our", "general", "each"]:
                        location_query = extracted
                        break
        if not location_query:
            location_query = DEFAULT_LOCATION_NAME

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

    def parse(self, user_prompt: str, bbox: Optional[List[float]] = None) -> VoxelExecutionContract:
        """
        Sends prompt to LLM and returns structured Execution Contract.
        Uses Gemini SDK -> Gemini REST -> Rule-based fallback.
        Supports explicit bbox override to avoid redundant network lookups.
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

        # Geocode to real BBOX coordinates (or use explicit bbox if provided)
        if bbox is not None:
            bbox_coords = [round(float(x), 6) for x in bbox]
            geocoder_used = "Interactive Map / Explicit BBOX Input"
            location_query = f"Custom AOI [{bbox_coords[0]:.4f}, {bbox_coords[1]:.4f}, {bbox_coords[2]:.4f}, {bbox_coords[3]:.4f}]"
        else:
            bbox_coords, geocoder_used = self._geocode_to_bbox(location_query)

        return VoxelExecutionContract(
            prompt=user_prompt,
            location_name=location_query,
            bbox=bbox_coords,
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
