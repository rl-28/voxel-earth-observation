"""
utils/api_manager.py

Centralized API Management & Security Infrastructure for the Voxel Platform.
Handles secure environment variable loading, multi-provider credentials,
status verification, diagnostics, and graceful offline fallbacks.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Attempt to load python-dotenv if available
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


def _is_valid_key(val: Optional[str]) -> bool:
    """Checks if key is non-empty and not a template placeholder."""
    if not val or not isinstance(val, str):
        return False
    val_clean = val.strip()
    if not val_clean:
        return False
    placeholders = [
        "your_gemini_api_key_here",
        "your_openai_api_key_here",
        "your_groq_api_key_here",
        "your_cdse_client_id_here",
        "your_cdse_client_secret_here",
        "your_google_maps_api_key_here",
        "your_mapbox_api_key_here",
        "your_wandb_api_key_here",
        "your_api_key_here",
        "none",
        "null"
    ]
    return val_clean.lower() not in placeholders


def _mask_key(key: Optional[str]) -> str:
    """Masks secret key for safe diagnostic display."""
    if not _is_valid_key(key):
        return "Not Set"
    k = str(key).strip()
    if len(k) <= 8:
        return "****"
    return f"{k[:4]}...{k[-4:]}"


class APIManager:
    """
    Singleton API & Environment Manager for Voxel EO Platform.
    Sources credentials from .env and environment variables.
    """
    _instance: Optional["APIManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(APIManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, env_path: Optional[str] = None, auto_load: bool = True):
        if getattr(self, "_initialized", False):
            return

        self._project_root = Path(__file__).resolve().parent.parent
        self._env_file = Path(env_path) if env_path is not None else (self._project_root / ".env")
        if auto_load:
            self._load_environment()
        self._initialized = True

    def _load_environment(self) -> None:
        """Loads environment variables from .env file with fallback parsing."""
        if HAS_DOTENV and self._env_file.exists():
            load_dotenv(dotenv_path=self._env_file, override=False)
            logger.debug(f"Loaded environment variables from {self._env_file}")
        elif self._env_file.exists():
            # Minimal pure-Python fallback .env parser if python-dotenv is not installed
            try:
                with open(self._env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k not in os.environ:
                                os.environ[k] = v
            except Exception as e:
                logger.warning(f"Failed to parse .env file: {e}")

    # --------------------------------------------------------------------------
    # 1. LLM Credential Getters
    # --------------------------------------------------------------------------
    def get_gemini_key(self) -> Optional[str]:
        """Returns Gemini API key from GEMINI_API_KEY or GOOGLE_API_KEY."""
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return key if _is_valid_key(key) else None

    def get_openai_key(self) -> Optional[str]:
        """Returns OpenAI API key from OPENAI_API_KEY."""
        key = os.getenv("OPENAI_API_KEY")
        return key if _is_valid_key(key) else None

    def get_groq_key(self) -> Optional[str]:
        """Returns Groq API key from GROQ_API_KEY."""
        key = os.getenv("GROQ_API_KEY")
        return key if _is_valid_key(key) else None

    def has_gemini(self) -> bool:
        return self.get_gemini_key() is not None

    def has_openai(self) -> bool:
        return self.get_openai_key() is not None

    def has_groq(self) -> bool:
        return self.get_groq_key() is not None

    def has_llm_provider(self) -> bool:
        return self.has_gemini() or self.has_openai() or self.has_groq()

    def get_active_llm_provider(self) -> str:
        """Determines active LLM provider priority order: Gemini -> OpenAI -> Groq -> Rule-Based Fallback."""
        if self.has_gemini():
            return "gemini"
        if self.has_openai():
            return "openai"
        if self.has_groq():
            return "groq"
        return "rule_based_fallback"

    # --------------------------------------------------------------------------
    # 2. Geospatial & Satellite Ingestion Credentials
    # --------------------------------------------------------------------------
    def get_copernicus_stac_url(self) -> str:
        return os.getenv(
            "COPERNICUS_STAC_URL",
            "https://sh.dataspace.copernicus.eu/api/catalogs/stac"
        )

    def get_copernicus_cdse_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        client_id = os.getenv("COPERNICUS_CDSE_CLIENT_ID")
        client_secret = os.getenv("COPERNICUS_CDSE_CLIENT_SECRET")
        cid = client_id if _is_valid_key(client_id) else None
        csec = client_secret if _is_valid_key(client_secret) else None
        return cid, csec

    def has_copernicus_cdse(self) -> bool:
        cid, csec = self.get_copernicus_cdse_credentials()
        return cid is not None and csec is not None

    def get_google_maps_key(self) -> Optional[str]:
        key = os.getenv("GOOGLE_MAPS_API_KEY")
        return key if _is_valid_key(key) else None

    def has_google_maps(self) -> bool:
        return self.get_google_maps_key() is not None

    # --------------------------------------------------------------------------
    # 3. Visualization & Basemap Credentials
    # --------------------------------------------------------------------------
    def get_mapbox_key(self) -> Optional[str]:
        key = os.getenv("MAPBOX_API_KEY")
        return key if _is_valid_key(key) else None

    def has_mapbox(self) -> bool:
        return self.get_mapbox_key() is not None

    # --------------------------------------------------------------------------
    # 4. MLOps & Experiment Tracking Credentials
    # --------------------------------------------------------------------------
    def get_wandb_key(self) -> Optional[str]:
        key = os.getenv("WANDB_API_KEY")
        return key if _is_valid_key(key) else None

    def has_wandb(self) -> bool:
        return self.get_wandb_key() is not None

    # --------------------------------------------------------------------------
    # 5. Health Check & Diagnostic Inspection
    # --------------------------------------------------------------------------
    def health_check(self) -> Dict[str, Any]:
        """
        Runs comprehensive security & capability health check across all Voxel stages.
        All secret keys are safely masked.
        """
        gemini_k = self.get_gemini_key()
        openai_k = self.get_openai_key()
        groq_k = self.get_groq_key()
        cdse_id, cdse_sec = self.get_copernicus_cdse_credentials()
        gmaps_k = self.get_google_maps_key()
        mapbox_k = self.get_mapbox_key()
        wandb_k = self.get_wandb_key()

        return {
            "environment": os.getenv("VOXEL_ENV", "development"),
            "active_llm_provider": self.get_active_llm_provider(),
            "stage_0_llm": {
                "gemini_available": self.has_gemini(),
                "gemini_key_masked": _mask_key(gemini_k),
                "openai_available": self.has_openai(),
                "openai_key_masked": _mask_key(openai_k),
                "groq_available": self.has_groq(),
                "groq_key_masked": _mask_key(groq_k),
                "offline_fallback_ready": True
            },
            "stage_1_ingestion": {
                "copernicus_stac_url": self.get_copernicus_stac_url(),
                "copernicus_cdse_configured": self.has_copernicus_cdse(),
                "cdse_client_id_masked": _mask_key(cdse_id),
                "google_maps_geocoding_available": self.has_google_maps(),
                "google_maps_key_masked": _mask_key(gmaps_k),
                "nominatim_geocoding_ready": True
            },
            "visualization": {
                "mapbox_available": self.has_mapbox(),
                "mapbox_key_masked": _mask_key(mapbox_k)
            },
            "mlops_tracking": {
                "wandb_available": self.has_wandb(),
                "wandb_key_masked": _mask_key(wandb_k),
                "wandb_project": os.getenv("WANDB_PROJECT", "voxel-earth-observation")
            }
        }


# Global singleton instance provider
_global_api_manager: Optional[APIManager] = None


def get_api_manager(env_path: Optional[str] = None) -> APIManager:
    """Returns the singleton APIManager instance."""
    global _global_api_manager
    if _global_api_manager is None:
        _global_api_manager = APIManager(env_path=env_path)
    return _global_api_manager
