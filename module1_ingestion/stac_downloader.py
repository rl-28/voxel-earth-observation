"""
STAC-based Sentinel-2 Level-2A data acquisition.

Provides programmatic STAC-based ingestion of Sentinel-2 Level-2A surface reflectance data.
Responsible for: STAC connection, catalog discovery, search, AOI/date/cloud filtering,
STAC item selection, asset discovery, and raw asset download.

Does NOT produce the final (256, 256, 13) spatial-spectral tensor (handled in tile_processor).
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
from pystac_client import Client

from .config import (
    COPERNICUS_STAC_URL,
    SENTINEL_2_L2A_COLLECTION,
    DEFAULT_CLOUD_COVER,
    STAC_TIMEOUT_SEC,
    RAW_DATA_DIR
)

logger = logging.getLogger(__name__)

class STACProvider:
    """Abstract base class for STAC providers."""
    def search_sentinel2_l2a(self, aoi: Dict[str, Any], start_date: str, end_date: str, cloud_cover: float, max_items: int) -> List[Any]:
        raise NotImplementedError
        
    def get_item_assets(self, item: Any) -> Dict[str, Any]:
        raise NotImplementedError
        
    def download_asset(self, asset_url: str, dest_path: str, overwrite: bool = False) -> str:
        raise NotImplementedError

class CopernicusSTACProvider(STACProvider):
    """Copernicus Data Space Ecosystem STAC Provider."""
    
    def __init__(self, endpoint_url: str = COPERNICUS_STAC_URL):
        self.endpoint_url = endpoint_url
        self._client = None
        
    def connect_to_catalog(self) -> Client:
        if not self._client:
            self._client = Client.open(self.endpoint_url)
        return self._client
        
    def search_sentinel2_l2a(
        self, 
        aoi: Dict[str, Any], 
        start_date: str, 
        end_date: str, 
        cloud_cover: float = DEFAULT_CLOUD_COVER, 
        max_items: int = 10
    ) -> List[Any]:
        """Searches for Sentinel-2 L2A STAC items."""
        # AOI Validation
        if not isinstance(aoi, dict) or "type" not in aoi or aoi["type"] not in ["Polygon", "MultiPolygon", "Point"]:
            raise ValueError("AOI must be a valid GeoJSON dictionary (e.g., {'type': 'Polygon', 'coordinates': [...]}).")
            
        # Date Validation
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            if start_dt > end_dt:
                raise ValueError("start_date cannot be after end_date.")
        except ValueError as e:
            if "start_date" in str(e):
                raise
            raise ValueError("Dates must be valid ISO format strings.")
            
        # Cloud cover validation
        if not (0 <= cloud_cover <= 100):
            raise ValueError("Cloud cover threshold must be between 0 and 100.")
            
        client = self.connect_to_catalog()
        
        search = client.search(
            collections=[SENTINEL_2_L2A_COLLECTION],
            intersects=aoi,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lte": cloud_cover}},
            max_items=max_items
        )
        
        return list(search.items())
        
    def get_item_assets(self, item: Any) -> Dict[str, Any]:
        """Discovers available assets for a STAC item."""
        # MAPPING NOTE:
        # Sentinel-2 L2A provides 12 surface-reflectance spectral bands.
        # The project interface currently requires 13 channels. The mapping
        # of the 13th channel is a project-level specification decision and
        # must not be fabricated by the ingestion pipeline. 
        # The 13th channel mapping remains: Pending project-level specification clarification.
        # We discover all available assets here. Mapping to the explicit 13-band stack
        # is deferred to the tile processing phase.
        return item.assets
        
    def download_asset(self, asset_url: str, dest_path: str, overwrite: bool = False) -> str:
        """Downloads a raw asset to disk using streaming to conserve memory."""
        if os.path.exists(dest_path) and not overwrite:
            logger.info(f"File {dest_path} already exists. Skipping download.")
            return dest_path
            
        # Make sure directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
        # NOTE: Copernicus asset download often requires OAuth tokens.
        # Ensure credentials (if needed) are passed via environment variables (e.g., COPERNICUS_ACCESS_TOKEN).
        headers = {}
        token = os.getenv("COPERNICUS_ACCESS_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
            
        try:
            response = requests.get(asset_url, stream=True, headers=headers, timeout=STAC_TIMEOUT_SEC)
            response.raise_for_status()
            
            # Streaming download to avoid memory spikes
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return dest_path
        except Exception as e:
            # Clean up partially downloaded file
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise RuntimeError(f"Failed to download asset {asset_url}: {e}")

class BhuNidhiProvider(STACProvider):
    """
    Placeholder for ISRO BhuNidhi STAC Provider.
    Currently, an official verified STAC-compatible BhuNidhi endpoint is not established.
    This class serves as a stub for future integration once verified.
    """
    def search_sentinel2_l2a(self, *args, **kwargs):
        raise NotImplementedError("BhuNidhi STAC endpoint is pending verification.")
