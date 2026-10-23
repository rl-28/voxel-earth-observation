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
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import requests
from pystac_client import Client

from .config import (
    DEFAULT_STAC_PROVIDER,
    COPERNICUS_STAC_URL,
    SENTINEL_2_L2A_COLLECTION,
    EARTH_SEARCH_STAC_URL,
    EARTH_SEARCH_COLLECTION,
    COPERNICUS_BAND_TO_ASSET,
    EARTH_SEARCH_BAND_TO_ASSET,
    SENTINEL_2_L2A_SPECTRAL_BANDS,
    DEFAULT_CLOUD_COVER,
    STAC_TIMEOUT_SEC,
    RAW_DATA_DIR
)

logger = logging.getLogger(__name__)

class STACProvider:
    """Abstract base class for STAC providers."""
    def search_sentinel2_l2a(self, aoi: Dict[str, Any], start_date: str, end_date: str, cloud_cover: float = DEFAULT_CLOUD_COVER, max_items: int = 10) -> List[Any]:
        raise NotImplementedError
        
    def get_item_assets(self, item: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def get_band_asset_key(self, band_name: str) -> str:
        raise NotImplementedError
        
    def download_asset(self, asset_url: str, dest_path: str, overwrite: bool = False) -> str:
        raise NotImplementedError

    def download_bands(self, item: Any, target_dir: str, bands: Optional[List[str]] = None, overwrite: bool = False) -> Dict[str, str]:
        raise NotImplementedError

class CopernicusSTACProvider(STACProvider):
    """Copernicus Data Space Ecosystem STAC Provider."""
    
    def __init__(self, endpoint_url: str = COPERNICUS_STAC_URL):
        self.endpoint_url = endpoint_url
        self.collection = SENTINEL_2_L2A_COLLECTION
        self.band_to_asset = dict(COPERNICUS_BAND_TO_ASSET)
        self._client = None
        
    def connect_to_catalog(self) -> Client:
        if not self._client:
            self._client = Client.open(self.endpoint_url)
        return self._client

    def get_band_asset_key(self, band_name: str) -> str:
        return self.band_to_asset.get(band_name, band_name)
        
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
            collections=[self.collection],
            intersects=aoi,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lte": cloud_cover}},
            max_items=max_items
        )
        
        return list(search.items())
        
    def get_item_assets(self, item: Any) -> Dict[str, Any]:
        """Discovers available assets for a STAC item."""
        return item.assets
        
    def download_asset(self, asset_url: str, dest_path: str, overwrite: bool = False) -> str:
        """Downloads a raw asset to disk using streaming to conserve memory."""
        if os.path.exists(dest_path) and not overwrite:
            logger.info(f"File {dest_path} already exists. Skipping download.")
            return dest_path
            
        # Make sure directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
        # NOTE: Copernicus asset download often requires OAuth tokens.
        headers = {}
        token = os.getenv("COPERNICUS_ACCESS_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
            
        try:
            response = requests.get(asset_url, stream=True, headers=headers, timeout=STAC_TIMEOUT_SEC)
            response.raise_for_status()
            
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return dest_path
        except Exception as e:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise RuntimeError(f"Failed to download asset {asset_url}: {e}")

    def download_bands(
        self,
        item: Any,
        target_dir: str,
        bands: Optional[List[str]] = None,
        overwrite: bool = False
    ) -> Dict[str, str]:
        requested_bands = bands or SENTINEL_2_L2A_SPECTRAL_BANDS
        band_paths: Dict[str, str] = {}
        os.makedirs(target_dir, exist_ok=True)

        for band_name in requested_bands:
            asset_key = self.get_band_asset_key(band_name)
            if asset_key not in item.assets:
                raise KeyError(
                    f"Asset key '{asset_key}' for band '{band_name}' not found in STAC item '{item.id}'. "
                    f"Available assets: {list(item.assets.keys())}"
                )
            asset_url = item.assets[asset_key].href
            dest_path = os.path.join(target_dir, f"{band_name}.tif")
            downloaded = self.download_asset(asset_url, dest_path, overwrite=overwrite)
            band_paths[band_name] = downloaded

        return band_paths

class EarthSearchSTACProvider(STACProvider):
    """
    Element84 Earth Search STAC Provider (AWS-hosted Sentinel-2 L2A COGs).
    Provides public HTTPS access without credentials.
    """
    
    def __init__(
        self,
        endpoint_url: str = EARTH_SEARCH_STAC_URL,
        collection: str = EARTH_SEARCH_COLLECTION
    ):
        self.endpoint_url = endpoint_url
        self.collection = collection
        self.band_to_asset = dict(EARTH_SEARCH_BAND_TO_ASSET)
        self._client = None

    def connect_to_catalog(self) -> Client:
        if not self._client:
            self._client = Client.open(self.endpoint_url)
        return self._client

    def get_band_asset_key(self, band_name: str) -> str:
        if band_name not in self.band_to_asset:
            raise KeyError(
                f"Band '{band_name}' not mapped in EarthSearchSTACProvider. Available: {list(self.band_to_asset.keys())}"
            )
        return self.band_to_asset[band_name]

    def search_sentinel2_l2a(
        self,
        aoi: Dict[str, Any],
        start_date: str,
        end_date: str,
        cloud_cover: float = DEFAULT_CLOUD_COVER,
        max_items: int = 10
    ) -> List[Any]:
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
            collections=[self.collection],
            intersects=aoi,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lte": cloud_cover}},
            max_items=max_items
        )
        
        return list(search.items())

    def get_item_assets(self, item: Any) -> Dict[str, Any]:
        return item.assets

    def download_asset(self, asset_url: str, dest_path: str, overwrite: bool = False) -> str:
        if os.path.exists(dest_path) and not overwrite:
            logger.info(f"File {dest_path} already exists. Skipping download.")
            return dest_path

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        try:
            response = requests.get(asset_url, stream=True, timeout=STAC_TIMEOUT_SEC)
            response.raise_for_status()

            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return dest_path
        except Exception as e:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise RuntimeError(f"Failed to download asset {asset_url}: {e}")

    def download_bands(
        self,
        item: Any,
        target_dir: str,
        bands: Optional[List[str]] = None,
        bbox: Optional[List[float]] = None,
        overwrite: bool = False
    ) -> Dict[str, str]:
        requested_bands = bands or SENTINEL_2_L2A_SPECTRAL_BANDS
        band_paths: Dict[str, str] = {}
        os.makedirs(target_dir, exist_ok=True)

        for band_name in requested_bands:
            asset_key = self.get_band_asset_key(band_name)
            if asset_key not in item.assets:
                raise KeyError(
                    f"Asset key '{asset_key}' for band '{band_name}' not found in STAC item '{item.id}'. "
                    f"Available assets: {list(item.assets.keys())}"
                )
            asset_url = item.assets[asset_key].href
            dest_path = os.path.join(target_dir, f"{band_name}.tif")


            if bbox is not None:
                try:
                    import math
                    import rasterio
                    from rasterio.windows import Window, from_bounds
                    from rasterio.warp import transform_bounds

                    with rasterio.Env(GDAL_HTTP_TIMEOUT=120, GDAL_HTTP_MAX_RETRY=3, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                        with rasterio.open(asset_url) as src:
                            tb = transform_bounds("EPSG:4326", src.crs, *bbox)
                            minx, miny, maxx, maxy = tb
                            gw = abs(maxx - minx)
                            gh = abs(maxy - miny)

                            # Standard model tile size in meters: 256 px at 10m GSD = 2560m
                            tile_m = 2560.0
                            tiles_x = max(1, int(math.ceil((gw - 50.0) / tile_m)))
                            tiles_y = max(1, int(math.ceil((gh - 50.0) / tile_m)))

                            target_gw = tiles_x * tile_m
                            target_gh = tiles_y * tile_m

                            res_x = abs(src.res[0])
                            res_y = abs(src.res[1])
                            expected_w = int(round(target_gw / res_x))
                            expected_h = int(round(target_gh / res_y))

                            # Check if valid cached crop exists and meets required spatial resolution
                            if os.path.exists(dest_path) and not overwrite and os.path.getsize(dest_path) > 1000:
                                try:
                                    with rasterio.open(dest_path) as chk:
                                        if chk.width >= expected_w and chk.height >= expected_h:
                                            band_paths[band_name] = dest_path
                                            continue
                                except Exception:
                                    pass

                            cx = (minx + maxx) / 2.0
                            cy = (miny + maxy) / 2.0
                            aligned_tb = (
                                cx - target_gw / 2.0,
                                cy - target_gh / 2.0,
                                cx + target_gw / 2.0,
                                cy + target_gh / 2.0,
                            )
                            win = from_bounds(*aligned_tb, transform=src.transform).round_offsets().round_lengths()

                            col_off = max(0, min(src.width - expected_w, int(win.col_off)))
                            row_off = max(0, min(src.height - expected_h, int(win.row_off)))
                            width = min(src.width - col_off, expected_w)
                            height = min(src.height - row_off, expected_h)

                            clamped_win = Window(col_off, row_off, width, height)

                            arr = src.read(1, window=clamped_win)
                            win_transform = rasterio.windows.transform(clamped_win, src.transform)
                            profile = src.profile.copy()
                            profile.update({
                                "height": height,
                                "width": width,
                                "transform": win_transform,
                            })
                            with rasterio.open(dest_path, "w", **profile) as dst:
                                dst.write(arr, 1)
                    band_paths[band_name] = dest_path
                    continue
                except Exception as e:
                    logger.warning(f"Windowed COG read failed for {band_name}, falling back to full download: {e}")

            downloaded = self.download_asset(asset_url, dest_path, overwrite=overwrite)
            band_paths[band_name] = downloaded

        return band_paths

class BhuNidhiProvider(STACProvider):
    """
    Placeholder for ISRO BhuNidhi STAC Provider.
    Currently, an official verified STAC-compatible BhuNidhi endpoint is not established.
    This class serves as a stub for future integration once verified.
    """
    def search_sentinel2_l2a(self, *args, **kwargs):
        raise NotImplementedError("BhuNidhi STAC endpoint is pending verification.")

    def get_item_assets(self, item: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def get_band_asset_key(self, band_name: str) -> str:
        raise NotImplementedError

    def download_asset(self, *args, **kwargs):
        raise NotImplementedError

    def download_bands(self, *args, **kwargs):
        raise NotImplementedError


def get_stac_provider(provider: Optional[str] = None, **kwargs) -> STACProvider:
    """
    Factory function to instantiate a STACProvider.
    Resolves in priority order:
      1. Explicit `provider` argument (e.g. 'earth_search' or 'copernicus')
      2. STAC_PROVIDER environment variable
      3. Default to 'copernicus' (preserves default behavior)
    """
    name = (provider or os.getenv("STAC_PROVIDER", DEFAULT_STAC_PROVIDER)).lower().strip()
    if name in ["earth_search", "earthsearch", "element84", "aws"]:
        return EarthSearchSTACProvider(**kwargs)
    elif name in ["copernicus", "cdse", "sentinel_hub"]:
        return CopernicusSTACProvider(**kwargs)
    elif name in ["bhunidhi", "isro"]:
        return BhuNidhiProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown STAC provider '{name}'. Supported providers: 'copernicus', 'earth_search', 'bhunidhi'"
        )


class STACDownloader:
    """
    High-level STAC acquisition interface.
    Supports constructor provider setting, falling back to STAC_PROVIDER env var,
    defaulting to Copernicus.
    """
    def __init__(self, provider: Optional[Union[str, STACProvider]] = None, **kwargs):
        if isinstance(provider, STACProvider):
            self.provider = provider
        else:
            self.provider = get_stac_provider(provider, **kwargs)

    @property
    def endpoint_url(self) -> str:
        return getattr(self.provider, "endpoint_url", "")

    def search_sentinel2_l2a(self, aoi: Dict[str, Any], start_date: str, end_date: str, cloud_cover: float = DEFAULT_CLOUD_COVER, max_items: int = 10) -> List[Any]:
        return self.provider.search_sentinel2_l2a(aoi=aoi, start_date=start_date, end_date=end_date, cloud_cover=cloud_cover, max_items=max_items)

    def get_item_assets(self, item: Any) -> Dict[str, Any]:
        return self.provider.get_item_assets(item)

    def get_band_asset_key(self, band_name: str) -> str:
        return self.provider.get_band_asset_key(band_name)

    def download_asset(self, asset_url: str, dest_path: str, overwrite: bool = False) -> str:
        return self.provider.download_asset(asset_url=asset_url, dest_path=dest_path, overwrite=overwrite)

    def download_bands(self, item: Any, target_dir: str, bands: Optional[List[str]] = None, overwrite: bool = False) -> Dict[str, str]:
        return self.provider.download_bands(item=item, target_dir=target_dir, bands=bands, overwrite=overwrite)

