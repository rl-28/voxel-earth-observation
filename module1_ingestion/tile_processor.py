"""
Geospatial Raster Processing and Tiling Pipeline.

Transforms raw geospatial raster assets into spatially aligned, non-overlapping
256x256 tiles, matching a specific reference grid. Maintains a 12-band L2A
intermediate representation prior to final 13-channel project specification.
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import Affine
from typing import Dict, Any, List, Generator, Tuple

from .config import (
    TILE_SIZE,
    SENTINEL_2_L2A_SPECTRAL_BANDS,
    OUTPUT_DTYPE
)

logger = logging.getLogger(__name__)

def inspect_raster(filepath: str) -> Dict[str, Any]:
    """
    Inspects a raster file without loading the entire raster array into memory.
    
    Args:
        filepath: Path to the GeoTIFF/raster file.
        
    Returns:
        Dict containing metadata: width, height, crs, transform, resolution, bounds, count, dtype, nodata.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raster file not found: {filepath}")
        
    with rasterio.open(filepath) as src:
        meta = {
            "width": src.width,
            "height": src.height,
            "crs": src.crs.to_string() if src.crs else None,
            "transform": src.transform,
            "resolution": src.res,
            "bounds": src.bounds,
            "count": src.count,
            "dtype": src.dtypes[0],
            "nodata": src.nodatavals[0] if src.nodatavals else None
        }
        return meta

def generate_tile_windows(
    width: int, 
    height: int, 
    tile_size: int = TILE_SIZE, 
    edge_policy: str = "discard"
) -> Generator[Tuple[int, int, Window], None, None]:
    """
    Generates non-overlapping rasterio Windows for a given grid size.
    
    Args:
        width: Full grid width.
        height: Full grid height.
        tile_size: Spatial size of the square tile.
        edge_policy: 'discard' to ignore incomplete edge windows.
        
    Yields:
        Tuple of (row_idx, col_idx, rasterio.windows.Window)
    """
    if edge_policy not in ["discard"]:
        raise ValueError(f"Unsupported edge policy: {edge_policy}. Only 'discard' is currently supported.")
        
    for row_idx, row_start in enumerate(range(0, height, tile_size)):
        for col_idx, col_start in enumerate(range(0, width, tile_size)):
            # Determine if this window is a complete tile
            if row_start + tile_size > height or col_start + tile_size > width:
                if edge_policy == "discard":
                    continue
                    
            window = Window(col_off=col_start, row_off=row_start, width=tile_size, height=tile_size)
            yield row_idx, col_idx, window

def get_window_transform(base_transform: Affine, window: Window) -> Affine:
    """Calculates the affine transform for a specific window."""
    return rasterio.windows.transform(window, base_transform)

def process_raster_to_tiles(
    band_paths: Dict[str, str], 
    reference_band_key: str = "B04"
) -> Generator[Dict[str, Any], None, None]:
    """
    Processes a dictionary of band paths into standardized intermediate 256x256x12 tiles.
    Uses windowed reading and dynamic reprojection to heavily conserve memory.
    
    Args:
        band_paths: Dictionary mapping band names (e.g. 'B01') to file paths.
        reference_band_key: The band to use as the spatial reference grid (typically a 10m band like B04).
        
    Yields:
        Dictionaries containing 'tile_data', 'row_idx', 'col_idx', 'transform', 'crs'.
        The 'tile_data' has shape (256, 256, 12).
    """
    # 1. Validation
    if reference_band_key not in band_paths:
        raise ValueError(f"Reference band {reference_band_key} is missing from the input paths.")
        
    for expected_band in SENTINEL_2_L2A_SPECTRAL_BANDS:
        if expected_band not in band_paths:
            raise ValueError(f"Missing required L2A spectral band: {expected_band}")
            
    # 2. Reference Grid Setup
    reference_meta = inspect_raster(band_paths[reference_band_key])
    
    # 3. Generate Tiles one by one (Memory disciplined approach)
    for row_idx, col_idx, window in generate_tile_windows(reference_meta["width"], reference_meta["height"]):
        tile_transform = get_window_transform(reference_meta["transform"], window)
        
        # Build the 12-band intermediate tile stack
        tile_stack = np.empty((window.height, window.width, len(SENTINEL_2_L2A_SPECTRAL_BANDS)), dtype=OUTPUT_DTYPE)
        
        for i, band_name in enumerate(SENTINEL_2_L2A_SPECTRAL_BANDS):
            path = band_paths[band_name]
            
            with rasterio.open(path) as src:
                src_crs = src.crs.to_string() if src.crs else None
                ref_crs = reference_meta["crs"]
                
                # Check for perfect spatial alignment
                needs_reprojection = (
                    src_crs != ref_crs or 
                    src.transform != reference_meta["transform"] or
                    src.res != reference_meta["resolution"]
                )
                
                if needs_reprojection:
                    # Reproject from the source dataset directly into the 256x256 tile buffer
                    # rasterio.warp.reproject handles pulling only the necessary bounds from the source
                    band_buffer = np.empty((window.height, window.width), dtype=OUTPUT_DTYPE)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=band_buffer,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=tile_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear
                    )
                    tile_stack[:, :, i] = band_buffer
                else:
                    # Perfect alignment, perform a direct windowed read
                    tile_stack[:, :, i] = src.read(1, window=window).astype(OUTPUT_DTYPE)
                    
        yield {
            "tile_data": tile_stack,
            "row_idx": row_idx,
            "col_idx": col_idx,
            "transform": tile_transform,
            "crs": reference_meta["crs"],
            "bounds": rasterio.windows.bounds(window, reference_meta["transform"]),
            # Check for Nodata. We do not filter them here, just flag.
            "nodata_present": np.isnan(tile_stack).any() or (tile_stack == 0.0).any()
        }
