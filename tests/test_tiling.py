"""
Tests for Phase 4 geospatial raster processing and 256x256 tiling.
Uses synthetic GeoTIFFs created via rasterio without external downloads.
"""

import os
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import tempfile
from pathlib import Path

from module1_ingestion.config import SENTINEL_2_L2A_SPECTRAL_BANDS, TILE_SIZE, OUTPUT_DTYPE
from module1_ingestion.tile_processor import (
    inspect_raster,
    generate_tile_windows,
    process_raster_to_tiles
)

@pytest.fixture
def temp_geotiff_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

def create_synthetic_geotiff(
    path: str, 
    width: int, 
    height: int, 
    transform=None, 
    crs=CRS.from_epsg(32631), 
    dtype=rasterio.float32,
    val=1.0
):
    if transform is None:
        transform = from_origin(300000.0, 5000000.0, 10.0, 10.0)
        
    data = np.full((1, height, width), val, dtype=dtype)
    
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=0.0
    ) as dst:
        dst.write(data)

def test_inspect_raster(temp_geotiff_dir):
    path = temp_geotiff_dir / "test_inspect.tif"
    create_synthetic_geotiff(str(path), 512, 512)
    
    meta = inspect_raster(str(path))
    assert meta["width"] == 512
    assert meta["height"] == 512
    assert "32631" in meta["crs"]
    assert meta["resolution"] == (10.0, 10.0)

def test_missing_raster_inspection():
    with pytest.raises(FileNotFoundError):
        inspect_raster("non_existent.tif")

def test_generate_tile_windows_exact_divisible():
    windows = list(generate_tile_windows(512, 512, tile_size=256, edge_policy="discard"))
    assert len(windows) == 4 # 2x2 grid
    for r, c, w in windows:
        assert w.width == 256
        assert w.height == 256

def test_generate_tile_windows_edge_discard():
    # 600 / 256 = 2 complete tiles, plus a remainder of 88. 
    # With discard policy, we should only get the 2x2 grid (4 tiles total)
    windows = list(generate_tile_windows(600, 600, tile_size=256, edge_policy="discard"))
    assert len(windows) == 4
    for r, c, w in windows:
        assert w.width == 256
        assert w.height == 256

def test_unsupported_edge_policy():
    with pytest.raises(ValueError, match="Unsupported edge policy"):
        list(generate_tile_windows(256, 256, edge_policy="pad"))

def test_process_raster_to_tiles_same_grid(temp_geotiff_dir):
    band_paths = {}
    ref_transform = from_origin(300000.0, 5000000.0, 10.0, 10.0)
    
    for band in SENTINEL_2_L2A_SPECTRAL_BANDS:
        p = temp_geotiff_dir / f"{band}.tif"
        create_synthetic_geotiff(str(p), 512, 512, transform=ref_transform, val=2.0)
        band_paths[band] = str(p)
        
    tiles = list(process_raster_to_tiles(band_paths, reference_band_key="B04"))
    
    assert len(tiles) == 4
    for t in tiles:
        # Check shape matches the 12-band intermediate representation
        assert t["tile_data"].shape == (256, 256, 12)
        assert t["tile_data"].dtype == OUTPUT_DTYPE
        assert np.isclose(np.mean(t["tile_data"]), 2.0)
        
    # The intermediate output explicitly is NOT the 13-band output yet
    assert tiles[0]["tile_data"].shape[-1] != 13

def test_process_raster_to_tiles_harmonization_and_crs(temp_geotiff_dir):
    band_paths = {}
    
    # B04 is 10m ref grid
    ref_transform = from_origin(300000.0, 5000000.0, 10.0, 10.0)
    p_b04 = temp_geotiff_dir / "B04.tif"
    create_synthetic_geotiff(str(p_b04), 256, 256, transform=ref_transform, val=1.0)
    band_paths["B04"] = str(p_b04)
    
    # B05 is 20m res
    p_b05 = temp_geotiff_dir / "B05.tif"
    b05_transform = from_origin(300000.0, 5000000.0, 20.0, 20.0)
    create_synthetic_geotiff(str(p_b05), 128, 128, transform=b05_transform, val=2.0)
    band_paths["B05"] = str(p_b05)
    
    # B06 in different CRS
    p_b06 = temp_geotiff_dir / "B06.tif"
    b06_crs = CRS.from_epsg(4326)
    b06_transform = from_origin(2.2, 45.1, 0.0001, 0.0001)
    create_synthetic_geotiff(str(p_b06), 256, 256, transform=b06_transform, crs=b06_crs, val=3.0)
    band_paths["B06"] = str(p_b06)
    
    # Fill remaining bands identical to B04
    for band in SENTINEL_2_L2A_SPECTRAL_BANDS:
        if band not in band_paths:
            p = temp_geotiff_dir / f"{band}.tif"
            create_synthetic_geotiff(str(p), 256, 256, transform=ref_transform, val=4.0)
            band_paths[band] = str(p)
            
    tiles = list(process_raster_to_tiles(band_paths, reference_band_key="B04"))
    
    assert len(tiles) == 1
    t = tiles[0]
    
    assert t["tile_data"].shape == (256, 256, 12)
    
    # Extract indices based on SENTINEL_2_L2A_SPECTRAL_BANDS order
    b04_idx = SENTINEL_2_L2A_SPECTRAL_BANDS.index("B04")
    b05_idx = SENTINEL_2_L2A_SPECTRAL_BANDS.index("B05")
    b06_idx = SENTINEL_2_L2A_SPECTRAL_BANDS.index("B06")
    b01_idx = SENTINEL_2_L2A_SPECTRAL_BANDS.index("B01")
    
    assert np.isclose(np.mean(t["tile_data"][:, :, b04_idx]), 1.0)
    assert np.isclose(np.mean(t["tile_data"][:, :, b05_idx]), 2.0)
    assert np.isclose(np.mean(t["tile_data"][:, :, b01_idx]), 4.0)
    # B06 reprojection value will be non-zero (since it's tested) but may vary due to bilinear resampling at borders
    assert np.mean(t["tile_data"][:, :, b06_idx]) > 0

def test_missing_band_fails(temp_geotiff_dir):
    band_paths = {"B04": str(temp_geotiff_dir / "B04.tif")}
    create_synthetic_geotiff(band_paths["B04"], 256, 256)
    
    with pytest.raises(ValueError, match="Missing required L2A spectral band"):
        list(process_raster_to_tiles(band_paths, reference_band_key="B04"))
