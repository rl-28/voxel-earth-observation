import os
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import tempfile
from pathlib import Path
import json

from module1_ingestion.config import SENTINEL_2_L2A_SPECTRAL_BANDS, OUTPUT_DTYPE
from module1_ingestion.evaluation_pipeline import EvaluationPipeline, EVALUATION_REGIONS

@pytest.fixture
def temp_eval_dir():
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

def test_evaluation_region_configuration():
    pipeline = EvaluationPipeline("serbia")
    assert pipeline.region.region_id == "serbia"
    assert pipeline.region.display_name == "Serbia (Ext SR)"
    
    with pytest.raises(ValueError, match="Unsupported evaluation region"):
        EvaluationPipeline("invalid_region")

def test_evaluation_pipeline_e2e(monkeypatch, temp_eval_dir):
    # Monkeypatch the output directory
    serbia_out = temp_eval_dir / "serbia_eval"
    monkeypatch.setattr(EVALUATION_REGIONS["serbia"], "output_directory", serbia_out)
    
    # Create synthetic geotiffs
    input_dir = temp_eval_dir / "input_tiffs"
    input_dir.mkdir()
    
    band_paths = {}
    for b in SENTINEL_2_L2A_SPECTRAL_BANDS:
        p = input_dir / f"{b}.tif"
        create_synthetic_geotiff(str(p), 512, 512, val=0.5)
        band_paths[b] = str(p)
        
    pipeline = EvaluationPipeline("serbia")
    manifest = pipeline.process_local_source(band_paths)
    
    # Check outputs
    assert manifest["region_identity"] == "serbia"
    assert manifest["number_of_tiles"] == 4 # 512x512 with 256x256 tiles -> 4 tiles
    assert len(manifest["tiles"]) == 4
    
    # Check files
    assert serbia_out.exists()
    assert (serbia_out / "manifest.json").exists()
    for tile_meta in manifest["tiles"]:
        tile_path = serbia_out / tile_meta["filename"]
        assert tile_path.exists()
        
        # Check projection verification in manifest
        assert tile_meta["output_crs"] is not None
        assert tile_meta["bounds"] is not None
        
        # Load array
        arr = np.load(tile_path)
        assert arr.shape == (256, 256, 12)
        assert arr.dtype == OUTPUT_DTYPE

def test_missing_source_fails():
    pipeline = EvaluationPipeline("western_balkans")
    with pytest.raises(ValueError, match="source is not configured"):
        pipeline.process_local_source({})

def test_evaluation_determinism(monkeypatch, temp_eval_dir):
    # Monkeypatch the output directory
    wb_out1 = temp_eval_dir / "wb_eval1"
    wb_out2 = temp_eval_dir / "wb_eval2"
    
    # Create synthetic geotiffs
    input_dir = temp_eval_dir / "input_tiffs"
    input_dir.mkdir()
    
    band_paths = {}
    # Use random noise
    np.random.seed(42)
    val = np.random.rand()
    for b in SENTINEL_2_L2A_SPECTRAL_BANDS:
        p = input_dir / f"{b}.tif"
        create_synthetic_geotiff(str(p), 256, 256, val=val)
        band_paths[b] = str(p)
        
    pipeline = EvaluationPipeline("western_balkans")
    
    monkeypatch.setattr(pipeline.region, "output_directory", wb_out1)
    manifest1 = pipeline.process_local_source(band_paths)
    
    monkeypatch.setattr(pipeline.region, "output_directory", wb_out2)
    manifest2 = pipeline.process_local_source(band_paths)
    
    arr1 = np.load(wb_out1 / manifest1["tiles"][0]["filename"])
    arr2 = np.load(wb_out2 / manifest2["tiles"][0]["filename"])
    
    assert np.allclose(arr1, arr2)
    assert manifest1["tiles"][0]["filename"] == manifest2["tiles"][0]["filename"]
    assert manifest1["tiles"][0]["bounds"] == manifest2["tiles"][0]["bounds"]
