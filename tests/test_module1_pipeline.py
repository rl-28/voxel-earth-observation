import pytest
import numpy as np
from module1_ingestion.pipeline import Module1Pipeline, Module1Result
from module1_ingestion.config import OUTPUT_DTYPE

def test_pipeline_12_band_input():
    tensor = np.zeros((256, 256, 12), dtype=OUTPUT_DTYPE)
    pipeline = Module1Pipeline()
    res = pipeline.process_tile({"tile_data": tensor, "row_idx": 0, "col_idx": 0})
    
    assert res.output_shape == (256, 256, 12)
    assert res.processed_tensor.shape == (256, 256, 12)
    assert res.processed_tensor.dtype == OUTPUT_DTYPE
    assert np.isfinite(res.processed_tensor).all()
    assert res.diagnostics["contract_type"] == "12-band intermediate"

def test_pipeline_13_band_input():
    tensor = np.zeros((256, 256, 13), dtype=OUTPUT_DTYPE)
    pipeline = Module1Pipeline()
    res = pipeline.process_tile({"tile_data": tensor, "row_idx": 1, "col_idx": 2})
    
    assert res.output_shape == (256, 256, 13)
    assert res.processed_tensor.shape == (256, 256, 13)
    assert res.processed_tensor.dtype == OUTPUT_DTYPE
    assert np.isfinite(res.processed_tensor).all()
    assert res.diagnostics["contract_type"] == "13-band final"
    assert res.tile_id == "tile_001_002"

def test_pipeline_determinism():
    tensor = np.random.randn(256, 256, 13).astype(OUTPUT_DTYPE)
    pipeline = Module1Pipeline()
    
    res1 = pipeline.process_tile({"tile_data": tensor})
    res2 = pipeline.process_tile({"tile_data": tensor})
    
    assert np.allclose(res1.processed_tensor, res2.processed_tensor)
    assert np.array_equal(res1.valid_mask, res2.valid_mask)
    if res1.contamination_mask is not None:
        assert np.array_equal(res1.contamination_mask, res2.contamination_mask)

def test_pipeline_failure_propagation():
    # Test invalid shape
    tensor = np.zeros((100, 100, 13), dtype=OUTPUT_DTYPE)
    pipeline = Module1Pipeline()
    with pytest.raises(ValueError, match="Stage 3 Failed: expected spatial dims 256x256"):
        pipeline.process_tile({"tile_data": tensor})

def test_pipeline_mask_propagation():
    tensor = np.zeros((256, 256, 13), dtype=OUTPUT_DTYPE)
    # create invalid region
    tensor[0:10, 0:10, :] = np.nan
    pipeline = Module1Pipeline()
    res = pipeline.process_tile({"tile_data": tensor})
    
    # Valid mask should be False in that region
    assert not res.valid_mask[0:10, 0:10].any()
    
    # Glare/shadow/atm masks should not exceed spatial dimensions
    assert res.glare_mask.shape == (256, 256)
    assert res.atmospheric_mask.shape == (256, 256)
    assert res.shadow_mask.shape == (256, 256)
    
    # Check combined mask
    assert np.array_equal(res.contamination_mask, res.glare_mask | res.atmospheric_mask | res.shadow_mask)

def test_pipeline_no_spatial_resize():
    tensor = np.zeros((256, 256, 13), dtype=OUTPUT_DTYPE)
    pipeline = Module1Pipeline()
    res = pipeline.process_tile({"tile_data": tensor})
    assert res.output_shape[0] == 256
    assert res.output_shape[1] == 256
