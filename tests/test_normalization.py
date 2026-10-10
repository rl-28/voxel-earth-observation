"""
Tests for Phase 5 Z-score normalization.
"""

import pytest
import numpy as np

from module1_ingestion.normalization import normalize_zscore
from module1_ingestion.config import NORMALIZATION_FILL_VALUE, ZSCORE_DTYPE

def test_normalization_basic_12_band():
    tensor = np.random.rand(256, 256, 12).astype(np.float32)
    norm, mask, means, stds = normalize_zscore(tensor)
    
    assert norm.shape == (256, 256, 12)
    assert norm.dtype == ZSCORE_DTYPE
    assert np.isfinite(norm).all()
    assert means.shape == (12,)
    assert stds.shape == (12,)
    
    # Check zero mean and unit variance for first band
    assert np.isclose(np.mean(norm[:, :, 0]), 0.0, atol=1e-5)
    assert np.isclose(np.std(norm[:, :, 0]), 1.0, atol=1e-5)

def test_normalization_basic_13_band():
    tensor = np.random.rand(256, 256, 13).astype(np.float32)
    norm, mask, means, stds = normalize_zscore(tensor)
    assert norm.shape == (256, 256, 13)
    assert norm.dtype == ZSCORE_DTYPE

def test_normalization_with_valid_mask():
    tensor = np.ones((10, 10, 3), dtype=np.float32)
    # Put random data in the top half
    tensor[:5, :, :] = np.random.rand(5, 10, 3) * 10
    
    valid_mask = np.zeros((10, 10), dtype=bool)
    valid_mask[:5, :] = True
    
    norm, final_mask, means, stds = normalize_zscore(tensor, valid_mask=valid_mask)
    
    # Invalid area should be filled with 0.0
    assert (norm[5:, :, :] == NORMALIZATION_FILL_VALUE).all()
    
    # Valid area should have 0 mean
    assert np.isclose(np.mean(norm[:5, :, 0]), 0.0, atol=1e-5)
    
def test_nan_inf_exclusion():
    tensor = np.random.rand(10, 10, 3).astype(np.float32)
    tensor[0, 0, 0] = np.nan
    tensor[1, 1, 0] = np.inf
    
    norm, final_mask, means, stds = normalize_zscore(tensor)
    
    assert np.isfinite(norm).all()
    assert final_mask[0, 0, 0] == False
    assert final_mask[1, 1, 0] == False
    assert norm[0, 0, 0] == NORMALIZATION_FILL_VALUE

def test_zero_variance_handling():
    tensor = np.full((10, 10, 2), 5.0, dtype=np.float32)
    norm, final_mask, means, stds = normalize_zscore(tensor)
    
    assert stds[0] == 0.0
    assert means[0] == 5.0
    assert np.isfinite(norm).all()
    assert (norm == 0.0).all()

def test_fully_invalid_band_fails():
    tensor = np.full((10, 10, 3), np.nan, dtype=np.float32)
    
    with pytest.raises(ValueError, match="contains no valid pixels"):
        normalize_zscore(tensor)

def test_zero_valued_valid_pixels():
    tensor = np.array([
        [[0.0], [0.0]],
        [[10.0], [10.0]]
    ], dtype=np.float32) # Shape (2, 2, 1)
    
    # The zeros are valid.
    norm, final_mask, means, stds = normalize_zscore(tensor)
    
    assert means[0] == 5.0
    assert stds[0] == 5.0
    # Values should be -1 and 1
    assert np.isclose(norm[0, 0, 0], -1.0)
    assert np.isclose(norm[1, 0, 0], 1.0)
    
def test_single_valid_pixel():
    tensor = np.full((10, 10, 1), np.nan, dtype=np.float32)
    tensor[0, 0, 0] = 42.0
    
    norm, final_mask, means, stds = normalize_zscore(tensor)
    
    assert means[0] == 42.0
    assert stds[0] == 0.0
    assert norm[0, 0, 0] == 0.0
    assert np.isfinite(norm).all()
