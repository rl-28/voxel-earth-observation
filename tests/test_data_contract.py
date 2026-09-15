"""
Tests for the Module 1 spatial-spectral data contract validation.
"""

import pytest
import numpy as np
from module1_ingestion.data_contract import validate_spatial_spectral_tensor
from module1_ingestion.config import EXPECTED_SHAPE, OUTPUT_DTYPE

def test_valid_tensor():
    """Valid tensor (256,256,13), float32, finite values -> passes."""
    tensor = np.zeros(EXPECTED_SHAPE, dtype=OUTPUT_DTYPE)
    # Should not raise an exception
    validate_spatial_spectral_tensor(tensor)

def test_wrong_spatial_height():
    """Wrong spatial height (128,256,13) -> validation fails."""
    tensor = np.zeros((128, 256, 13), dtype=OUTPUT_DTYPE)
    with pytest.raises(ValueError, match="Expected shape"):
        validate_spatial_spectral_tensor(tensor)

def test_wrong_spatial_width():
    """Wrong spatial width (256,128,13) -> validation fails."""
    tensor = np.zeros((256, 128, 13), dtype=OUTPUT_DTYPE)
    with pytest.raises(ValueError, match="Expected shape"):
        validate_spatial_spectral_tensor(tensor)

def test_wrong_band_count_12():
    """Wrong band count (256,256,12) -> validation fails."""
    tensor = np.zeros((256, 256, 12), dtype=OUTPUT_DTYPE)
    with pytest.raises(ValueError, match="Expected shape"):
        validate_spatial_spectral_tensor(tensor)

def test_wrong_band_count_14():
    """Wrong band count (256,256,14) -> validation fails."""
    tensor = np.zeros((256, 256, 14), dtype=OUTPUT_DTYPE)
    with pytest.raises(ValueError, match="Expected shape"):
        validate_spatial_spectral_tensor(tensor)

def test_wrong_axis_order():
    """Wrong axis order (13,256,256) -> validation fails."""
    tensor = np.zeros((13, 256, 256), dtype=OUTPUT_DTYPE)
    with pytest.raises(ValueError, match="Expected shape"):
        validate_spatial_spectral_tensor(tensor)

def test_wrong_dtype():
    """Wrong dtype (float64) -> validation fails."""
    tensor = np.zeros(EXPECTED_SHAPE, dtype=np.float64)
    with pytest.raises(TypeError, match="Expected dtype"):
        validate_spatial_spectral_tensor(tensor)

def test_nan_values():
    """NaN values -> validation fails."""
    tensor = np.zeros(EXPECTED_SHAPE, dtype=OUTPUT_DTYPE)
    tensor[10, 10, 5] = np.nan
    with pytest.raises(ValueError, match="contains non-finite values"):
        validate_spatial_spectral_tensor(tensor)

def test_inf_values():
    """Inf values -> validation fails."""
    tensor = np.zeros(EXPECTED_SHAPE, dtype=OUTPUT_DTYPE)
    tensor[10, 10, 5] = np.inf
    with pytest.raises(ValueError, match="contains non-finite values"):
        validate_spatial_spectral_tensor(tensor)

def test_valid_realistic_tensor():
    """Valid tensor with realistic synthetic floating-point values -> validation passes."""
    # Create random synthetic data representing normalized values (e.g. Z-scores)
    np.random.seed(42)
    tensor = np.random.randn(*EXPECTED_SHAPE).astype(OUTPUT_DTYPE)
    
    # Should not raise an exception
    validate_spatial_spectral_tensor(tensor)
