"""
Spatial-Spectral Data Contract for Module 1.

This module defines the validation logic for the standardized 
spatial-spectral tensors that Module 1 passes to Module 2.

The contract specifies:
- Type: NumPy ndarray
- Shape: (256, 256, 13)
- Axis order: (height, width, spectral_bands)
- Dtype: numpy.float32
- Values: All values must be finite (no NaN or Inf)

Normalization expectation:
- Z-score normalization (Z = (X - mean) / std) is expected to be
  applied consistently across spectral data prior to this validation,
  avoiding generation of NaN/Inf values.
"""

import numpy as np
from .config import EXPECTED_SHAPE, OUTPUT_DTYPE

def validate_spatial_spectral_tensor(tensor):
    """
    Validates that a given tensor strictly conforms to the Module 1 output contract.
    
    Args:
        tensor: The input object to validate.
        
    Raises:
        TypeError: If the tensor is not a NumPy array or has incorrect dtype.
        ValueError: If the shape is incorrect or if the tensor contains non-finite values.
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"Expected a NumPy ndarray, got {type(tensor)}")
        
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3 dimensions (height, width, bands), got {tensor.ndim}")
        
    if tensor.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected shape {EXPECTED_SHAPE}, got {tensor.shape}")
        
    if tensor.dtype != OUTPUT_DTYPE:
        raise TypeError(f"Expected dtype {OUTPUT_DTYPE}, got {tensor.dtype}")
        
    if not np.isfinite(tensor).all():
        raise ValueError("Tensor contains non-finite values (NaN or Inf). All values must be finite.")
