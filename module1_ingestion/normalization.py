"""
Z-Score Normalization for spatial-spectral tiles.
"""

import numpy as np
from typing import Tuple, Optional

from .config import (
    ZSCORE_DTYPE,
    ZSCORE_STAT_DTYPE,
    ZSCORE_DDOF,
    NORMALIZATION_FILL_VALUE
)

def normalize_zscore(
    tensor: np.ndarray, 
    valid_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies Z-score normalization independently per spectral band across a tile.
    
    Args:
        tensor: 3D NumPy array of shape (H, W, C) representing a spectral tile.
        valid_mask: Optional 2D Boolean array of shape (H, W). True indicates a valid pixel.
                    If None, any finite value in the tensor is treated as valid.
                    
    Returns:
        Tuple containing:
        - normalized_tensor: Shape (H, W, C), float32, all values finite. Invalid pixels filled with NORMALIZATION_FILL_VALUE.
        - final_valid_mask: Shape (H, W, C), Boolean. True for pixels that were used/normalized.
        - means: Shape (C,), float64
        - stds: Shape (C,), float64
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {tensor.shape}")
        
    H, W, C = tensor.shape
    
    normalized = np.full_like(tensor, NORMALIZATION_FILL_VALUE, dtype=ZSCORE_DTYPE)
    means = np.zeros(C, dtype=ZSCORE_STAT_DTYPE)
    stds = np.zeros(C, dtype=ZSCORE_STAT_DTYPE)
    final_valid_mask = np.zeros((H, W, C), dtype=bool)
    
    # Pre-calculate base valid mask
    if valid_mask is not None:
        if valid_mask.shape != (H, W):
            raise ValueError(f"valid_mask shape {valid_mask.shape} does not match spatial dims {(H, W)}")
        base_mask = valid_mask
    else:
        base_mask = np.ones((H, W), dtype=bool)
        
    for c in range(C):
        band_data = tensor[:, :, c]
        
        # Combine user valid_mask with finite check
        band_valid_mask = base_mask & np.isfinite(band_data)
        final_valid_mask[:, :, c] = band_valid_mask
        
        valid_pixels = band_data[band_valid_mask]
        
        if valid_pixels.size == 0:
            raise ValueError(f"Band {c} contains no valid pixels for normalization.")
            
        mean_val = np.mean(valid_pixels, dtype=ZSCORE_STAT_DTYPE)
        std_val = np.std(valid_pixels, ddof=ZSCORE_DDOF, dtype=ZSCORE_STAT_DTYPE)
        
        means[c] = mean_val
        stds[c] = std_val
        
        if np.isclose(std_val, 0.0):
            # Constant band: normalize valid pixels to 0.0
            normalized[band_valid_mask, c] = 0.0
        else:
            # Z-score normalization
            z_scores = (valid_pixels - mean_val) / std_val
            normalized[band_valid_mask, c] = z_scores.astype(ZSCORE_DTYPE)
            
    # Final safety check
    if not np.isfinite(normalized).all():
        raise RuntimeError("Normalization produced non-finite values (NaN/Inf).")
        
    return normalized, final_valid_mask, means, stds
