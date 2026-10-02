"""
module2_optimization/physics_partitioning.py

Physics-guided spectral domain partitioning for Sentinel-2 Level-2A data cubes.
Partitions the 13-band spatial-spectral tensor into physical electromagnetic regimes:
- VISIBLE: B02, B03, B04
- NIR_RED_EDGE: B05, B06, B07, B08, B8A
- SWIR: B11, B12
- AUXILIARY: B01, B09, B10
"""

import logging
from typing import Dict, Any, List
import numpy as np

from .config import (
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    EXPECTED_CHANNELS,
    EXPECTED_DTYPE,
    SPECTRUM_DOMAINS
)

logger = logging.getLogger(__name__)


def validate_input_tensor(tensor: np.ndarray) -> np.ndarray:
    """
    Validates that the input tensor strictly adheres to the Module 2 input specification.

    Args:
        tensor: A 3D NumPy array of shape (256, 256, 13) or 4D array of shape (N, 256, 256, 13).

    Returns:
        np.ndarray: Validated tensor in float32 format.

    Raises:
        TypeError: If tensor is not a NumPy array or cannot be coerced to float32.
        ValueError: If tensor dimensions, spatial resolution, or channel counts do not match contract,
                    or if non-finite (NaN / Inf) values are detected.
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"Expected input to be a numpy.ndarray, got {type(tensor).__name__}")

    if tensor.ndim not in (3, 4):
        raise ValueError(
            f"Expected 3D (256, 256, 13) or 4D (N, 256, 256, 13) tensor, but got {tensor.ndim}D tensor "
            f"with shape {tensor.shape}"
        )

    # Check spatial dimensions and spectral channel count
    spatial_h, spatial_w, channels = tensor.shape[-3], tensor.shape[-2], tensor.shape[-1]

    if spatial_h != EXPECTED_HEIGHT or spatial_w != EXPECTED_WIDTH:
        raise ValueError(
            f"Spatial dimensions must be ({EXPECTED_HEIGHT}, {EXPECTED_WIDTH}), but got ({spatial_h}, {spatial_w})"
        )

    if channels != EXPECTED_CHANNELS:
        raise ValueError(
            f"Spectral channel count must be {EXPECTED_CHANNELS}, but got {channels}"
        )

    if tensor.dtype != EXPECTED_DTYPE:
        logger.debug(f"Casting input tensor dtype from {tensor.dtype} to {EXPECTED_DTYPE}")
        tensor = tensor.astype(EXPECTED_DTYPE)

    if not np.isfinite(tensor).all():
        raise ValueError("Input tensor contains non-finite values (NaN or Inf).")

    return tensor


def partition_spectrum(tensor: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Partitions a 3D (256, 256, 13) or 4D (N, 256, 256, 13) tensor into physical domain sub-tensors.

    Args:
        tensor: Validated spatial-spectral tensor conforming to the 13-channel contract.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary keyed by domain name containing:
            - 'bands': List of band name strings (e.g. ['B02', 'B03', 'B04'])
            - 'indices': List of integer channel indices (e.g. [1, 2, 3])
            - 'data': Sliced sub-tensor along the channel dimension
            - 'description': Domain description string
    """
    validated_tensor = validate_input_tensor(tensor)
    partitioned = {}

    for domain_name, info in SPECTRUM_DOMAINS.items():
        indices: List[int] = info["indices"]
        sub_tensor = validated_tensor[..., indices]
        
        partitioned[domain_name] = {
            "bands": list(info["bands"]),
            "indices": list(indices),
            "data": sub_tensor,
            "description": info["description"]
        }

    return partitioned
