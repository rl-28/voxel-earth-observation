"""
VOXEL Band Selection Mask Utilities.
Loads, validates, and describes pluggable band selection masks
from JSON, NumPy (.npy), or PyTorch (.pt/.pth) formats.
"""

import json
import os
import sys
import warnings
from typing import List, Optional, Union

import numpy as np
import torch

from config.constants import (
    DEFAULT_SELECTED_BAND_INDICES,
    NUM_BANDS,
    SENTINEL2_BAND_KEYS,
    SENTINEL2_BANDS,
)


def default_band_mask() -> torch.Tensor:
    """
    Returns the default fallback 6-band mask [B02, B03, B04, B08, B11, B12].
    Emits a visible warning to alert the user that a placeholder mask is active.
    """
    warnings.warn(
        "[WARNING] Using default placeholder band mask [B02, B03, B04, B08, B11, B12] "
        "(indices [1, 2, 3, 7, 11, 12]). Provide --mask_path to use optimized bands from Module 2.",
        UserWarning,
        stacklevel=2,
    )
    print(
        "⚠️  [WARNING] Using default placeholder band mask [B02, B03, B04, B08, B11, B12] "
        "(indices [1, 2, 3, 7, 11, 12]). Provide --mask_path to use optimized bands from Module 2.",
        file=sys.stderr,
    )
    mask = np.zeros(NUM_BANDS, dtype=np.float32)
    for idx in DEFAULT_SELECTED_BAND_INDICES:
        mask[idx] = 1.0
    return torch.tensor(mask, dtype=torch.float32)


def validate_band_mask(mask: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
    """
    Validates that a band mask conforms to the Module 3 specification:
      - 1D array/tensor of length exactly 13
      - Strictly binary values (0 or 1)
      - At least 1 band selected
      
    Args:
        mask: Tensor, array, or list representing the mask.
        
    Returns:
        torch.Tensor: Validated float32 tensor of shape (13,).
        
    Raises:
        ValueError: If length != 13, non-binary values, or 0 active bands.
    """
    if isinstance(mask, list):
        mask = np.array(mask, dtype=np.float32)

    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).float()

    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, np.ndarray, or list, got {type(mask).__name__}")

    mask = mask.squeeze().float()

    if mask.dim() != 1:
        raise ValueError(f"Band mask must be 1-dimensional, got shape {tuple(mask.shape)}")

    if mask.numel() != NUM_BANDS:
        raise ValueError(
            f"Band mask length must be exactly {NUM_BANDS} (Sentinel-2 bands), got {mask.numel()}"
        )

    # Check for strictly binary values
    unique_vals = torch.unique(mask)
    for val in unique_vals:
        if not torch.isclose(val, torch.tensor(0.0)) and not torch.isclose(val, torch.tensor(1.0)):
            raise ValueError(
                f"Band mask must contain only binary values (0 or 1), found invalid value {val.item()}"
            )

    active_count = int(torch.sum(mask).item())
    if active_count == 0:
        raise ValueError("Band mask is empty! At least 1 band must be selected (active_count >= 1).")

    return mask


def describe_band_mask(mask: torch.Tensor) -> List[str]:
    """
    Describes the selected spectral bands using canonical Sentinel-2 names.
    
    Args:
        mask: 1D binary tensor of shape (13,).
        
    Returns:
        List[str]: List of selected band descriptions.
    """
    validated = validate_band_mask(mask)
    active_indices = torch.nonzero(validated > 0.5, as_tuple=False).squeeze(-1).tolist()
    if isinstance(active_indices, int):
        active_indices = [active_indices]

    selected_descriptions = [SENTINEL2_BANDS[i] for i in active_indices]
    return selected_descriptions


def load_band_mask(path: Optional[str] = None) -> torch.Tensor:
    """
    Pluggable band mask loader supporting JSON, .npy, and .pt/.pth files.
    
    Args:
        path: Path to mask file. If None or empty, returns default_band_mask().
        
    Returns:
        torch.Tensor: Validated float32 tensor of shape (13,).
    """
    if path is None or str(path).strip() == "":
        return default_band_mask()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Band mask file not found at: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            mask_tensor = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, dict):
            if "selected_indices" in data:
                indices = data["selected_indices"]
                mask = np.zeros(NUM_BANDS, dtype=np.float32)
                for idx in indices:
                    if 0 <= idx < NUM_BANDS:
                        mask[idx] = 1.0
                    else:
                        raise ValueError(f"Selected band index {idx} out of range [0, {NUM_BANDS - 1}]")
                mask_tensor = torch.tensor(mask, dtype=torch.float32)
            elif "mask" in data:
                mask_tensor = torch.tensor(data["mask"], dtype=torch.float32)
            elif "selected_bands" in data:
                band_keys = [b.strip().upper() for b in data["selected_bands"]]
                mask = np.zeros(NUM_BANDS, dtype=np.float32)
                for b_key in band_keys:
                    # Match against B01, B02, etc.
                    matched = False
                    for idx, canonical in enumerate(SENTINEL2_BAND_KEYS):
                        if b_key == canonical or b_key in canonical:
                            mask[idx] = 1.0
                            matched = True
                            break
                    if not matched:
                        raise ValueError(f"Unknown Sentinel-2 band key '{b_key}' in selected_bands")
                mask_tensor = torch.tensor(mask, dtype=torch.float32)
            else:
                raise ValueError(
                    f"JSON mask format unrecognized. Expected 'selected_indices', 'mask', or 'selected_bands'. Keys found: {list(data.keys())}"
                )
        else:
            raise ValueError(f"JSON root must be dict or list, got {type(data).__name__}")

    elif ext == ".npy":
        arr = np.load(path)
        mask_tensor = torch.from_numpy(arr).float()

    elif ext in [".pt", ".pth"]:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "band_mask" in loaded:
            mask_tensor = loaded["band_mask"]
        elif isinstance(loaded, dict) and "mask" in loaded:
            mask_tensor = loaded["mask"]
        elif isinstance(loaded, torch.Tensor):
            mask_tensor = loaded
        else:
            raise ValueError(f"PyTorch file {path} did not contain a valid tensor or mask key")

    else:
        raise ValueError(f"Unsupported mask file extension '{ext}'. Choose from .json, .npy, .pt, or .pth")

    return validate_band_mask(mask_tensor)
