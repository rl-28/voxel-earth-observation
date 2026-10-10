import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict

from .config import (
    SENTINEL_2_L2A_SPECTRAL_BANDS,
    FILTER_FILL_VALUE,
    GLARE_Z_THRESHOLD,
    SHADOW_Z_THRESHOLD,
    ATMOSPHERIC_Z_THRESHOLD,
    GLARE_REFLECTANCE_THRESHOLD,
    SHADOW_REFLECTANCE_THRESHOLD,
    ATMOSPHERIC_REFLECTANCE_THRESHOLD,
)

@dataclass
class AlbedoFilterResult:
    filtered: np.ndarray
    valid_mask: np.ndarray
    contamination_mask: np.ndarray
    glare_mask: np.ndarray
    atmospheric_mask: np.ndarray
    shadow_mask: np.ndarray
    statistics: Dict[str, float]

def filter_albedo_contamination(
    tensor: np.ndarray,
    input_scale: str = "normalized",
    valid_mask: Optional[np.ndarray] = None,
    band_names: Optional[List[str]] = None,
) -> AlbedoFilterResult:
    """
    Applies physically motivated reflection and contamination suppression.
    Note: This is residual anomaly suppression, not a replacement for the 
    Sentinel-2 Level-2A atmospheric correction processor.
    
    This function identifies:
    - Likely reflection/glare anomalies (e.g. broadband high response)
    - Likely residual atmospheric anomalies (e.g. isolated high coastal/blue response)
    - Likely cloud-shadow-like anomalies (e.g. unusually low visible and NIR)
    
    It does not calculate absolute physical albedo.
    
    Args:
        tensor: (H, W, C) numpy array, float32
        input_scale: "normalized" or "reflectance"
        valid_mask: Optional boolean mask (H, W), True = valid pixel
        band_names: List of band names matching the tensor's spectral dimension.
                    Defaults to SENTINEL_2_L2A_SPECTRAL_BANDS.
                    
    Returns:
        AlbedoFilterResult containing the filtered tensor and relevant masks.
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError("Input must be a numpy ndarray")
        
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor (H, W, C), got {tensor.ndim}D")
        
    if not np.isfinite(tensor).all():
        raise ValueError("Input tensor contains NaN or Inf values")

    H, W, C = tensor.shape
    band_names = band_names or SENTINEL_2_L2A_SPECTRAL_BANDS
    
    # Safely match bands even if the tensor has an extra unmapped band (e.g., 13th channel)
    b_idx = {name: i for i, name in enumerate(band_names) if i < C}
    
    if valid_mask is None:
        valid_mask = np.ones((H, W), dtype=bool)
    else:
        valid_mask = valid_mask.copy()
        
    if input_scale == "normalized":
        glare_t = GLARE_Z_THRESHOLD
        shadow_t = SHADOW_Z_THRESHOLD
        atm_t = ATMOSPHERIC_Z_THRESHOLD
    elif input_scale == "reflectance":
        glare_t = GLARE_REFLECTANCE_THRESHOLD
        shadow_t = SHADOW_REFLECTANCE_THRESHOLD
        atm_t = ATMOSPHERIC_REFLECTANCE_THRESHOLD
        # If input tensor is in [0, 1] reflectance scale, adjust thresholds by 10000.0
        if tensor.size > 0 and tensor.max() <= 1.0:
            glare_t /= 10000.0
            shadow_t /= 10000.0
            atm_t /= 10000.0
    else:
        raise ValueError(f"Unknown input_scale: {input_scale}")

    glare_mask = np.zeros((H, W), dtype=bool)
    atm_mask = np.zeros((H, W), dtype=bool)
    shadow_mask = np.zeros((H, W), dtype=bool)

    # 1. Atmospheric Anomaly (e.g. unusually high response in Coastal Aerosol/Blue)
    # Sentinel-2 L2A residual haze usually shows up strongly in B01, B02
    if "B01" in b_idx and "B02" in b_idx:
        atm_mask = (tensor[:, :, b_idx["B01"]] > atm_t) & (tensor[:, :, b_idx["B02"]] > atm_t)
        
    # 2. Glare / Reflection Anomaly
    # High response across Vis + NIR + SWIR. 
    # Use B03 (Green), B08 (NIR), B11 (SWIR) to require broadband reflection
    if "B03" in b_idx and "B08" in b_idx and "B11" in b_idx:
        glare_mask = (
            (tensor[:, :, b_idx["B03"]] > glare_t) & 
            (tensor[:, :, b_idx["B08"]] > glare_t) & 
            (tensor[:, :, b_idx["B11"]] > glare_t)
        )
        
    # 3. Cloud-shadow-like Anomaly
    # Extremely dark in both visible and NIR, but not 0 (which might be no-data).
    # Using < threshold ensures we capture shadows, but we require valid pixels.
    if "B04" in b_idx and "B08" in b_idx:
        shadow_mask = (
            (tensor[:, :, b_idx["B04"]] < shadow_t) & 
            (tensor[:, :, b_idx["B08"]] < shadow_t)
        )

    # Mask only valid pixels
    glare_mask &= valid_mask
    atm_mask &= valid_mask
    shadow_mask &= valid_mask
    
    contamination_mask = glare_mask | atm_mask | shadow_mask
    
    # Create filtered output
    filtered = tensor.copy()
    filtered[contamination_mask] = FILTER_FILL_VALUE
    
    new_valid_mask = valid_mask & (~contamination_mask)
    
    total_valid = int(np.sum(valid_mask))
    stats = {
        "total_valid_pixels": total_valid,
        "glare_pixels": int(np.sum(glare_mask)),
        "atmospheric_pixels": int(np.sum(atm_mask)),
        "shadow_pixels": int(np.sum(shadow_mask)),
        "total_contaminated_pixels": int(np.sum(contamination_mask)),
    }
    
    if total_valid > 0:
        stats["glare_pct"] = (stats["glare_pixels"] / total_valid) * 100
        stats["atmospheric_pct"] = (stats["atmospheric_pixels"] / total_valid) * 100
        stats["shadow_pct"] = (stats["shadow_pixels"] / total_valid) * 100
        stats["total_contaminated_pct"] = (stats["total_contaminated_pixels"] / total_valid) * 100
    else:
        stats["glare_pct"] = 0.0
        stats["atmospheric_pct"] = 0.0
        stats["shadow_pct"] = 0.0
        stats["total_contaminated_pct"] = 0.0
        
    return AlbedoFilterResult(
        filtered=filtered,
        valid_mask=new_valid_mask,
        contamination_mask=contamination_mask,
        glare_mask=glare_mask,
        atmospheric_mask=atm_mask,
        shadow_mask=shadow_mask,
        statistics=stats
    )
