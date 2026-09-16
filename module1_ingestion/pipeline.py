import time
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Generator

from .config import (
    WAVELET_NAME, WAVELET_LEVEL, WAVELET_MODE, 
    OUTPUT_DTYPE, SENTINEL_2_L2A_SPECTRAL_BANDS
)
from .tile_processor import process_raster_to_tiles
from .normalization import normalize_zscore
from .wavelet_3d import apply_3d_dwt, Wavelet3DResult
from .albedo_filter import filter_albedo_contamination, AlbedoFilterResult
from .data_contract import validate_spatial_spectral_tensor

logger = logging.getLogger(__name__)

@dataclass
class Module1Result:
    tile_id: str
    input_shape: tuple
    output_shape: tuple
    processed_tensor: np.ndarray
    valid_mask: np.ndarray
    wavelet_result: Optional[Wavelet3DResult]
    glare_mask: Optional[np.ndarray]
    atmospheric_mask: Optional[np.ndarray]
    shadow_mask: Optional[np.ndarray]
    contamination_mask: Optional[np.ndarray]
    diagnostics: Dict[str, Any]
    metadata: Dict[str, Any]

class Module1Pipeline:
    """
    Orchestration layer connecting the completed Module 1 components:
    GeoTIFF -> CRS alignment -> 256x256 tiles -> valid mask -> 
    Z-score normalization -> 3D-DWT -> Albedo/reflection filtering -> validation
    """
    def __init__(
        self,
        enable_wavelet: bool = True,
        enable_albedo_filter: bool = True,
        wavelet_name: str = WAVELET_NAME,
        wavelet_level: int = WAVELET_LEVEL,
        wavelet_mode: str = WAVELET_MODE
    ):
        self.enable_wavelet = enable_wavelet
        self.enable_albedo_filter = enable_albedo_filter
        self.wavelet_name = wavelet_name
        self.wavelet_level = wavelet_level
        self.wavelet_mode = wavelet_mode

    def process_scene(self, band_paths: Dict[str, str], reference_band_key: str = "B04") -> Generator[Module1Result, None, None]:
        """
        Executes the end-to-end Module 1 pipeline incrementally per tile.
        """
        for tile_dict in process_raster_to_tiles(band_paths, reference_band_key):
            yield self.process_tile(tile_dict)
            
    def process_tile(self, tile_dict: Dict[str, Any]) -> Module1Result:
        start_time = time.time()
        
        # 1. Raster ingestion / Tiling (Already done by upstream process_raster_to_tiles)
        raw_tensor = tile_dict["tile_data"]
        row_idx = tile_dict.get("row_idx", 0)
        col_idx = tile_dict.get("col_idx", 0)
        tile_id = f"tile_{row_idx:03d}_{col_idx:03d}"
        
        H, W, C = raw_tensor.shape
        if H != 256 or W != 256:
            raise ValueError(f"Stage 3 Failed: expected spatial dims 256x256, got {H}x{W}")
            
        diagnostics = {
            "input_shape": raw_tensor.shape,
            "input_dtype": str(raw_tensor.dtype),
        }
        
        # 4. Valid-mask extraction
        base_valid_mask = np.isfinite(raw_tensor).all(axis=2)
        
        # 5. Z-score normalization
        norm_start = time.time()
        normalized_tensor, final_valid_mask_3d, means, stds = normalize_zscore(
            raw_tensor, valid_mask=base_valid_mask
        )
        valid_mask = final_valid_mask_3d.all(axis=2)
        
        if not np.isfinite(normalized_tensor).all():
            raise ValueError(f"Stage 5 Failed: Normalization produced non-finite values for {tile_id}")
            
        diagnostics["stage_5_normalization_time"] = time.time() - norm_start
        
        # 6. 3D-DWT processing
        wavelet_res = None
        current_tensor = normalized_tensor
        if self.enable_wavelet:
            wt_start = time.time()
            wavelet_res = apply_3d_dwt(
                current_tensor, 
                wavelet=self.wavelet_name, 
                level=self.wavelet_level, 
                mode=self.wavelet_mode
            )
            # Use spatially preserved reconstruction representation
            current_tensor = wavelet_res.reconstruct_approximation()
            if current_tensor.shape != (256, 256, C):
                raise ValueError(f"Stage 6 Failed: DWT spatial reconstruction altered shape to {current_tensor.shape}")
            if not np.isfinite(current_tensor).all():
                raise ValueError(f"Stage 6 Failed: DWT produced non-finite values")
            diagnostics["stage_6_dwt_time"] = time.time() - wt_start
            
        # 7. Albedo/reflection contamination filtering
        albedo_res = None
        if self.enable_albedo_filter:
            alb_start = time.time()
            albedo_res = filter_albedo_contamination(
                current_tensor,
                input_scale="normalized",
                valid_mask=valid_mask,
                band_names=SENTINEL_2_L2A_SPECTRAL_BANDS
            )
            current_tensor = albedo_res.filtered
            valid_mask = albedo_res.valid_mask
            if current_tensor.shape != (256, 256, C):
                raise ValueError(f"Stage 7 Failed: Albedo filter altered shape to {current_tensor.shape}")
            if not np.isfinite(current_tensor).all():
                raise ValueError(f"Stage 7 Failed: Albedo filter produced non-finite values")
            
            # Incorporate albedo diagnostics
            for k, v in albedo_res.statistics.items():
                diagnostics[f"albedo_{k}"] = v
            diagnostics["stage_7_albedo_time"] = time.time() - alb_start

        # 8. Final validation
        val_start = time.time()
        
        if current_tensor.dtype != OUTPUT_DTYPE:
            raise TypeError(f"Stage 8 Failed: expected dtype {OUTPUT_DTYPE}, got {current_tensor.dtype}")
            
        if not np.isfinite(current_tensor).all():
            raise ValueError(f"Stage 8 Failed: tensor contains non-finite values")
            
        if C == 13:
            # 13-channel project contract
            validate_spatial_spectral_tensor(current_tensor)
            diagnostics["contract_type"] = "13-band final"
        elif C == 12:
            # 12-band real L2A intermediate
            if current_tensor.shape != (256, 256, 12):
                raise ValueError(f"Stage 8 Failed: expected shape (256, 256, 12), got {current_tensor.shape}")
            diagnostics["contract_type"] = "12-band intermediate"
        else:
            raise ValueError(f"Stage 8 Failed: Unsupported number of bands {C}")
            
        diagnostics["stage_8_validation_time"] = time.time() - val_start
        diagnostics["total_time"] = time.time() - start_time
        
        metadata = {
            "transform": tile_dict.get("transform"),
            "crs": tile_dict.get("crs"),
            "bounds": tile_dict.get("bounds"),
            "row_idx": row_idx,
            "col_idx": col_idx,
            "wavelet_enabled": self.enable_wavelet,
            "albedo_filter_enabled": self.enable_albedo_filter
        }
        
        return Module1Result(
            tile_id=tile_id,
            input_shape=raw_tensor.shape,
            output_shape=current_tensor.shape,
            processed_tensor=current_tensor,
            valid_mask=valid_mask,
            wavelet_result=wavelet_res,
            glare_mask=albedo_res.glare_mask if albedo_res else None,
            atmospheric_mask=albedo_res.atmospheric_mask if albedo_res else None,
            shadow_mask=albedo_res.shadow_mask if albedo_res else None,
            contamination_mask=albedo_res.contamination_mask if albedo_res else None,
            diagnostics=diagnostics,
            metadata=metadata
        )
