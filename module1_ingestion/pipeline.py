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
        wavelet_mode: str = WAVELET_MODE,
        output_mode: str = "zscore",
        output_13_bands: bool = False,
        spatial_only_wavelet: bool = False,
        keep_water: bool = False
    ):
        self.enable_wavelet = enable_wavelet
        self.enable_albedo_filter = enable_albedo_filter
        self.wavelet_name = wavelet_name
        self.wavelet_level = wavelet_level
        self.wavelet_mode = wavelet_mode
        self.output_mode = output_mode
        self.output_13_bands = output_13_bands
        self.spatial_only_wavelet = spatial_only_wavelet
        self.keep_water = keep_water

    @classmethod
    def for_module3(cls, **overrides) -> "Module1Pipeline":
        """
        Factory method configured specifically for Module 3 model consumption:
        - output_mode="reflectance": physical surface reflectance in [0, 1] (DN / 10000.0)
        - output_13_bands=True: produces (13, 256, 256) channel-first tensor matching data contract
        - spatial_only_wavelet=True: restricts DWT to spatial axes (0, 1) to avoid averaging adjacent spectral bands
        - keep_water=True: preserves water pixels (NDWI > 0, B03 > B08) from being zeroed by cloud shadow filter
        
        Any keyword arguments passed in **overrides will override these default settings.
        """
        config = {
            "output_mode": "reflectance",
            "output_13_bands": True,
            "spatial_only_wavelet": True,
            "keep_water": True,
        }
        config.update(overrides)
        return cls(**config)

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
        
        # 5. Radiometric scaling / Normalization
        norm_start = time.time()
        if self.output_mode == "reflectance":
            # Surface reflectance = DN / 10000.0, clipped to [0.0, 1.0], no Z-score
            normalized_tensor = np.clip(raw_tensor / 10000.0, 0.0, 1.0).astype(OUTPUT_DTYPE)
            valid_mask = base_valid_mask
            diagnostics["normalization_type"] = "reflectance"
        else:
            # Default: Z-score normalization
            normalized_tensor, final_valid_mask_3d, means, stds = normalize_zscore(
                raw_tensor, valid_mask=base_valid_mask
            )
            valid_mask = final_valid_mask_3d.all(axis=2)
            diagnostics["normalization_type"] = "zscore"
        
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
                mode=self.wavelet_mode,
                spatial_only=self.spatial_only_wavelet
            )
            # Use spatially preserved reconstruction representation
            current_tensor = wavelet_res.reconstruct_approximation()
            if self.output_mode == "reflectance":
                current_tensor = np.clip(current_tensor, 0.0, 1.0).astype(OUTPUT_DTYPE)
            if current_tensor.shape != (256, 256, C):
                raise ValueError(f"Stage 6 Failed: DWT spatial reconstruction altered shape to {current_tensor.shape}")
            if not np.isfinite(current_tensor).all():
                raise ValueError(f"Stage 6 Failed: DWT produced non-finite values")
            diagnostics["stage_6_dwt_time"] = time.time() - wt_start
            diagnostics["spatial_only_wavelet"] = self.spatial_only_wavelet
            
        # 7. Albedo/reflection contamination filtering
        albedo_res = None
        if self.enable_albedo_filter:
            alb_start = time.time()
            albedo_scale = "reflectance" if self.output_mode == "reflectance" else "normalized"
            albedo_res = filter_albedo_contamination(
                current_tensor,
                input_scale=albedo_scale,
                valid_mask=valid_mask,
                band_names=SENTINEL_2_L2A_SPECTRAL_BANDS,
                keep_water=self.keep_water
            )
            current_tensor = albedo_res.filtered
            valid_mask = albedo_res.valid_mask
            if self.output_mode == "reflectance":
                current_tensor = np.clip(current_tensor, 0.0, 1.0).astype(OUTPUT_DTYPE)
            if current_tensor.shape != (256, 256, C):
                raise ValueError(f"Stage 7 Failed: Albedo filter altered shape to {current_tensor.shape}")
            if not np.isfinite(current_tensor).all():
                raise ValueError(f"Stage 7 Failed: Albedo filter produced non-finite values")
            
            # Incorporate albedo diagnostics
            for k, v in albedo_res.statistics.items():
                diagnostics[f"albedo_{k}"] = v
            diagnostics["stage_7_albedo_time"] = time.time() - alb_start

        # 7.5 Optional 13-band channel-first formatting
        if self.output_13_bands:
            if C == 12:
                # Insert zero-filled B10 channel at index 10 (between B09 and B11)
                b10_zeros = np.zeros((H, W, 1), dtype=OUTPUT_DTYPE)
                tensor_13_last = np.concatenate([current_tensor[:, :, :10], b10_zeros, current_tensor[:, :, 10:]], axis=2)
            elif C == 13:
                tensor_13_last = current_tensor
            else:
                raise ValueError(f"Stage 7.5 Failed: cannot construct 13-band output from {C} channels")
            
            # Format as channel-first (13, 256, 256)
            final_tensor = np.transpose(tensor_13_last, (2, 0, 1)).astype(OUTPUT_DTYPE)
            diagnostics["channel_layout"] = "channels_first (13, 256, 256)"
            diagnostics["b10_placeholder"] = (C == 12)
        else:
            final_tensor = current_tensor
            diagnostics["channel_layout"] = f"channels_last (256, 256, {C})"

        # 8. Final validation
        val_start = time.time()
        
        if final_tensor.dtype != OUTPUT_DTYPE:
            raise TypeError(f"Stage 8 Failed: expected dtype {OUTPUT_DTYPE}, got {final_tensor.dtype}")
            
        if not np.isfinite(final_tensor).all():
            raise ValueError(f"Stage 8 Failed: tensor contains non-finite values")
            
        if self.output_13_bands:
            if final_tensor.shape != (13, 256, 256):
                raise ValueError(f"Stage 8 Failed: expected shape (13, 256, 256), got {final_tensor.shape}")
            diagnostics["contract_type"] = "13-band channel-first (Module 3 contract)"
        else:
            if C == 13:
                # 13-channel project contract
                validate_spatial_spectral_tensor(final_tensor)
                diagnostics["contract_type"] = "13-band final"
            elif C == 12:
                # 12-band real L2A intermediate
                if final_tensor.shape != (256, 256, 12):
                    raise ValueError(f"Stage 8 Failed: expected shape (256, 256, 12), got {final_tensor.shape}")
                diagnostics["contract_type"] = "12-band intermediate"
            else:
                raise ValueError(f"Stage 8 Failed: Unsupported number of bands {C}")
            
        diagnostics["stage_8_validation_time"] = time.time() - val_start
        diagnostics["total_time"] = time.time() - start_time
        diagnostics["output_mode"] = self.output_mode
        diagnostics["output_13_bands"] = self.output_13_bands
        diagnostics["spatial_only_wavelet"] = self.spatial_only_wavelet
        diagnostics["keep_water"] = self.keep_water
        
        metadata = {
            "transform": tile_dict.get("transform"),
            "crs": tile_dict.get("crs"),
            "bounds": tile_dict.get("bounds"),
            "row_idx": row_idx,
            "col_idx": col_idx,
            "wavelet_enabled": self.enable_wavelet,
            "albedo_filter_enabled": self.enable_albedo_filter,
            "output_mode": self.output_mode,
            "output_13_bands": self.output_13_bands,
            "spatial_only_wavelet": self.spatial_only_wavelet,
            "keep_water": self.keep_water
        }
        
        return Module1Result(
            tile_id=tile_id,
            input_shape=raw_tensor.shape,
            output_shape=final_tensor.shape,
            processed_tensor=final_tensor,
            valid_mask=valid_mask,
            wavelet_result=wavelet_res,
            glare_mask=albedo_res.glare_mask if albedo_res else None,
            atmospheric_mask=albedo_res.atmospheric_mask if albedo_res else None,
            shadow_mask=albedo_res.shadow_mask if albedo_res else None,
            contamination_mask=albedo_res.contamination_mask if albedo_res else None,
            diagnostics=diagnostics,
            metadata=metadata
        )
