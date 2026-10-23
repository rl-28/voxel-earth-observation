"""
VOXEL: Task-Adaptive Earth Observation & Spectral Intelligence Pipeline
voxel_pipeline.py - Unified Master Orchestrator

Connects:
  Stage 0: Natural Language Query Compiler & OSM Geocoder
  Module 1: Sentinel-2 L2A Ingestion, Tiling, Wavelets & Reflectance Scaling
  Module 2: Physics-Guided Spectral Band Optimization (BWOA) / Task Masks
  Module 3: Hybrid CNN-ViT (FusionNet) Classification, SHAP & FAISS Retrieval

Modes Supported:
  - MODE A (Offline Smoke Test): Synthetic/fixture imagery, mock model inference,
    strict software connectivity verification, explicitly labeled mock outputs.
  - MODE B (Real Satellite Execution): Live geocoding, real Sentinel-2 L2A STAC
    acquisition, radiometric processing, spectral masking, and classification
    strictly gated on genuine trained model checkpoints (reports PARTIALLY_COMPLETE
    with actionable guidance if genuine checkpoint is missing).
"""

import os
import sys
import time
import json
import math
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn

# Ensure Windows UTF-8 stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# VOXEL Geospatial Compatibility Adapter
try:
    from utils.geospatial_raster import ensure_rasterio_shim
    ensure_rasterio_shim()
except ImportError:
    pass

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("voxel_pipeline")

# VOXEL Components
from utils.llm_parser import VoxelStageZeroParser, VoxelExecutionContract
from module1_ingestion.pipeline import Module1Pipeline, Module1Result
from module1_ingestion.stac_downloader import EarthSearchSTACProvider, CopernicusSTACProvider
from module2_optimization.orchestrator import run_module2_optimization, save_band_mask
from module2_optimization.config import (
    TOTAL_BANDS,
    INDEX_TO_BAND,
    BAND_TO_INDEX,
    BAND_NAMES,
    DEFAULT_SCORECARD_WEIGHTS,
)
from models.fusion_net import FusionNet
from explainability.shap_engine import compute_permutation_shap_attributions, plot_band_importance
from explainability.faiss_search import query_similar_terrain
from config.constants import (
    NUM_BANDS,
    NUM_CLASSES,
    EMBEDDING_DIM,
    LAND_COVER_CLASSES,
    TASKS,
    get_task_classes,
    SENTINEL2_BAND_KEYS,
    SENTINEL2_BANDS,
)

# Configurable Default Sentinel-2 STAC Search Window (Fix 3)
DEFAULT_STAC_START_DATE = "2023-01-01"
DEFAULT_STAC_END_DATE = "2024-12-31"
DEFAULT_STAC_CLOUD_COVER = 20.0


@dataclass
class VoxelPipelineResult:
    """
    Structured machine-readable result contract for the VOXEL execution pipeline.
    Suitable for CLI consumption, automated validation, and Streamlit dashboards.
    """
    query: str
    execution_mode: str                         # "offline_mock" | "real_satellite"
    status: str                                 # "COMPLETE" | "PARTIALLY_COMPLETE" | "FAILED"
    is_synthetic: bool                          # True for mock/synthetic data
    is_mock_prediction: bool                    # True when mock weights or synthetic data used
    data_source: str                            # e.g., "Synthetic Deterministic Fixture (Seed: 42)"
    resolved_location: Dict[str, Any]           # {"name": ..., "bbox": [...], "geocoder": ...}
    contract: Dict[str, Any]                    # Stage 0 execution contract dictionary
    task_type: str                              # "waterbody" | "forest_cover" | "general"
    num_classes: int
    class_list: List[str]
    input_tensor_shape: List[int]               # e.g. [13, 256, 256]
    selected_bands: List[str]                   # e.g. ["B01", "B03", "B04", "B06", "B8A", "B11"]
    selected_indices: List[int]                 # e.g. [0, 2, 3, 5, 8, 11]
    band_reduction_pct: float                   # e.g. 53.8%
    mask_provenance: str                        # "precomputed_mask" | "live_bwoa_optimization" | "default_mask"
    mask_path: Optional[str] = None
    candidate_bwoa_bands: Optional[List[str]] = None
    mask_compatibility_note: Optional[str] = None
    model_checkpoint: Optional[str] = None
    model_provenance: str = "unavailable"       # "genuine_trained_model" | "mock_checkpoint" | "unavailable"
    prediction: Optional[Dict[str, Any]] = None # {"predicted_class_id": ..., "predicted_class_name": ..., ...}
    embedding_dim: Optional[int] = None         # 1216
    explainability: Optional[Dict[str, Any]] = None # SHAP band attributions
    similar_terrain: Optional[List[Dict[str, Any]]] = None # FAISS nearest neighbours
    timing_breakdown_ms: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the result contract to a standard dictionary."""
        d = asdict(self)
        def _sanitize(val):
            if isinstance(val, np.ndarray):
                return None
            elif isinstance(val, (np.floating, float)):
                return float(val)
            elif isinstance(val, (np.integer, int)):
                return int(val)
            elif isinstance(val, dict):
                return {
                    k: _sanitize(v) for k, v in val.items()
                    if not (isinstance(v, np.ndarray) or k in ["rgb_image_array", "calibrated_tensor", "raw_tensor", "processed_tensor"])
                }
            elif isinstance(val, list):
                return [_sanitize(v) for v in val if not isinstance(v, np.ndarray)]
            return val
        return _sanitize(d)

    def save_json(self, output_path: Union[str, Path]) -> str:
        """Saves execution result to a machine-readable JSON file."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Execution result saved to: {out_path}")
        return str(out_path)


class VoxelPipeline:
    """
    Unified Master Pipeline Orchestrator connecting all VOXEL system stages:
    Stage 0 (LLM Query Compiler & OSM Geocoding)
      -> Module 1 (Satellite Data Preprocessing & 13-Band Radiometric Formatting)
      -> Module 2 (Spectral Band Optimization / Task Mask Routing)
      -> Module 3 (FusionNet CNN-ViT Classification, SHAP & FAISS Retrieval)
    """

    def __init__(
        self,
        default_mode: str = "offline",
        checkpoint_dir: str = "checkpoints",
        mask_dir: str = "checkpoints/masks",
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.default_mode = default_mode
        self.checkpoint_dir = Path(checkpoint_dir)
        self.mask_dir = Path(mask_dir)
        self.verbose = verbose

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize Stage 0 Query Compiler
        self.stage0_parser = VoxelStageZeroParser()

        # STAC Providers for Real-Data Mode
        self.earth_search_stac = EarthSearchSTACProvider()
        self.copernicus_stac = CopernicusSTACProvider()

    # -------------------------------------------------------------------------
    # Helper: Resolve Downstream Task & Class Specifications
    # -------------------------------------------------------------------------
    def resolve_task_profile(
        self, contract_task: str, user_task_override: Optional[str] = None, user_prompt: Optional[str] = None
    ) -> Tuple[str, Optional[Path], int, List[str], bool]:
        """
        Maps the Stage 0 contract task to a downstream classifier profile,
        precomputed mask path, class names, and binary flag.

        Priority Rules:
        1. An explicit user_task_override takes absolute precedence over prompt or contract.
        2. If user_task_override is None or 'auto', infer the task from user_prompt and contract_task.
        """
        # 1. Explicit User Override Takes Absolute Precedence
        if user_task_override and str(user_task_override).strip().lower() != "auto":
            override_clean = str(user_task_override).strip().lower()
            if any(w in override_clean for w in ["water", "lake", "river", "wetland", "flood"]):
                task_key = "waterbody"
                mask_path = self.mask_dir / "waterbody_mask.json"
                classes = TASKS["waterbody"]["classes"]
                return task_key, mask_path, len(classes), classes, True
            elif any(w in override_clean for w in ["forest", "tree", "canopy", "woodland"]):
                task_key = "forest_cover"
                mask_path = self.mask_dir / "forest_cover_mask.json"
                classes = TASKS["forest_cover"]["classes"]
                return task_key, mask_path, len(classes), classes, True
            elif any(w in override_clean for w in ["general", "landcover", "eurosat"]):
                task_key = "general"
                mask_path = None
                classes = list(LAND_COVER_CLASSES)
                return task_key, mask_path, len(classes), classes, False

        # 2. Inferred from Query Prompt and Contract Task
        text_to_check = f"{user_prompt or ''} {contract_task or ''}".upper()

        if any(w in text_to_check for w in ["WATER", "MOISTURE", "INUNDATION", "FLOOD", "LAKE", "RIVER", "WETLAND"]):
            task_key = "waterbody"
            mask_path = self.mask_dir / "waterbody_mask.json"
            classes = TASKS["waterbody"]["classes"]
            return task_key, mask_path, len(classes), classes, True

        elif any(w in text_to_check for w in ["FOREST", "DEFORESTATION", "CANOPY", "TREE", "WOODLAND"]):
            task_key = "forest_cover"
            mask_path = self.mask_dir / "forest_cover_mask.json"
            classes = TASKS["forest_cover"]["classes"]
            return task_key, mask_path, len(classes), classes, True

        else:
            task_key = "general"
            mask_path = None
            classes = list(LAND_COVER_CLASSES)
            return task_key, mask_path, len(classes), classes, False

    # -------------------------------------------------------------------------
    # Helper: Generate Deterministic Synthetic Tile for Mode A (Offline)
    # -------------------------------------------------------------------------
    def generate_synthetic_tile(
        self, task_key: str, seed: int = 42
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generates a deterministic 13-channel (13, 256, 256) synthetic surface reflectance
        tile scaled in [0.0, 1.0]. Faithfully mimics Sentinel-2 L2A physics by ensuring
        the unavailable B10 channel at index 10 is zero-filled.
        """
        rng = np.random.default_rng(seed)
        H, W = 256, 256

        # Base reflectance across 13 bands in [0.05, 0.45]
        tile = rng.uniform(0.05, 0.45, size=(13, H, W)).astype(np.float32)

        if task_key == "waterbody":
            # Synthesize realistic water spectral signature in central region
            # High Blue (B02/idx 1), Green (B03/idx 2); Very Low NIR (B08/idx 7) and SWIR (B11/idx 11, B12/idx 12)
            y, x = np.ogrid[:H, :W]
            dist_center = np.sqrt((x - 128) ** 2 + (y - 128) ** 2)
            water_mask = dist_center < 75.0

            tile[1, water_mask] = rng.uniform(0.15, 0.25, size=np.sum(water_mask))  # B02 Blue
            tile[2, water_mask] = rng.uniform(0.20, 0.35, size=np.sum(water_mask))  # B03 Green
            tile[3, water_mask] = rng.uniform(0.02, 0.08, size=np.sum(water_mask))  # B04 Red
            tile[7, water_mask] = rng.uniform(0.005, 0.03, size=np.sum(water_mask)) # B08 NIR
            tile[11, water_mask] = rng.uniform(0.001, 0.02, size=np.sum(water_mask))# B11 SWIR1
            tile[12, water_mask] = rng.uniform(0.001, 0.01, size=np.sum(water_mask))# B12 SWIR2

        elif task_key == "forest_cover":
            # Synthesize dense vegetation spectral signature
            # Moderate Green, High Red-Edge / NIR (idx 7), Low SWIR
            y, x = np.ogrid[:H, :W]
            forest_mask = (y < 180) & (x < 200)
            tile[3, forest_mask] = rng.uniform(0.03, 0.06, size=np.sum(forest_mask)) # B04 Red (chlorophyll absorption)
            tile[7, forest_mask] = rng.uniform(0.45, 0.70, size=np.sum(forest_mask)) # B08 NIR (canopy scatter)

        # L2A physics invariant: Cirrus Band 10 is excluded in L2A -> placeholder zero channel
        tile[10, :, :] = 0.0

        metadata = {
            "synthetic_seed": seed,
            "provenance": "synthetic_mock_fixture",
            "tile_shape": list(tile.shape),
            "dtype": str(tile.dtype),
            "min_val": float(tile.min()),
            "max_val": float(tile.max()),
            "b10_placeholder": True,
        }
        return tile, metadata

    # -------------------------------------------------------------------------
    # Helper: Acquire Real Satellite Imagery via STAC & Process via Module 1
    # -------------------------------------------------------------------------
    def acquire_and_process_real_satellite_data(
        self,
        bbox: List[float],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cloud_cover: float = DEFAULT_STAC_CLOUD_COVER,
        allow_fallback: bool = True,
        cache_dir: str = "data/cache/stac",
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any], List[str]]:
        """
        Acquires real Sentinel-2 Level-2A imagery for the requested BBOX using
        public AWS Earth Search STAC, downloads bands, and executes Module 1 processing.

        Fix 3:
        1. Accepts user-specified acquisition date range (start_date, end_date).
        2. Where no date is specified, uses configurable default (DEFAULT_STAC_START_DATE/END_DATE).
        3. Filters candidate items to ensure spectral assets are present and sorts by lowest cloud cover.
        4. Provides controlled fallback search (relaxing cloud cover / expanding date window) when
           allow_fallback is True, recording transparent notices in warnings.
        5. Preserves full real STAC provenance, item IDs, and acquisition timestamps.
        """
        errors = []
        stac_warnings = []
        min_lon, min_lat, max_lon, max_lat = bbox
        geojson_aoi = {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]],
        }

        actual_start_date = start_date.strip() if (start_date and str(start_date).strip()) else DEFAULT_STAC_START_DATE
        actual_end_date = end_date.strip() if (end_date and str(end_date).strip()) else DEFAULT_STAC_END_DATE
        actual_cloud_cover = float(cloud_cover) if cloud_cover is not None else DEFAULT_STAC_CLOUD_COVER
        relaxation_info = None

        def execute_search(s_date: str, e_date: str, cc: float) -> List[Any]:
            found = self.earth_search_stac.search_sentinel2_l2a(
                aoi=geojson_aoi,
                start_date=s_date,
                end_date=e_date,
                cloud_cover=cc,
                max_items=15,
            )
            valid_items = []
            for it in found:
                assets = it.assets
                # Check for required optical assets
                has_assets = all(
                    self.earth_search_stac.get_band_asset_key(b) in assets
                    for b in ["B02", "B03", "B04", "B08"]
                )
                if has_assets:
                    valid_items.append(it)
            # Sort ascending by cloud cover to select best scene
            valid_items.sort(key=lambda it: float(it.properties.get("eo:cloud_cover", 100.0)))
            return valid_items

        logger.info(
            f"Searching Sentinel-2 L2A STAC scenes for BBOX {bbox} "
            f"between {actual_start_date} and {actual_end_date} (cloud_cover <= {actual_cloud_cover}%)..."
        )
        try:
            items = execute_search(actual_start_date, actual_end_date, actual_cloud_cover)
        except Exception as e:
            errors.append(f"STAC search request failed: {e}")
            return None, {}, errors

        # Controlled Fallback Relaxation (visible to user, never silent)
        if not items and allow_fallback:
            logger.info("Initial STAC search returned 0 scenes. Initiating controlled search relaxation...")
            # Fallback Stage 1: Relax cloud cover to 40% (if original was < 40%)
            if actual_cloud_cover < 40.0:
                try:
                    items = execute_search(actual_start_date, actual_end_date, 40.0)
                    if items:
                        notice = (
                            f"Notice: Initial search criteria (cloud_cover <= {actual_cloud_cover}%) yielded 0 scenes. "
                            f"Controlled fallback relaxed cloud cover threshold to <= 40.0% ({actual_start_date} to {actual_end_date})."
                        )
                        stac_warnings.append(notice)
                        relaxation_info = {"relaxed_cloud_cover": 40.0, "stage": 1}
                except Exception as e:
                    logger.debug(f"Fallback stage 1 search error: {e}")

            # Fallback Stage 2: Expand date window to 2022-01-01 with cloud cover <= 40%
            if not items:
                try:
                    expanded_start = "2022-01-01"
                    cc_target = max(actual_cloud_cover, 40.0)
                    items = execute_search(expanded_start, actual_end_date, cc_target)
                    if items:
                        notice = (
                            f"Notice: Initial search ({actual_start_date} to {actual_end_date}, cloud_cover <= {actual_cloud_cover}%) "
                            f"yielded 0 scenes. Controlled fallback expanded date window to {expanded_start} to {actual_end_date} "
                            f"with cloud cover <= {cc_target}%."
                        )
                        stac_warnings.append(notice)
                        relaxation_info = {"relaxed_start_date": expanded_start, "relaxed_cloud_cover": cc_target, "stage": 2}
                except Exception as e:
                    logger.debug(f"Fallback stage 2 search error: {e}")

            # Fallback Stage 3: Relax cloud cover to 75% across expanded window
            if not items:
                try:
                    expanded_start = "2022-01-01"
                    items = execute_search(expanded_start, actual_end_date, 75.0)
                    if items:
                        notice = (
                            f"Notice: Controlled fallback relaxed cloud cover threshold to <= 75.0% "
                            f"across {expanded_start} to {actual_end_date} to find available Sentinel-2 imagery."
                        )
                        stac_warnings.append(notice)
                        relaxation_info = {"relaxed_start_date": expanded_start, "relaxed_cloud_cover": 75.0, "stage": 3}
                except Exception as e:
                    logger.debug(f"Fallback stage 3 search error: {e}")

        if not items:
            errors.append(
                f"No Sentinel-2 L2A scenes found meeting cloud cover <= {actual_cloud_cover}% "
                f"for BBOX {bbox} between {actual_start_date} and {actual_end_date}."
                + (" Fallback relaxation also found no scenes." if allow_fallback else " (Fallback relaxation disabled).")
            )
            return None, {}, errors

        # Item with lowest cloud cover is selected
        selected_item = items[0]
        logger.info(
            f"Selected STAC Item: {selected_item.id} "
            f"(Cloud Cover: {selected_item.properties.get('eo:cloud_cover', 'N/A')}%, "
            f"Datetime: {selected_item.datetime.isoformat() if selected_item.datetime else 'N/A'})"
        )

        # Isolate cache directory per unique AOI to prevent cross-location scene collisions
        bbox_slug = f"{min_lon:.4f}_{min_lat:.4f}_{max_lon:.4f}_{max_lat:.4f}".replace('.', 'p').replace('-', 'm')
        item_cache_dir = os.path.join(cache_dir, f"{selected_item.id}_{bbox_slug}")
        legacy_cache_dir = os.path.join(cache_dir, selected_item.id)

        target_item_dir = item_cache_dir
        if not os.path.exists(item_cache_dir) and os.path.exists(legacy_cache_dir):
            target_item_dir = legacy_cache_dir

        is_already_cached = (
            os.path.exists(target_item_dir)
            and any(
                f.endswith(".tif") and os.path.getsize(os.path.join(target_item_dir, f)) > 1000
                for f in os.listdir(target_item_dir)
            )
        )

        try:
            band_paths = self.earth_search_stac.download_bands(
                item=selected_item,
                target_dir=target_item_dir,
                bbox=bbox,
                overwrite=False,
            )
        except Exception as e:
            errors.append(f"Failed to download Sentinel-2 spectral bands: {e}")
            return None, {}, errors

        # Process through Module 1 Pipeline for Module 3 specification (reflectance [0, 1], 13 bands)
        logger.info("Executing Module 1 preprocessing on downloaded scene...")
        mod1_pipeline = Module1Pipeline.for_module3()
        try:
            results = list(mod1_pipeline.process_scene(band_paths))
            if not results:
                errors.append("Module 1 pipeline produced 0 valid 256x256 tiles from downloaded scene.")
                return None, {}, errors

            first_tile_res: Module1Result = results[0]
            processed_tensor = first_tile_res.processed_tensor  # (13, 256, 256)

            from rasterio.warp import transform_bounds
            tiles_data = []
            for r in results:
                raw_b = r.metadata.get("bounds") if hasattr(r, "metadata") and isinstance(r.metadata, dict) else (r.diagnostics.get("bounds") if hasattr(r, "diagnostics") and isinstance(r.diagnostics, dict) else None)
                crs = r.metadata.get("crs") if hasattr(r, "metadata") and isinstance(r.metadata, dict) else None
                wgs84_b = None
                if raw_b and crs:
                    try:
                        wb = transform_bounds(crs, "EPSG:4326", *raw_b)
                        wgs84_b = [round(wb[0], 4), round(wb[1], 4), round(wb[2], 4), round(wb[3], 4)]
                    except Exception:
                        wgs84_b = list(raw_b)
                elif raw_b:
                    wgs84_b = list(raw_b)

                tiles_data.append({
                    "tile_id": r.tile_id,
                    "processed_tensor": r.processed_tensor,
                    "diagnostics": r.diagnostics,
                    "bounds": wgs84_b,
                    "raw_bounds": raw_b,
                    "crs": crs,
                })

            metadata = {
                "stac_item_id": selected_item.id,
                "scene_id": selected_item.id,
                "selected_item": selected_item.id,
                "stac_provider": "AWS Earth Search (Element84)",
                "collection": selected_item.collection_id,
                "datetime": selected_item.datetime.isoformat() if selected_item.datetime else None,
                "cloud_cover": selected_item.properties.get("eo:cloud_cover"),
                "tile_id": first_tile_res.tile_id,
                "diagnostics": first_tile_res.diagnostics,
                "total_scene_tiles": len(results),
                "tiles_data": tiles_data,
                "is_cached": is_already_cached,
                "cache_status": "Cached on Disk (Zero Network IO)" if is_already_cached else "Newly Acquired via STAC COG",
                "search_start_date": actual_start_date,
                "search_end_date": actual_end_date,
                "search_cloud_threshold": actual_cloud_cover,
                "fallback_applied": relaxation_info is not None,
                "relaxation_info": relaxation_info,
                "warnings": stac_warnings,
            }
            return processed_tensor, metadata, []

        except Exception as e:
            errors.append(f"Module 1 tile processing failed: {e}")
            return None, {}, errors

    # -------------------------------------------------------------------------
    # Helper: Resolve Spectral Band Mask (Strategy 1: Precomputed vs Strategy 2: BWOA)
    # -------------------------------------------------------------------------
    def resolve_band_mask(
        self,
        tensor_13_bands: np.ndarray,
        task_key: str,
        precomputed_mask_path: Optional[Path] = None,
        optimize: bool = False,
        target_band_count: Optional[int] = None,
        scorecard_weights: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, List[int], List[str], str, float, Optional[str], Dict[str, Any]]:
        """
        Resolves spectral band selection mask via either:
        1. Precomputed task-specific mask (e.g. checkpoints/masks/waterbody_mask.json)
        2. Live Binary Whale Optimization Algorithm (BWOA) search on the prepared tensor.
        """
        # Ensure tensor is channel-last for Module 2 if running BWOA
        if tensor_13_bands.shape[0] == 13 and tensor_13_bands.shape[1] == 256:
            tensor_channel_last = np.transpose(tensor_13_bands, (1, 2, 0))
        else:
            tensor_channel_last = tensor_13_bands

        # Strategy 2: Live BWOA Metaheuristic Search requested
        if optimize:
            logger.info(f"Executing Live Module 2 BWOA Band Optimization (target_bands={target_band_count})...")
            target_count = target_band_count or (4 if task_key == "waterbody" else 6)
            weights = scorecard_weights or DEFAULT_SCORECARD_WEIGHTS

            best_mask_arr, selected_indices, opt_metrics = run_module2_optimization(
                input_tensor=tensor_channel_last,
                target_count=target_count,
                weights=weights,
                enforce_domain_balance=True,
                exclude_placeholder_b10=True,
                seed=seed,
            )
            mask_tensor = torch.tensor(best_mask_arr, dtype=torch.float32)
            selected_bands = [INDEX_TO_BAND[idx] for idx in selected_indices if idx in INDEX_TO_BAND]
            data_reduction = float(((TOTAL_BANDS - len(selected_indices)) / TOTAL_BANDS) * 100.0)

            return (
                mask_tensor,
                selected_indices,
                selected_bands,
                "live_bwoa_optimization",
                data_reduction,
                None,
                opt_metrics,
            )

        # Strategy 1: Load precomputed task-specific mask
        if precomputed_mask_path and precomputed_mask_path.exists():
            logger.info(f"Loading validated precomputed task mask from: {precomputed_mask_path}")
            with open(precomputed_mask_path, "r", encoding="utf-8") as f:
                mask_data = json.load(f)

            selected_indices = sorted(mask_data.get("selected_indices", [1, 2, 3, 7, 11, 12]))
            # Strict safety: Enforce refusal of placeholder band B10 (index 10)
            if 10 in selected_indices:
                logger.warning("Placeholder band index 10 found in precomputed mask; removing index 10.")
                selected_indices = [idx for idx in selected_indices if idx != 10]

            mask_arr = np.zeros(TOTAL_BANDS, dtype=np.float32)
            mask_arr[selected_indices] = 1.0
            mask_tensor = torch.tensor(mask_arr, dtype=torch.float32)
            selected_bands = [INDEX_TO_BAND[idx] for idx in selected_indices if idx in INDEX_TO_BAND]
            data_reduction = float(((TOTAL_BANDS - len(selected_indices)) / TOTAL_BANDS) * 100.0)

            diagnostics = {
                "mask_file": str(precomputed_mask_path),
                "method": mask_data.get("method", "BWOA"),
                "dataset": mask_data.get("dataset", "Sentinel-2 L2A"),
            }
            return (
                mask_tensor,
                selected_indices,
                selected_bands,
                "precomputed_mask",
                data_reduction,
                str(precomputed_mask_path),
                diagnostics,
            )

        # Fallback: Default canonical 6-band mask
        logger.info("Using default canonical 6-band selection mask (B02, B03, B04, B08, B11, B12).")
        default_indices = [1, 2, 3, 7, 11, 12]
        mask_arr = np.zeros(TOTAL_BANDS, dtype=np.float32)
        mask_arr[default_indices] = 1.0
        mask_tensor = torch.tensor(mask_arr, dtype=torch.float32)
        selected_bands = [INDEX_TO_BAND[idx] for idx in default_indices]
        data_reduction = float(((TOTAL_BANDS - len(default_indices)) / TOTAL_BANDS) * 100.0)

        return (
            mask_tensor,
            default_indices,
            selected_bands,
            "default_mask",
            data_reduction,
            None,
            {"default_indices": default_indices},
        )

    # -------------------------------------------------------------------------
    # Helper: Checkpoint Inspection & Model Loading
    # -------------------------------------------------------------------------
    def inspect_and_load_model(
        self,
        checkpoint_path: Optional[Union[str, Path]],
        num_classes: int,
        selected_channels: int,
        is_real_mode: bool,
        task_key: Optional[str] = None,
        selected_indices: Optional[List[int]] = None,
    ) -> Tuple[Optional[FusionNet], str, Optional[str], List[str]]:
        """
        Inspects model checkpoint provenance and compatibility.
        In Real Mode, strictly refuses mock checkpoints for final classification,
        and verifies that active spectral channels match checkpoint training configuration.
        """
        warnings = []
        if checkpoint_path is None:
            # Check for real task-specific checkpoint first
            if task_key:
                real_ckpt = self.checkpoint_dir / f"real_{task_key}" / "best_fusion_net.pth"
                if real_ckpt.exists():
                    checkpoint_path = real_ckpt
            # Check for generic task-specific checkpoint next
            if checkpoint_path is None and task_key:
                task_ckpt = self.checkpoint_dir / task_key / "best_fusion_net.pth"
                if task_ckpt.exists():
                    checkpoint_path = task_ckpt
            if checkpoint_path is None:
                default_ckpt = self.checkpoint_dir / "best_fusion_net.pth"
                if default_ckpt.exists():
                    checkpoint_path = default_ckpt
                else:
                    return None, "unavailable", None, ["No model checkpoint provided and default checkpoint not found."]

        ckpt_path_obj = Path(checkpoint_path)
        if not ckpt_path_obj.exists():
            return None, "unavailable", str(ckpt_path_obj), [f"Model checkpoint file not found at: {ckpt_path_obj}"]

        try:
            ckpt = torch.load(ckpt_path_obj, map_location=self.device, weights_only=False)
        except Exception as e:
            return None, "unavailable", str(ckpt_path_obj), [f"Failed to load checkpoint file: {e}"]

        config_args = ckpt.get("config_args", {})
        metrics = ckpt.get("metrics", {})
        is_mock_ckpt = bool(config_args.get("mock") or metrics.get("is_mock"))
        ckpt_task = ckpt.get("task", None)
        ckpt_num_classes = ckpt.get("num_classes", len(ckpt.get("class_list", [])) or 10)
        ckpt_channels = ckpt.get("selected_channels", 6)

        # Provenance classification
        provenance = "mock_checkpoint" if is_mock_ckpt else "genuine_trained_model"

        # Strict Gate for Real Satellite Execution Mode
        if is_real_mode and is_mock_ckpt:
            logger.warning(
                f"Checkpoint at {ckpt_path_obj} is identified as a MOCK/SYNTHETIC model. "
                "In Real Satellite Mode, inference predictions are withheld to preserve scientific integrity."
            )
            return None, "mock_checkpoint_withheld_for_real_mode", str(ckpt_path_obj), [
                "Mock/synthetic checkpoint detected. Classification prediction withheld during real satellite execution.",
                "To execute full inference, train a genuine model on real data or provide a real trained checkpoint.",
            ]

        # Verify spectral channel count compatibility in Real Satellite Mode
        if ckpt_channels != selected_channels:
            if is_real_mode and not is_mock_ckpt:
                logger.warning(
                    f"Spectral channel count mismatch: Checkpoint expects {ckpt_channels} channels but current mask has {selected_channels}. "
                    "Classification inference withheld to prevent running uncalibrated weights."
                )
                return None, "channel_count_mismatch_gated", str(ckpt_path_obj), [
                    f"Selected spectral mask has {selected_channels} channels, but calibrated model '{task_key or 'default'}' was trained on {ckpt_channels} channels.",
                    "Classification inference is safely gated to prevent running untrained weights on real satellite data. To execute classification, use the calibrated task profile.",
                ]
            else:
                warnings.append(
                    f"Checkpoint trained on {ckpt_channels} channels but current mask has {selected_channels} channels. "
                    "Channel-specific weights re-initialized for execution compatibility."
                )

        # Verify spectral band identity compatibility in Real Satellite Mode
        ckpt_band_mask = ckpt.get("band_mask")
        if is_real_mode and not is_mock_ckpt and ckpt_band_mask is not None and selected_indices is not None:
            ckpt_idx_list = [i for i, v in enumerate(ckpt_band_mask) if (v == 1 if isinstance(v, (int, float)) else bool(v))]
            if sorted(ckpt_idx_list) != sorted(selected_indices):
                logger.warning(
                    f"Spectral channel identity mismatch: candidate mask ({selected_indices}) differs from checkpoint bands ({ckpt_idx_list})."
                )
                return None, "band_selection_mismatch_gated", str(ckpt_path_obj), [
                    f"Candidate mask bands {selected_indices} differ from calibrated model training bands {ckpt_idx_list}.",
                    "Classification inference is safely gated to prevent passing out-of-distribution spectral combinations into real model weights. To execute classification, use the calibrated task profile.",
                ]

        # Check architecture head compatibility
        active_classes = ckpt_num_classes if is_mock_ckpt else num_classes

        # Instantiate FusionNet
        model = FusionNet(
            num_classes=active_classes,
            total_bands=TOTAL_BANDS,
            selected_channels=selected_channels,
            cnn_feature_dim=1024,
            vit_feature_dim=192,
        ).to(self.device)

        # Load weights if channel dimensions match
        if ckpt_channels == selected_channels:
            try:
                model.load_state_dict(ckpt["model_state_dict"], strict=(ckpt_num_classes == active_classes))
            except Exception as e:
                logger.warning(f"Partial state dict mismatch when loading checkpoint: {e}")
                warnings.append(f"Checkpoint loaded with partial weight alignment: {e}")

        model.eval()
        return model, provenance, str(ckpt_path_obj), warnings

    # -------------------------------------------------------------------------
    # Master Execution Entry Point
    # -------------------------------------------------------------------------
    def run(
        self,
        query: str,
        mode: Optional[str] = None,
        task_override: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        optimize: bool = False,
        target_band_count: Optional[int] = None,
        run_shap: bool = False,
        run_faiss: bool = True,
        output_json: Optional[str] = None,
        synthetic_seed: int = 42,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cloud_cover: float = DEFAULT_STAC_CLOUD_COVER,
        allow_fallback: bool = True,
        bbox: Optional[List[float]] = None,
        **kwargs,
    ) -> VoxelPipelineResult:
        """
        Executes the unified end-to-end VOXEL pipeline for a natural language query.
        """
        # Support common aliases and dynamic STAC parameters
        if "explain_shap" in kwargs:
            run_shap = kwargs["explain_shap"]
        if "optimize_bands" in kwargs:
            optimize = kwargs["optimize_bands"]
        if "start_date" in kwargs and start_date is None:
            start_date = kwargs["start_date"]
        if "end_date" in kwargs and end_date is None:
            end_date = kwargs["end_date"]
        if "cloud_cover" in kwargs and cloud_cover == DEFAULT_STAC_CLOUD_COVER:
            cloud_cover = kwargs["cloud_cover"]
        if "allow_fallback" in kwargs:
            allow_fallback = kwargs["allow_fallback"]

        start_total = time.perf_counter()
        timing_breakdown: Dict[str, float] = {}
        diagnostics: Dict[str, Any] = {}
        errors: List[str] = []
        warnings: List[str] = []

        execution_mode = (mode or self.default_mode).lower()
        if execution_mode not in ["offline", "offline_mock", "real", "real_satellite"]:
            raise ValueError(f"Unsupported execution mode '{execution_mode}'. Choose 'offline' or 'real'.")

        is_real_mode = execution_mode in ["real", "real_satellite"]
        display_mode = "real_satellite" if is_real_mode else "offline_mock"

        if self.verbose:
            print("\n" + "=" * 78)
            print(f"  VOXEL EARTH OBSERVATION PIPELINE — EXECUTION START")
            print(f"  Mode: {display_mode.upper()} | Query: \"{query}\"")
            print("=" * 78)

        # ---------------------------------------------------------------------
        # STAGE 0: Natural Language Query Compilation & Geocoding
        # ---------------------------------------------------------------------
        # Explicit Bounding Box Override (e.g. from Interactive Map AOI selector)
        explicit_bbox = bbox or kwargs.get("bbox") or kwargs.get("coordinates")
        validated_bbox = None
        if explicit_bbox is not None:
            if not isinstance(explicit_bbox, (list, tuple)) or len(explicit_bbox) != 4:
                raise ValueError(f"Invalid BBOX specification: {explicit_bbox}. Expected [min_lon, min_lat, max_lon, max_lat].")
            min_lon, min_lat, max_lon, max_lat = [float(x) for x in explicit_bbox]
            if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0 and -90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
                raise ValueError(f"Coordinates out of physical bounds: [{min_lon}, {min_lat}, {max_lon}, {max_lat}]")
            if min_lon >= max_lon or min_lat >= max_lat:
                raise ValueError(f"Invalid bounding box geometry: min must be strictly less than max. Got [{min_lon}, {min_lat}, {max_lon}, {max_lat}]")

            # Explicit AOI-size limit to protect CPU/memory (Requirement 3)
            span_lon = max_lon - min_lon
            span_lat = max_lat - min_lat
            if span_lon > 0.08 or span_lat > 0.08:
                raise ValueError(
                    f"Selected AOI span ({span_lon*111.3:.1f} km × {span_lat*111.3:.1f} km) exceeds maximum supported single-scene limit (8.0 km × 8.0 km / 0.08°). "
                    "Please draw a smaller rectangular area or select a point to protect system resources."
                )
            validated_bbox = [round(min_lon, 6), round(min_lat, 6), round(max_lon, 6), round(max_lat, 6)]

        t0_start = time.perf_counter()
        logger.info("[STAGE 0] Compiling natural language query and resolving geographic location...")
        contract: VoxelExecutionContract = self.stage0_parser.parse(query, bbox=validated_bbox)
        timing_breakdown["stage_0_query_compilation_ms"] = (time.perf_counter() - t0_start) * 1000.0

        if validated_bbox is not None:
            contract.bbox = validated_bbox
            contract.location_name = f"Custom AOI [{validated_bbox[0]:.4f}, {validated_bbox[1]:.4f}, {validated_bbox[2]:.4f}, {validated_bbox[3]:.4f}]"
            contract.geocoder = "Interactive Map / Explicit BBOX Input"

        resolved_location = {
            "name": contract.location_name,
            "bbox": contract.bbox,
            "geocoder": contract.geocoder,
            "llm_provider": contract.llm_provider,
        }

        task_key, precomputed_mask_path, num_classes, class_list, is_binary = self.resolve_task_profile(
            contract.task_type, user_task_override=task_override, user_prompt=query
        )
        diagnostics["task_key"] = task_key
        diagnostics["is_binary"] = is_binary

        # Task Selection Override Notice (Honors explicit selection, warns of query text mismatch)
        if task_override and str(task_override).strip().lower() != "auto":
            q_upper_check = query.upper()
            if task_key == "forest_cover" and any(w in q_upper_check for w in ["WATER", "LAKE", "RIVER", "WETLAND", "FLOOD"]):
                mismatch_note = f"Task Selection Override: Query mentions water bodies, but explicit profile '{task_override}' was selected. System strictly executed the Forest Cover CNN–ViT checkpoint."
                warnings.append(mismatch_note)
                diagnostics["task_mismatch_note"] = mismatch_note
            elif task_key == "waterbody" and any(w in q_upper_check for w in ["FOREST", "TREE", "CANOPY", "WOODLAND"]):
                mismatch_note = f"Task Selection Override: Query mentions forest canopy, but explicit profile '{task_override}' was selected. System strictly executed the Waterbody CNN–ViT checkpoint."
                warnings.append(mismatch_note)
                diagnostics["task_mismatch_note"] = mismatch_note

        # Task Scope and Scientific Qualification
        qualification_notes = []
        q_upper = query.upper()
        if any(w in q_upper for w in ["LEVEL", "DEPTH", "VOLUME", "CAPACITY", "FLUCTUATION"]):
            qualification_notes.append(
                "Scientific Scope Notice: Pipeline performs whole-tile land-cover classification (Water vs Non-water at 256×256 px footprint). It does NOT quantify temporal water level/depth fluctuations or volumetric reservoir capacity."
            )
        if any(w in q_upper for w in ["CANOPY", "DENSITY", "TREE COVER PERCENTAGE", "TREE COUNT"]):
            qualification_notes.append(
                "Scientific Scope Notice: Pipeline performs whole-tile binary forest-cover presence classification (Forest vs Non-forest). It does NOT produce continuous canopy density percentages or pixel-level segmentation."
            )
        if any(w in q_upper for w in ["CROP", "AGRICULTURE", "HARVEST", "YIELD"]):
            qualification_notes.append(
                "Scientific Scope Notice: Crop health task profile routes to multi-spectral reference models. Dedicated real-data Indian models currently exist for Waterbody and Forest Cover tasks."
            )
        if qualification_notes:
            diagnostics["task_qualifications"] = qualification_notes
            warnings.extend(qualification_notes)

        if self.verbose:
            print(f"  [+] Stage 0 Resolved Location : {contract.location_name}")
            print(f"  [+] Stage 0 Bounding Box     : {contract.bbox}")
            print(f"  [+] Stage 0 Detected Task    : {task_key.upper()} ({num_classes} classes)")
            print(f"  [+] Stage 0 LLM / Geocoder   : {contract.llm_provider} / {contract.geocoder}")

        # ---------------------------------------------------------------------
        # MODULE 1: Satellite Data Acquisition & Radiometric Preprocessing
        # ---------------------------------------------------------------------
        t1_start = time.perf_counter()
        logger.info(f"[MODULE 1] Acquiring and preprocessing satellite tensor (Mode: {display_mode})...")

        if is_real_mode:
            raw_tensor, mod1_meta, mod1_errors = self.acquire_and_process_real_satellite_data(
                bbox=contract.bbox,
                start_date=start_date,
                end_date=end_date,
                cloud_cover=cloud_cover,
                allow_fallback=allow_fallback,
            )
            if mod1_meta.get("warnings"):
                warnings.extend(mod1_meta["warnings"])
            if raw_tensor is None:
                errors.extend(mod1_errors)
                timing_breakdown["total_execution_ms"] = (time.perf_counter() - start_total) * 1000.0
                return VoxelPipelineResult(
                    query=query,
                    execution_mode=display_mode,
                    status="FAILED",
                    is_synthetic=False,
                    is_mock_prediction=False,
                    data_source="Sentinel-2 L2A STAC Acquisition",
                    resolved_location=resolved_location,
                    contract=contract.to_dict(),
                    task_type=task_key,
                    num_classes=num_classes,
                    class_list=class_list,
                    input_tensor_shape=[],
                    selected_bands=[],
                    selected_indices=[],
                    band_reduction_pct=0.0,
                    mask_provenance="none",
                    timing_breakdown_ms=timing_breakdown,
                    diagnostics=diagnostics,
                    errors=errors,
                    warnings=warnings,
                )
            cache_tag = f" [{mod1_meta.get('cache_status', 'Live')}]"
            data_source_label = f"Real Sentinel-2 L2A Scene ({mod1_meta.get('stac_item_id')}){cache_tag} via Element84 STAC"
            is_synthetic_flag = False
            s_id = mod1_meta.get("stac_item_id") or mod1_meta.get("scene_id") or mod1_meta.get("selected_item")
            if not s_id:
                import re
                match = re.search(r"S2[AB]_[0-9A-Z_]+_L2A", data_source_label)
                if match:
                    s_id = match.group(0)
            diagnostics["module1_metadata"] = mod1_meta
            diagnostics["scene_id"] = s_id
            diagnostics["stac_item_id"] = s_id
            diagnostics["cache_status"] = mod1_meta.get("cache_status", "Unknown")
        else:
            # Check if multi-tile is requested based on span
            from utils.llm_parser import calculate_geographic_span
            span_w_km, span_h_km = calculate_geographic_span(contract.bbox)
            # Minimal 0.001 km (1.0m) numerical tolerance absorbs sub-pixel/float discretization
            # while strictly guaranteeing that any AOI > 2.56 km receives >= 2 tiles (0% undercoverage)
            tiles_x = max(1, int(math.ceil((span_w_km - 0.001) / 2.56)))
            tiles_y = max(1, int(math.ceil((span_h_km - 0.001) / 2.56)))
            total_tiles = tiles_x * tiles_y

            if total_tiles > 1:
                tiles_data = []
                d_lon = (contract.bbox[2] - contract.bbox[0]) / tiles_x
                d_lat = (contract.bbox[3] - contract.bbox[1]) / tiles_y
                for r_idx in range(tiles_y):
                    for c_idx in range(tiles_x):
                        tile_t, _ = self.generate_synthetic_tile(task_key, seed=synthetic_seed + r_idx * tiles_x + c_idx)
                        t_b = [
                            round(contract.bbox[0] + c_idx * d_lon, 6),
                            round(contract.bbox[3] - (r_idx + 1) * d_lat, 6),
                            round(contract.bbox[0] + (c_idx + 1) * d_lon, 6),
                            round(contract.bbox[3] - r_idx * d_lat, 6),
                        ]
                        tiles_data.append({
                            "tile_id": f"tile_{r_idx:03d}_{c_idx:03d}",
                            "processed_tensor": tile_t,
                            "bounds": t_b,
                            "diagnostics": {"tile_shape": list(tile_t.shape)},
                        })
                raw_tensor = tiles_data[0]["processed_tensor"]
                mod1_meta = {
                    "synthetic_seed": synthetic_seed,
                    "provenance": "synthetic_mock_fixture",
                    "total_scene_tiles": len(tiles_data),
                    "tiles_data": tiles_data,
                }
            else:
                raw_tensor, mod1_meta = self.generate_synthetic_tile(task_key, seed=synthetic_seed)
                mod1_meta["tiles_data"] = [{
                    "tile_id": "tile_000_000",
                    "processed_tensor": raw_tensor,
                    "bounds": contract.bbox,
                    "diagnostics": {"tile_shape": list(raw_tensor.shape)},
                }]
            data_source_label = f"Synthetic Deterministic Surface Reflectance Tile (Seed: {synthetic_seed})"
            is_synthetic_flag = True
            diagnostics["module1_metadata"] = mod1_meta

        timing_breakdown["module_1_ingestion_ms"] = (time.perf_counter() - t1_start) * 1000.0

        # Tensor Layout Compatibility Handling: Ensure (13, 256, 256)
        if raw_tensor.ndim == 3 and raw_tensor.shape[2] == 13:
            tensor_channel_first = np.transpose(raw_tensor, (2, 0, 1))
        elif raw_tensor.ndim == 3 and raw_tensor.shape[0] == 13:
            tensor_channel_first = raw_tensor
        else:
            raise ValueError(f"Invalid tensor shape from Module 1: {raw_tensor.shape}")

        input_shape_list = list(tensor_channel_first.shape)
        if self.verbose:
            print(f"  [+] Module 1 Tensor Prepared : Shape {input_shape_list} | Scale: [0.0, 1.0]")
            print(f"  [+] Module 1 Data Source     : {data_source_label}")

        # Store calibrated surface reflectance tensor & per-band statistics in diagnostics
        diagnostics["calibrated_tensor"] = tensor_channel_first
        diagnostics["band_spectral_means"] = [float(np.mean(tensor_channel_first[i])) for i in range(13)]
        diagnostics["band_spectral_stds"] = [float(np.std(tensor_channel_first[i])) for i in range(13)]

        # Generate and save RGB preview composite (B04: Red, B03: Green, B02: Blue)
        try:
            r = tensor_channel_first[3]  # B04
            g = tensor_channel_first[2]  # B03
            b = tensor_channel_first[1]  # B02
            rgb = np.stack([r, g, b], axis=-1)
            p2, p98 = np.percentile(rgb, (2, 98))
            if p98 > p2:
                rgb_scaled = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
            else:
                rgb_scaled = np.clip(rgb, 0.0, 1.0)
            
            rgb_preview_path = (Path(__file__).resolve().parent / "results" / "latest_inference_rgb.png").resolve()
            rgb_preview_path.parent.mkdir(parents=True, exist_ok=True)
            import matplotlib.image as mpimg
            mpimg.imsave(rgb_preview_path, rgb_scaled)
            diagnostics["rgb_preview_path"] = str(rgb_preview_path)
            diagnostics["rgb_image_array"] = rgb_scaled
        except Exception as e:
            logger.debug(f"RGB preview generation skipped: {e}")

        # ---------------------------------------------------------------------
        # MODULE 2: Physics-Guided Spectral Band Optimization / Task Masking
        # ---------------------------------------------------------------------
        t2_start = time.perf_counter()
        logger.info("[MODULE 2] Applying task-specific spectral band selection...")

        effective_target_count = target_band_count or contract.target_band_count
        (
            mask_tensor,
            selected_indices,
            selected_bands,
            mask_prov,
            reduction_pct,
            used_mask_path,
            mod2_diag,
        ) = self.resolve_band_mask(
            tensor_13_bands=tensor_channel_first,
            task_key=task_key,
            precomputed_mask_path=precomputed_mask_path,
            optimize=optimize,
            target_band_count=effective_target_count,
            scorecard_weights=contract.scorecard_weights,
            seed=synthetic_seed,
        )
        timing_breakdown["module_2_optimization_ms"] = (time.perf_counter() - t2_start) * 1000.0
        diagnostics["module2_diagnostics"] = mod2_diag

        if self.verbose:
            print(f"  [+] Module 2 Mask Strategy   : {mask_prov} ({len(selected_indices)} bands: {selected_bands})")
            print(f"  [+] Module 2 Data Reduction  : {reduction_pct:.1f}% data volume reduction")

        # ---------------------------------------------------------------------
        # Scientific Decoupling & Mask Compatibility Verification
        # ---------------------------------------------------------------------
        candidate_bwoa_bands = None
        candidate_bwoa_indices = None
        mask_compatibility_note = None
        inference_indices = selected_indices
        inference_bands = selected_bands
        inference_mask_tensor = mask_tensor

        target_ckpt = checkpoint_path
        if target_ckpt is None and task_key:
            real_ckpt = self.checkpoint_dir / f"real_{task_key}" / "best_fusion_net.pth"
            if real_ckpt.exists():
                target_ckpt = real_ckpt

        if target_ckpt and Path(target_ckpt).exists():
            try:
                ckpt_meta = torch.load(Path(target_ckpt), map_location="cpu", weights_only=False)
                ckpt_mask_raw = ckpt_meta.get("band_mask")
                if ckpt_mask_raw is not None:
                    ckpt_trained_indices = [
                        i for i, v in enumerate(ckpt_mask_raw)
                        if (v == 1 if isinstance(v, (int, float)) else bool(v))
                    ]
                    if sorted(selected_indices) != sorted(ckpt_trained_indices):
                        # Candidate BWOA mask differs from verified checkpoint training mask
                        candidate_bwoa_bands = list(selected_bands)
                        candidate_bwoa_indices = list(selected_indices)
                        diagnostics["live_bwoa_candidate_mask"] = {
                            "selected_indices": candidate_bwoa_indices,
                            "selected_bands": candidate_bwoa_bands,
                            "reduction_pct": reduction_pct,
                            "optimization_metrics": mod2_diag,
                        }
                        # Decouple for inference to preserve model validity
                        inference_indices = ckpt_trained_indices
                        inference_bands = [INDEX_TO_BAND[i] for i in ckpt_trained_indices if i in INDEX_TO_BAND]
                        inf_mask_arr = np.zeros(TOTAL_BANDS, dtype=np.float32)
                        inf_mask_arr[inference_indices] = 1.0
                        inference_mask_tensor = torch.tensor(inf_mask_arr, dtype=torch.float32)

                        mask_compatibility_note = (
                            f"Scientific Decoupling Notice: Fresh BWOA optimization produced candidate mask {candidate_bwoa_bands} ({len(candidate_bwoa_bands)} channels, {reduction_pct:.1f}% volume reduction). "
                            f"Because this candidate mask differs from the calibrated model's verified training mask {inference_bands} ({len(inference_bands)} channels), "
                            f"classification inference was executed using the verified training mask to avoid feeding uncalibrated spectral combinations into trained weights."
                        )
                        warnings.append("Live BWOA candidate mask is decoupled from classification inference to preserve model validity.")
                        mask_prov = "live_bwoa_candidate_decoupled"
            except Exception as e:
                logger.debug(f"Pre-check of checkpoint mask failed: {e}")

        # ---------------------------------------------------------------------
        # MODULE 3: Hybrid CNN-ViT Classification, Explainability & Search
        # ---------------------------------------------------------------------
        t3_start = time.perf_counter()
        logger.info("[MODULE 3] Loading model and executing classification inference...")

        model, model_prov, active_ckpt_path, model_warnings = self.inspect_and_load_model(
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            selected_channels=len(inference_indices),
            is_real_mode=is_real_mode,
            task_key=task_key,
            selected_indices=inference_indices,
        )
        warnings.extend(model_warnings)

        prediction_result = None
        embedding_np = None
        status = "COMPLETE"

        if model is None:
            # Gated check in Real Mode or missing checkpoint
            status = "PARTIALLY_COMPLETE"
            logger.info("Pipeline reached PARTIALLY_COMPLETE status (preprocessing and band selection complete).")
        else:
            # Prepare torch tensor: (1, 13, 256, 256) on target device
            tile_input = torch.from_numpy(tensor_channel_first).unsqueeze(0).float().to(self.device)
            mask_device = inference_mask_tensor.to(self.device)

            with torch.no_grad():
                logits, embedding = model(tile_input, mask=mask_device, return_embedding=True)
                probs = torch.softmax(logits, dim=-1).squeeze(0)

            pred_class_id = int(torch.argmax(probs).item())
            active_class_list = class_list if len(class_list) == len(probs) else [f"Class_{i}" for i in range(len(probs))]
            pred_class_name = active_class_list[pred_class_id] if 0 <= pred_class_id < len(active_class_list) else f"Class_{pred_class_id}"
            conf_pct = float(probs[pred_class_id].item() * 100.0)
            embedding_np = embedding.squeeze(0).cpu().numpy()

            prediction_result = {
                "predicted_class_id": pred_class_id,
                "predicted_class_name": pred_class_name,
                "confidence_pct": conf_pct,
                "all_probabilities": {
                    active_class_list[i]: float(probs[i].item() * 100.0)
                    for i in range(len(active_class_list))
                },
                "provenance": "synthetic_mock_prediction" if is_synthetic_flag or model_prov == "mock_checkpoint" else "real_inference",
            }
            if self.verbose:
                print(f"  [+] Module 3 Prediction      : {pred_class_name} (Confidence: {conf_pct:.2f}%)")
                print(f"  [+] Module 3 Embedding       : Shape ({embedding_np.shape[0]},) [1216-dim representation]")

        timing_breakdown["module_3_inference_ms"] = (time.perf_counter() - t3_start) * 1000.0

        # Multi-tile AOI Support (Requirements 3 & 4: Preserve spatial integrity across multi-tile rectangles)
        tiles_data = mod1_meta.get("tiles_data", [])
        if len(tiles_data) > 1 and model is not None:
            multi_tile_preds = []
            for idx, t_item in enumerate(tiles_data):
                t_id = t_item.get("tile_id", f"tile_{idx}")
                t_tensor = t_item.get("processed_tensor", t_item.get("data"))
                if t_tensor is None:
                    continue
                t_inp = torch.from_numpy(t_tensor).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    t_logits = model(t_inp, mask=mask_device)
                    t_probs = torch.softmax(t_logits, dim=-1).squeeze(0)
                t_pid = int(torch.argmax(t_probs).item())
                t_pname = active_class_list[t_pid] if 0 <= t_pid < len(active_class_list) else f"Class_{t_pid}"
                t_conf = float(t_probs[t_pid].item() * 100.0)
                t_bounds = t_item.get("bounds", t_item.get("diagnostics", {}).get("bounds"))
                multi_tile_preds.append({
                    "tile_id": t_id,
                    "predicted_class_id": t_pid,
                    "predicted_class_name": t_pname,
                    "confidence_pct": t_conf,
                    "bounds": t_bounds,
                    "all_probabilities": {
                        active_class_list[i]: float(t_probs[i].item() * 100.0)
                        for i in range(len(active_class_list))
                    },
                })
            diagnostics["multi_tile_predictions"] = multi_tile_preds
            diagnostics["total_tiles_analyzed"] = len(multi_tile_preds)
            spatial_msg = f"Multi-tile Footprint: Evaluated {len(multi_tile_preds)} spatial tiles (256×256 px each) across the rectangular AOI. Individual tile classifications are reported without whole-area artificial averaging."
            diagnostics["spatial_integrity_warning"] = spatial_msg
            warnings.append(spatial_msg)

            # Record union bounding box of all processed tiles
            valid_b = [tp["bounds"] for tp in multi_tile_preds if isinstance(tp.get("bounds"), (list, tuple)) and len(tp["bounds"]) == 4]
            if valid_b:
                actual_proc_bbox = [
                    round(min(b[0] for b in valid_b), 6),
                    round(min(b[1] for b in valid_b), 6),
                    round(max(b[2] for b in valid_b), 6),
                    round(max(b[3] for b in valid_b), 6),
                ]
            else:
                actual_proc_bbox = contract.bbox
            diagnostics["actual_processed_bbox"] = actual_proc_bbox
            diagnostics["actual_processed_tile_count"] = len(multi_tile_preds)
        else:
            single_b = None
            if tiles_data and isinstance(tiles_data[0].get("bounds"), (list, tuple)) and len(tiles_data[0]["bounds"]) == 4:
                single_b = tiles_data[0]["bounds"]
            diagnostics["actual_processed_bbox"] = single_b or contract.bbox
            diagnostics["actual_processed_tile_count"] = 1

        # Clean bulky ndarrays from module1_metadata tiles_data for clean serialization
        if "tiles_data" in mod1_meta and isinstance(mod1_meta["tiles_data"], list):
            for td in mod1_meta["tiles_data"]:
                if isinstance(td, dict):
                    td.pop("processed_tensor", None)
                    td.pop("data", None)

        # ---------------------------------------------------------------------
        # Optional: Explainability (SHAP Attributions)
        # ---------------------------------------------------------------------
        explainability_result = None
        if run_shap and model is not None:
            t_shap = time.perf_counter()
            logger.info("Computing Permutation SHAP channel attributions across active bands...")
            try:
                shap_attributions = compute_permutation_shap_attributions(
                    model=model,
                    sample_images=tile_input,
                    mask=mask_device,
                    num_permutations=5,
                    device=str(self.device),
                )
                explainability_result = {
                    "channel_attributions": shap_attributions,
                    "top_contributing_bands": sorted(
                        shap_attributions.values(),
                        key=lambda x: x.get("percentage", 0),
                        reverse=True
                    ),
                }
                if self.verbose:
                    top_b = explainability_result["top_contributing_bands"][0]
                    print(f"  [+] SHAP Top Spectral Band   : {top_b['name']} ({top_b['percentage']:.1f}% relative impact)")
            except Exception as e:
                warnings.append(f"SHAP attribution computation failed: {e}")
            timing_breakdown["shap_explainability_ms"] = (time.perf_counter() - t_shap) * 1000.0

        # ---------------------------------------------------------------------
        # Optional: Vector Retrieval (FAISS Nearest Terrain Neighbors)
        # ---------------------------------------------------------------------
        similar_terrain_results = None
        if run_faiss and embedding_np is not None:
            t_faiss = time.perf_counter()
            faiss_index_file = self.checkpoint_dir / "terrain_index.faiss"
            if faiss_index_file.exists():
                try:
                    scores, indices, metadata, latency_ms = query_similar_terrain(
                        query_embedding=embedding_np,
                        top_k=5,
                        index_path=str(faiss_index_file),
                    )
                    similar_terrain_results = []
                    for rank, (s, idx, meta) in enumerate(zip(scores, indices, metadata), 1):
                        retrieved_class_id = meta.get("true_class", -1)
                        retrieved_class_name = (
                            LAND_COVER_CLASSES[retrieved_class_id]
                            if 0 <= retrieved_class_id < len(LAND_COVER_CLASSES)
                            else "N/A"
                        )
                        similar_terrain_results.append({
                            "rank": rank,
                            "similarity_score": float(s),
                            "index": int(idx),
                            "tile_id": meta.get("tile_id", f"TILE_{idx}"),
                            "split": meta.get("split", "unknown"),
                            "class_id": retrieved_class_id,
                            "class_name": retrieved_class_name,
                        })
                    diagnostics["faiss_retrieval_latency_ms"] = latency_ms
                    if self.verbose and similar_terrain_results:
                        print(f"  [+] FAISS Nearest Neighbor   : {similar_terrain_results[0]['tile_id']} ({similar_terrain_results[0]['class_name']}, sim={similar_terrain_results[0]['similarity_score']:.4f})")
                except Exception as e:
                    warnings.append(f"FAISS nearest neighbor retrieval failed: {e}")
            else:
                diagnostics["faiss_index_status"] = "Index file not found; skipping retrieval."
            timing_breakdown["faiss_retrieval_ms"] = (time.perf_counter() - t_faiss) * 1000.0

        # Total timing
        timing_breakdown["total_pipeline_latency_ms"] = (time.perf_counter() - start_total) * 1000.0

        # Build final structured result object
        is_mock_pred = bool(is_synthetic_flag or model_prov.startswith("mock") or prediction_result is None)

        result_obj = VoxelPipelineResult(
            query=query,
            execution_mode=display_mode,
            status=status,
            is_synthetic=is_synthetic_flag,
            is_mock_prediction=is_mock_pred,
            data_source=data_source_label,
            resolved_location=resolved_location,
            contract=contract.to_dict(),
            task_type=task_key,
            num_classes=num_classes,
            class_list=class_list,
            input_tensor_shape=input_shape_list,
            selected_bands=inference_bands,
            selected_indices=inference_indices,
            band_reduction_pct=reduction_pct,
            mask_provenance=mask_prov,
            mask_path=used_mask_path,
            candidate_bwoa_bands=candidate_bwoa_bands,
            mask_compatibility_note=mask_compatibility_note,
            model_checkpoint=active_ckpt_path,
            model_provenance=model_prov,
            prediction=prediction_result,
            embedding_dim=EMBEDDING_DIM if embedding_np is not None else None,
            explainability=explainability_result,
            similar_terrain=similar_terrain_results,
            timing_breakdown_ms=timing_breakdown,
            diagnostics=diagnostics,
            errors=errors,
            warnings=warnings,
        )

        if output_json:
            result_obj.save_json(output_json)

        if self.verbose:
            print("-" * 78)
            print(f"  PIPELINE STATUS : {status}")
            print(f"  TOTAL LATENCY   : {timing_breakdown['total_pipeline_latency_ms']:.2f} ms")
            if warnings:
                print(f"  WARNINGS        : {len(warnings)} reported")
            if errors:
                print(f"  ERRORS          : {len(errors)} reported")
            print("=" * 78 + "\n")

        return result_obj


# -----------------------------------------------------------------------------
# Command Line Interface (CLI) Entry Point
# -----------------------------------------------------------------------------
def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VOXEL Unified Earth Observation & Spectral Intelligence Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default="Identify water bodies around Chilika Lake, Odisha, India",
        help="Natural language remote sensing query or task objective.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["offline", "real"],
        default="offline",
        help="Execution mode: 'offline' for synthetic smoke test; 'real' for live satellite acquisition.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Explicit task override (e.g. waterbody, forest_cover, general).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained FusionNet PyTorch checkpoint (.pth).",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Execute live Module 2 BWOA metaheuristic optimization instead of precomputed task mask.",
    )
    parser.add_argument(
        "--target-bands",
        type=int,
        default=None,
        help="Target spectral band budget for BWOA optimization (e.g. 4, 5, 6).",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Compute Permutation SHAP spectral band attributions.",
    )
    parser.add_argument(
        "--no-faiss",
        action="store_true",
        help="Disable FAISS vector database nearest-neighbor terrain search.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Destination JSON file path to persist structured pipeline results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic offline reproducibility.",
    )
    return parser


def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    pipeline = VoxelPipeline(default_mode=args.mode)
    result = pipeline.run(
        query=args.query,
        mode=args.mode,
        task_override=args.task,
        checkpoint_path=args.checkpoint,
        optimize=args.optimize,
        target_band_count=args.target_bands,
        run_shap=args.shap,
        run_faiss=not args.no_faiss,
        output_json=args.output,
        synthetic_seed=args.seed,
    )
    return 0 if result.status in ["COMPLETE", "PARTIALLY_COMPLETE"] else 1


if __name__ == "__main__":
    sys.exit(main())
