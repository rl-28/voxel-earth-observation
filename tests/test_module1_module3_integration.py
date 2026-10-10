"""
Integration unit tests for Module 1 and Module 2 extensions supporting Module 3:
1. Optional reflectance output mode (DN / 10000 clipped to [0, 1], no Z-score).
2. Optional 13-band output mode (zero-filled B10 at index 10, channel-first (13, 256, 256)).
3. Module 2 save_band_mask() and refusal of placeholder index 10.
4. Default behavior preservation across both modules.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from config.constants import NUM_BANDS, SENTINEL2_BAND_KEYS
from module1_ingestion.config import OUTPUT_DTYPE
from module1_ingestion.pipeline import Module1Pipeline
from module2_optimization import save_band_mask, run_module2_optimization
from utils.band_mask import load_band_mask, validate_band_mask


def test_module1_default_behavior_preserved():
    """Confirms default pipeline maintains (256, 256, C) channels-last Z-score output."""
    raw_12 = np.random.uniform(500, 4000, size=(256, 256, 12)).astype(OUTPUT_DTYPE)
    pipeline = Module1Pipeline()
    res = pipeline.process_tile({"tile_data": raw_12, "row_idx": 0, "col_idx": 0})

    assert res.output_shape == (256, 256, 12)
    assert res.processed_tensor.shape == (256, 256, 12)
    assert res.processed_tensor.dtype == OUTPUT_DTYPE
    # Z-score normalization standardly produces negative values around mean 0
    assert (res.processed_tensor < 0.0).any()
    assert res.diagnostics["contract_type"] == "12-band intermediate"


def test_module1_reflectance_output_mode():
    """Verifies reflectance mode yields surface reflectance in [0, 1] without Z-score."""
    # Synthetic DN values between 0 and 15000 (standard Sentinel-2 L2A range)
    raw_12 = np.linspace(0, 15000, 256 * 256 * 12, dtype=OUTPUT_DTYPE).reshape(256, 256, 12)
    pipeline = Module1Pipeline(output_mode="reflectance")
    res = pipeline.process_tile({"tile_data": raw_12, "row_idx": 0, "col_idx": 0})

    assert res.output_shape == (256, 256, 12)
    assert res.processed_tensor.dtype == OUTPUT_DTYPE
    # All values must be in [0.0, 1.0]
    assert res.processed_tensor.min() >= 0.0
    assert res.processed_tensor.max() <= 1.0
    # No negative values
    assert not (res.processed_tensor < 0.0).any()
    assert res.diagnostics["output_mode"] == "reflectance"


def test_module1_13_band_output_mode():
    """Verifies output_13_bands=True inserts zero B10 at index 10 and produces (13, 256, 256)."""
    raw_12 = np.random.uniform(200, 5000, size=(256, 256, 12)).astype(OUTPUT_DTYPE)
    pipeline = Module1Pipeline(output_13_bands=True)
    res = pipeline.process_tile({"tile_data": raw_12, "row_idx": 1, "col_idx": 2})

    # Contract requirement: (13, 256, 256) channel-first float32
    assert res.output_shape == (13, 256, 256)
    assert res.processed_tensor.shape == (13, 256, 256)
    assert res.processed_tensor.dtype == OUTPUT_DTYPE
    # Channel index 10 (Cirrus B10 placeholder) must be identically zero
    assert np.all(res.processed_tensor[10, :, :] == 0.0)
    # Surrounding channels (e.g. index 9, index 11) must not be all zeros
    assert not np.all(res.processed_tensor[9, :, :] == 0.0)
    assert not np.all(res.processed_tensor[11, :, :] == 0.0)
    assert "13-band channel-first" in res.diagnostics["contract_type"]


def test_module1_combined_reflectance_and_13_bands():
    """Verifies combined options output (13, 256, 256) in [0, 1] range for Module 3."""
    raw_12 = np.random.uniform(0, 10000, size=(256, 256, 12)).astype(OUTPUT_DTYPE)
    pipeline = Module1Pipeline(output_mode="reflectance", output_13_bands=True)
    res = pipeline.process_tile({"tile_data": raw_12, "row_idx": 0, "col_idx": 0})

    tensor = res.processed_tensor
    assert tensor.shape == (13, 256, 256)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    assert np.all(tensor[10] == 0.0)


def test_module2_save_band_mask_refuses_placeholder_b10(tmp_path):
    """Verifies save_band_mask excludes index 10 and writes standard JSON format."""
    out_json = tmp_path / "test_mask.json"
    # Provide indices including 10 (B10)
    indices_with_b10 = [1, 2, 3, 7, 10, 11, 12]
    
    saved_dict = save_band_mask(
        path=out_json,
        selected_indices=indices_with_b10,
        exclude_placeholder_index=10
    )

    assert out_json.exists()
    assert 10 not in saved_dict["selected_indices"]
    assert "B10" not in saved_dict["selected_bands"]
    assert saved_dict["selected_indices"] == [1, 2, 3, 7, 11, 12]
    assert saved_dict["total_bands"] == 13
    assert saved_dict["method"] == "BWOA"
    assert saved_dict["dataset"] == "Sentinel-2 L2A"

    # Verify JSON content matches config/example_band_mask.json schema
    with open(out_json, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)

    assert loaded_json == saved_dict
    assert loaded_json["total_bands"] == 13

    # Verify compatibility with Module 3's load_band_mask
    mask_tensor = load_band_mask(str(out_json))
    assert isinstance(mask_tensor, torch.Tensor)
    assert mask_tensor.shape == (13,)
    assert mask_tensor[10].item() == 0.0
    assert mask_tensor[1].item() == 1.0
    assert mask_tensor[7].item() == 1.0


def test_module2_optimization_exclude_placeholder_flag(tmp_path):
    """Verifies run_module2_optimization respects exclude_placeholder_b10 and exports valid mask."""
    # Small synthetic input (2 tiles of 256x256x13)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(2, 256, 256, 13)).astype(np.float32)
    # Band 10 is zero placeholder
    X[..., 10] = 0.0

    out_mask_path = tmp_path / "exported_mask.json"
    best_mask, selected_indices, metrics = run_module2_optimization(
        input_tensor=X,
        target_count=6,
        pop_size=4,
        max_iter=3,
        seed=42,
        export_mask_path=out_mask_path,
        exclude_placeholder_b10=True
    )

    assert 10 not in selected_indices
    assert best_mask[10] == 0
    assert out_mask_path.exists()

    with open(out_mask_path, "r", encoding="utf-8") as f:
        exported_data = json.load(f)

    assert 10 not in exported_data["selected_indices"]
    assert "B10" not in exported_data["selected_bands"]
    assert exported_data["total_bands"] == 13


def test_module1_spatial_only_wavelet_preserves_distinct_bands():
    """
    Verifies that spatial_only_wavelet=True prevents spectral smoothing along axis 2,
    keeping adjacent bands distinct, while default spatial_only_wavelet=False averages pairs.
    """
    # Create synthetic tile with distinct band levels: band c has baseline (c+1)*800 DN
    rng = np.random.default_rng(123)
    raw_12 = np.zeros((256, 256, 12), dtype=OUTPUT_DTYPE)
    for c in range(12):
        raw_12[:, :, c] = (c + 1) * 800.0 + rng.uniform(10, 50, size=(256, 256)).astype(OUTPUT_DTYPE)

    # 1. Pipeline with spatial_only_wavelet=True
    pipeline_spatial = Module1Pipeline(
        output_mode="reflectance",
        output_13_bands=True,
        spatial_only_wavelet=True
    )
    res_spatial = pipeline_spatial.process_tile({"tile_data": raw_12, "row_idx": 0, "col_idx": 0})
    tensor_sp = res_spatial.processed_tensor  # (13, 256, 256)

    # In 13-band layout, the original 12 bands map to:
    # 0:B01, 1:B02, 2:B03, 3:B04, 4:B05, 5:B06, 6:B07, 7:B08, 8:B8A, 9:B09, 10:B10 (placeholder), 11:B11, 12:B12
    # The pairs that were previously identical are:
    test_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (11, 12)]
    for idx1, idx2 in test_pairs:
        diff = np.abs(tensor_sp[idx1] - tensor_sp[idx2])
        assert diff.max() > 0.05, f"Bands {idx1} and {idx2} should be distinct in spatial_only mode, but max diff is {diff.max()}"
        assert not np.allclose(tensor_sp[idx1], tensor_sp[idx2], atol=1e-5)

    # 2. Compare against default pipeline (spatial_only_wavelet=False)
    pipeline_default = Module1Pipeline(
        output_mode="reflectance",
        output_13_bands=True,
        spatial_only_wavelet=False
    )
    res_default = pipeline_default.process_tile({"tile_data": raw_12, "row_idx": 0, "col_idx": 0})
    tensor_def = res_default.processed_tensor
    for idx1, idx2 in test_pairs:
        assert np.allclose(tensor_def[idx1], tensor_def[idx2], atol=1e-6), (
            f"Bands {idx1} and {idx2} should be identical in default 3D DWT mode"
        )


def test_module1_keep_water_preserves_water_pixels():
    """
    Verifies that keep_water=True preserves pixels matching the physical water signature
    (NDWI > 0, Green > NIR) which would otherwise be zeroed by the cloud shadow heuristic.
    """
    # Create synthetic tile with reflectance values
    # In 12-band input: index 1=B02, 2=B03(Green), 3=B04(Red), 7=B08(NIR)
    raw_12 = np.full((256, 256, 12), 1500.0, dtype=OUTPUT_DTYPE)

    # Set half the tile to water signature: low Red (0.02 = 200 DN), low NIR (0.01 = 100 DN),
    # but higher Green (0.08 = 800 DN) -> NDWI > 0
    raw_12[:128, :, 2] = 800.0   # B03 (Green)
    raw_12[:128, :, 3] = 200.0   # B04 (Red)
    raw_12[:128, :, 7] = 100.0   # B08 (NIR)

    # Set the other half to cloud shadow: low across all bands (100 DN)
    raw_12[128:, :, 2] = 100.0   # B03 (Green)
    raw_12[128:, :, 3] = 100.0   # B04 (Red)
    raw_12[128:, :, 7] = 100.0   # B08 (NIR)

    # Pipeline with keep_water=False (default behavior)
    p_default = Module1Pipeline(output_mode="reflectance", keep_water=False, spatial_only_wavelet=True)
    res_default = p_default.process_tile({"tile_data": raw_12, "row_idx": 0, "col_idx": 0})
    # Both halves are zeroed under default shadow rule because B04 < 0.05 and B08 < 0.05
    assert np.all(res_default.processed_tensor[:128, :, :] == 0.0)

    # Pipeline with keep_water=True
    p_water = Module1Pipeline(output_mode="reflectance", keep_water=True, spatial_only_wavelet=True)
    res_water = p_water.process_tile({"tile_data": raw_12, "row_idx": 0, "col_idx": 0})
    # Water half must be preserved (non-zero)
    assert not np.all(res_water.processed_tensor[:128, :, :] == 0.0)
    assert (res_water.processed_tensor[:128, :, :] > 0.0).any()
    # Non-water shadow half should still be zeroed
    assert np.all(res_water.processed_tensor[128:, :, :] == 0.0)

