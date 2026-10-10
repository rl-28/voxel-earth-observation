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
