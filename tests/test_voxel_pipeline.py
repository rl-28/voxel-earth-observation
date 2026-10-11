"""
tests/test_voxel_pipeline.py - Unified End-to-End VOXEL Pipeline Integration Tests

Tests:
1. Stage 0 to Module 1 handoff (query parsing, geocoding to BBOX, and contract resolution).
2. Module 1 to Module 2 tensor conversion (channel-first to channel-last layout and reflectance scaling).
3. Spectral mask application and refusal of placeholder Band 10.
4. Module 2 to Module 3 compatibility (dynamic channel slicing and dimensional invariants).
5. Successful offline smoke-test inference end-to-end.
6. Correct handling and withholding of mock checkpoint in Real Mode (PARTIALLY_COMPLETE).
7. Invalid query and location fallback handling.
8. Correct band ordering and canonical indexing.
9. Mock vs real output labeling and data provenance.
10. Complete pipeline result schema and JSON persistence.
"""

import os
import json
import pytest
import numpy as np
import torch
from pathlib import Path

from voxel_pipeline import VoxelPipeline, VoxelPipelineResult
from utils.llm_parser import VoxelStageZeroParser, VoxelExecutionContract
from models.fusion_net import FusionNet
from config.constants import (
    NUM_BANDS,
    NUM_CLASSES,
    EMBEDDING_DIM,
    LAND_COVER_CLASSES,
    TASKS,
    SENTINEL2_BAND_KEYS,
)


@pytest.fixture
def pipeline():
    """Instantiate standard test pipeline."""
    return VoxelPipeline(
        default_mode="offline",
        checkpoint_dir="checkpoints",
        mask_dir="checkpoints/masks",
        verbose=False,
    )


# -----------------------------------------------------------------------------
# 1. Stage 0 to Module 1 Handoff
# -----------------------------------------------------------------------------
def test_stage0_to_module1_handoff(pipeline):
    """Verify Stage 0 parses natural language query into valid contract for Module 1."""
    query = "Detect water bodies and wetlands in Chilika Lake, Odisha, India"
    parser = VoxelStageZeroParser()
    contract = parser.parse(query)

    assert isinstance(contract, VoxelExecutionContract)
    assert len(contract.bbox) == 4
    min_lon, min_lat, max_lon, max_lat = contract.bbox
    assert min_lon < max_lon
    assert min_lat < max_lat

    task_key, mask_path, num_classes, class_list, is_binary = pipeline.resolve_task_profile(contract.task_type)
    assert task_key == "waterbody"
    assert is_binary is True
    assert num_classes == 2
    assert class_list == ["Non-water", "Water"]


# -----------------------------------------------------------------------------
# 2. Module 1 to Module 2 Tensor Conversion & Reflectance Scaling
# -----------------------------------------------------------------------------
def test_module1_to_module2_tensor_conversion(pipeline):
    """Verify Module 1 produces valid 13-band reflectance tensor in [0, 1] with B10 placeholder."""
    tile, meta = pipeline.generate_synthetic_tile(task_key="waterbody", seed=100)

    assert tile.shape == (13, 256, 256)
    assert tile.dtype == np.float32
    assert np.all(tile >= 0.0) and np.all(tile <= 1.0)
    assert np.all(np.isfinite(tile))

    # Cirrus B10 at index 10 must be placeholder zero
    assert np.all(tile[10, :, :] == 0.0)

    # Conversion to channel-last (256, 256, 13) for Module 2
    tile_channel_last = np.transpose(tile, (1, 2, 0))
    assert tile_channel_last.shape == (256, 256, 13)


# -----------------------------------------------------------------------------
# 3. Spectral Mask Application & Placeholder B10 Refusal
# -----------------------------------------------------------------------------
def test_spectral_mask_application_and_b10_refusal(pipeline):
    """Verify precomputed and BWOA masks strictly refuse placeholder B10 (index 10)."""
    tile, _ = pipeline.generate_synthetic_tile(task_key="waterbody", seed=42)

    # Test precomputed waterbody mask
    mask_tensor, indices, bands, prov, reduction, path, diag = pipeline.resolve_band_mask(
        tensor_13_bands=tile,
        task_key="waterbody",
        precomputed_mask_path=Path("checkpoints/masks/waterbody_mask.json"),
        optimize=False,
    )

    assert isinstance(mask_tensor, torch.Tensor)
    assert mask_tensor.shape == (13,)
    assert 10 not in indices, "Placeholder Band 10 must never be active in mask!"
    assert "B10" not in bands
    assert reduction > 50.0  # At least 50% data volume reduction
    assert prov == "precomputed_mask"


# -----------------------------------------------------------------------------
# 4. Module 2 to Module 3 Compatibility & Dynamic Channel Slicing
# -----------------------------------------------------------------------------
def test_module2_to_module3_compatibility():
    """Verify FusionNet accepts the active mask channels without creating random conv layers."""
    device = torch.device("cpu")
    num_selected = 6
    mask = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float32)

    model = FusionNet(
        num_classes=2,
        total_bands=13,
        selected_channels=num_selected,
    ).to(device)

    x = torch.rand(2, 13, 256, 256, device=device)
    logits, embedding = model(x, mask=mask, return_embedding=True)

    assert logits.shape == (2, 2)
    assert embedding.shape == (2, EMBEDDING_DIM)

    # Incompatible channel count should raise ValueError rather than create random Conv2d
    incompatible_mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32) # 3 bands
    with pytest.raises(ValueError, match="Active mask band count"):
        model(x, mask=incompatible_mask)


# -----------------------------------------------------------------------------
# 5. Successful Offline Smoke-Test End-to-End
# -----------------------------------------------------------------------------
def test_offline_smoke_test_end_to_end(pipeline, tmp_path):
    """Verify end-to-end execution in Offline Smoke-Test Mode produces complete structured output."""
    out_json = tmp_path / "test_smoke_result.json"
    result = pipeline.run(
        query="Map forest canopy loss in Amazon Basin",
        mode="offline",
        run_shap=False,
        run_faiss=True,
        output_json=str(out_json),
    )

    assert isinstance(result, VoxelPipelineResult)
    assert result.status == "COMPLETE"
    assert result.execution_mode == "offline_mock"
    assert result.is_synthetic is True
    assert result.is_mock_prediction is True
    assert result.task_type == "forest_cover"
    assert result.num_classes == 2
    assert result.input_tensor_shape == [13, 256, 256]
    assert len(result.selected_bands) > 0
    assert result.band_reduction_pct > 0.0
    assert result.prediction is not None
    assert result.prediction["provenance"] == "synthetic_mock_prediction"
    assert "total_pipeline_latency_ms" in result.timing_breakdown_ms
    assert out_json.exists()


# -----------------------------------------------------------------------------
# 6. Real Mode Gating on Mock Checkpoint (PARTIALLY_COMPLETE)
# -----------------------------------------------------------------------------
def test_real_mode_mock_checkpoint_gating(pipeline):
    """Verify that in Real Mode, mock checkpoints are withheld and reported as PARTIALLY_COMPLETE."""
    model, prov, ckpt_path, warnings = pipeline.inspect_and_load_model(
        checkpoint_path="checkpoints/best_fusion_net.pth",
        num_classes=2,
        selected_channels=6,
        is_real_mode=True,
    )

    assert model is None, "In Real Mode, mock checkpoint must not be loaded for prediction!"
    assert prov == "mock_checkpoint_withheld_for_real_mode"
    assert any("withheld" in w.lower() for w in warnings)


# -----------------------------------------------------------------------------
# 7. Invalid Query & Location Fallback Handling
# -----------------------------------------------------------------------------
def test_invalid_query_fallback(pipeline):
    """Verify empty or unrecognized queries fallback gracefully without crashing."""
    result = pipeline.run(query="   ", mode="offline")
    assert result.status == "COMPLETE"
    assert result.resolved_location["name"] is not None
    assert len(result.resolved_location["bbox"]) == 4


# -----------------------------------------------------------------------------
# 8. Correct Canonical Band Order
# -----------------------------------------------------------------------------
def test_canonical_band_order():
    """Verify the 13 canonical spectral band keys match Sentinel-2 ordering."""
    assert len(SENTINEL2_BAND_KEYS) == 13
    assert SENTINEL2_BAND_KEYS[0] == "B01"
    assert SENTINEL2_BAND_KEYS[3] == "B04"
    assert SENTINEL2_BAND_KEYS[7] == "B08"
    assert SENTINEL2_BAND_KEYS[10] == "B10"
    assert SENTINEL2_BAND_KEYS[12] == "B12"


# -----------------------------------------------------------------------------
# 9. Pipeline Result Serialization & Schema
# -----------------------------------------------------------------------------
def test_result_serialization_schema(pipeline, tmp_path):
    """Verify VoxelPipelineResult serializes to valid JSON matching the full schema."""
    result = pipeline.run(
        query="Identify water bodies around Chilika Lake, Odisha, India",
        mode="offline",
        output_json=str(tmp_path / "schema_test.json"),
    )

    d = result.to_dict()
    required_keys = [
        "query",
        "execution_mode",
        "status",
        "is_synthetic",
        "is_mock_prediction",
        "data_source",
        "resolved_location",
        "contract",
        "task_type",
        "num_classes",
        "class_list",
        "input_tensor_shape",
        "selected_bands",
        "selected_indices",
        "band_reduction_pct",
        "mask_provenance",
        "model_provenance",
        "prediction",
        "timing_breakdown_ms",
        "diagnostics",
        "errors",
        "warnings",
    ]
    for key in required_keys:
        assert key in d, f"Missing required key in pipeline result schema: {key}"

    # Load back saved JSON
    with open(tmp_path / "schema_test.json", "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["query"] == result.query
    assert loaded["status"] == "COMPLETE"


# -----------------------------------------------------------------------------
# 10. Stage 0 Task Scope Qualifications
# -----------------------------------------------------------------------------
def test_stage0_task_scope_qualifications(pipeline):
    """Verify that Stage 0 qualifies limitations for temporal depth and canopy density."""
    # Test temporal water levels query
    res_water = pipeline.run(
        query="Track water levels over time in Loktak Lake",
        mode="offline",
        run_shap=False,
        run_faiss=False,
    )
    assert any("does not quantify temporal water level/depth" in w.lower() for w in res_water.warnings)

    # Test pixel canopy density query
    res_forest = pipeline.run(
        query="Measure tree canopy density percentage in Jim Corbett",
        mode="offline",
        run_shap=False,
        run_faiss=False,
    )
    assert any("does not produce continuous canopy density" in w.lower() for w in res_forest.warnings)


# -----------------------------------------------------------------------------
# 11. BWOA Mask Decoupling & Model Channel Compatibility
# -----------------------------------------------------------------------------
def test_bwoa_mask_checkpoint_decoupling(pipeline):
    """Verify that candidate BWOA masks differing from trained checkpoints are decoupled honestly."""
    # Forest checkpoint was calibrated with [B02, B04, B06, B11, B12]
    # We pass optimize=True to simulate candidate exploration
    res = pipeline.run(
        query="Assess forest cover in Jim Corbett National Park",
        mode="offline",
        optimize_bands=True,
        run_shap=False,
        run_faiss=False,
    )
    assert res.status == "COMPLETE"
    assert res.mask_provenance in ["live_bwoa_candidate_decoupled", "checkpoint_trained_mask_applied", "precomputed_mask", "bwoa_optimized"]
    # If live BWOA candidate differed from trained checkpoint, compatibility note must exist
    if res.candidate_bwoa_bands:
        assert res.mask_compatibility_note is not None
        assert "decoupled" in res.mask_compatibility_note.lower() or "candidate" in res.mask_compatibility_note.lower()


# -----------------------------------------------------------------------------
# 12. Single-Tile Spatial Permutation SHAP Test
# -----------------------------------------------------------------------------
def test_single_tile_spatial_shap_permutation():
    """Verify that compute_permutation_shap_attributions on a single tile (B=1) runs spatial permutation."""
    from explainability.shap_engine import compute_permutation_shap_attributions
    from models.fusion_net import FusionNet

    torch.manual_seed(42)
    device = torch.device("cpu")
    mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float32)  # 5 bands: B02, B04, B06, B11, B12
    active_channels = int(mask.sum().item())

    model = FusionNet(
        num_classes=2,
        total_bands=13,
        selected_channels=active_channels,
    ).to(device)
    model.eval()

    # Create a single tile (1, 13, 256, 256) with non-zero spatial variance
    single_tile = torch.rand(1, 13, 256, 256, device=device)

    attributions = compute_permutation_shap_attributions(
        model=model,
        sample_images=single_tile,
        mask=mask,
        num_permutations=2,
        device="cpu",
    )

    assert isinstance(attributions, dict)
    assert len(attributions) == 5  # 5 active channels
    for band_idx, data in attributions.items():
        assert "name" in data
        assert "raw_score" in data
        assert "percentage" in data
        assert "is_informative" in data
        assert isinstance(data["is_informative"], bool)


