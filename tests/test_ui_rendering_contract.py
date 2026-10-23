import ast
import os
import sys
from pathlib import Path
from dataclasses import fields
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voxel_pipeline import VoxelPipelineResult, VoxelPipeline


def test_result_contract_contains_all_ui_rendered_attributes():
    """
    Contract Integrity Test:
    Ensures that every single attribute accessed on `res` in `app.py`
    exists as a declared field or property on `VoxelPipelineResult`.
    This prevents any future AttributeError crashes during dashboard rendering.
    """
    app_path = PROJECT_ROOT / "app.py"
    with open(app_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    # Find render_dashboard function in app.py
    render_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "render_dashboard":
            render_fn = node
            break

    assert render_fn is not None, "render_dashboard function not found in app.py"

    # Extract all attribute accesses on `res` inside render_dashboard
    res_accessed_attrs = set()
    for node in ast.walk(render_fn):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "res":
            res_accessed_attrs.add(node.attr)

    # Get all fields and properties on VoxelPipelineResult
    declared_field_names = {f.name for f in fields(VoxelPipelineResult)}
    class_properties = {
        name for name, val in vars(VoxelPipelineResult).items()
        if isinstance(val, property) or callable(val)
    }
    all_supported_attrs = declared_field_names | class_properties

    # Verify that every attribute accessed on `res` in the dashboard exists
    missing_attrs = res_accessed_attrs - all_supported_attrs
    assert not missing_attrs, (
        f"VoxelPipelineResult is missing attributes accessed in app.py render_dashboard(): {missing_attrs}"
    )


def test_checkpoint_path_property_backward_compatibility():
    """
    Verifies that the `checkpoint_path` property correctly mirrors `model_checkpoint`
    for backward compatibility across all legacy tools and scripts.
    """
    dummy_result = VoxelPipelineResult(
        query="Test query",
        execution_mode="real_satellite",
        status="COMPLETE",
        is_synthetic=False,
        is_mock_prediction=False,
        data_source="Test Data Source",
        resolved_location={"name": "Test Location", "bbox": [78.0, 29.0, 78.1, 29.1]},
        contract={},
        task_type="waterbody",
        num_classes=2,
        class_list=["Non-water", "Water"],
        input_tensor_shape=[13, 256, 256],
        selected_bands=["B01", "B03", "B04", "B06", "B8A", "B11"],
        selected_indices=[0, 2, 3, 5, 8, 11],
        band_reduction_pct=53.8,
        mask_provenance="precomputed_mask",
        model_checkpoint="checkpoints/real_waterbody/best_fusion_net.pth",
        model_provenance="genuine_trained_model",
        prediction={"predicted_class_id": 1, "predicted_class_name": "Water", "confidence_pct": 96.77},
    )

    assert dummy_result.model_checkpoint == "checkpoints/real_waterbody/best_fusion_net.pth"
    assert dummy_result.checkpoint_path == "checkpoints/real_waterbody/best_fusion_net.pth"
    assert dummy_result.checkpoint_path == dummy_result.model_checkpoint


def test_ui_rendering_logic_single_and_multi_tile():
    """
    UI Smoke Test:
    Executes the exact dashboard formatting, telemetry calculation,
    and checkpoint extraction logic from `render_dashboard()` on both
    single-tile and multi-tile result objects to ensure zero runtime exceptions.
    """
    multi_tile_predictions = [
        {"tile_id": "tile_0", "bounds": [93.80, 24.54, 93.82, 24.56], "predicted_class_name": "Water", "confidence_pct": 97.5},
        {"tile_id": "tile_1", "bounds": [93.82, 24.54, 93.84, 24.56], "predicted_class_name": "Water", "confidence_pct": 95.8},
    ]

    res = VoxelPipelineResult(
        query="Track waterbody levels in Loktak Lake, Manipur",
        execution_mode="real_satellite",
        status="COMPLETE",
        is_synthetic=False,
        is_mock_prediction=False,
        data_source="Real Sentinel-2 L2A Scene (S2A_46REN_20241228_0_L2A) [Cached on Disk] via Element84 STAC",
        resolved_location={"name": "Loktak Lake, Manipur", "bbox": [93.8007, 24.5459, 93.8257, 24.5709]},
        contract={},
        task_type="waterbody",
        num_classes=2,
        class_list=["Non-water", "Water"],
        input_tensor_shape=[13, 256, 256],
        selected_bands=["B01", "B03", "B04", "B06", "B8A", "B11"],
        selected_indices=[0, 2, 3, 5, 8, 11],
        band_reduction_pct=53.8,
        mask_provenance="precomputed_mask",
        model_checkpoint="checkpoints/real_waterbody/best_fusion_net.pth",
        model_provenance="genuine_trained_model",
        prediction={
            "predicted_class_id": 1,
            "predicted_class_name": "Water",
            "confidence_pct": 96.77,
            "all_probabilities": {"Non-water": 3.23, "Water": 96.77},
            "provenance": "real_inference",
        },
        diagnostics={
            "scene_id": "S2A_46REN_20241228_0_L2A",
            "actual_processed_bbox": [93.8007, 24.5459, 93.8257, 24.5709],
            "actual_processed_tile_count": 2,
            "multi_tile_predictions": multi_tile_predictions,
            "module1_metadata": {
                "datetime": "2024-12-28T04:31:38.473000+00:00",
                "cloud_cover": 0.039,
                "stac_provider": "AWS Earth Search (Element84)",
            },
        },
    )

    # Replicate app.py line 934-940 logic
    bbox_coords = res.resolved_location.get("bbox", "N/A")
    actual_proc_bbox = res.diagnostics.get("actual_processed_bbox", bbox_coords)
    num_analyzed_tiles = res.diagnostics.get("actual_processed_tile_count") or len(res.diagnostics.get("multi_tile_predictions", [])) or (res.diagnostics.get("total_scene_tiles") or 1)
    tile_count_str = f"{num_analyzed_tiles} Tile{'s' if num_analyzed_tiles > 1 else ''} ({num_analyzed_tiles * 2.56:.2f} km² Analyzed)"
    coverage_label = f"Full AOI Coverage ({num_analyzed_tiles} Tiles)" if num_analyzed_tiles > 1 else "Canonical 1-Tile Coverage (2.56 km)"
    
    exec_ckpt_path = res.model_checkpoint or f"checkpoints/real_{res.task_type}/best_fusion_net.pth"
    exec_ckpt_name = os.path.basename(exec_ckpt_path) if exec_ckpt_path else "None"

    assert exec_ckpt_path == "checkpoints/real_waterbody/best_fusion_net.pth"
    assert exec_ckpt_name == "best_fusion_net.pth"
    assert coverage_label == "Full AOI Coverage (2 Tiles)"
    assert tile_count_str == "2 Tiles (5.12 km² Analyzed)"
    assert res.checkpoint_path == "checkpoints/real_waterbody/best_fusion_net.pth"


def test_streamlit_apptest_waterbody_and_forest_cover_rendering():
    """
    Integration Smoke Test using Streamlit AppTest:
    Simulates genuine Streamlit UI execution with pre-computed real inference results
    for both waterbody and forest-cover tasks, confirming that all result cards,
    metrics, expanders, and markdown elements render with zero exceptions.
    """
    from streamlit.testing.v1 import AppTest

    # 1. Test Waterbody Result Rendering
    water_res = VoxelPipelineResult(
        query="Track waterbody levels in Loktak Lake, Manipur",
        execution_mode="real_satellite",
        status="COMPLETE",
        is_synthetic=False,
        is_mock_prediction=False,
        data_source="Real Sentinel-2 L2A Scene (S2A_46REN_20241228_0_L2A) [Cached on Disk] via Element84 STAC",
        resolved_location={"name": "Loktak Lake, Manipur", "bbox": [93.8007, 24.5459, 93.8257, 24.5709]},
        contract={},
        task_type="waterbody",
        num_classes=2,
        class_list=["Non-water", "Water"],
        input_tensor_shape=[13, 256, 256],
        selected_bands=["B01", "B03", "B04", "B06", "B8A", "B11"],
        selected_indices=[0, 2, 3, 5, 8, 11],
        band_reduction_pct=53.8,
        mask_provenance="precomputed_mask",
        model_checkpoint="checkpoints/real_waterbody/best_fusion_net.pth",
        model_provenance="genuine_trained_model",
        prediction={
            "predicted_class_id": 1,
            "predicted_class_name": "Water",
            "confidence_pct": 96.77,
            "all_probabilities": {"Non-water": 3.23, "Water": 96.77},
            "provenance": "real_inference",
        },
        diagnostics={
            "scene_id": "S2A_46REN_20241228_0_L2A",
            "actual_processed_bbox": [93.8007, 24.5459, 93.8257, 24.5709],
            "actual_processed_tile_count": 1,
            "calibrated_tensor": np.ones((13, 256, 256), dtype=np.float32) * 0.1,
            "module1_metadata": {
                "datetime": "2024-12-28T04:31:38.473000+00:00",
                "cloud_cover": 0.039,
                "stac_provider": "AWS Earth Search (Element84)",
            },
        },
    )

    at_water = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
    at_water.session_state["active_page"] = "🔍 Satellite Analysis"
    at_water.session_state["top_nav_bar"] = "🔍 Satellite Analysis"
    at_water.session_state["task_profile_radio"] = "auto"
    at_water.session_state["active_task_mode"] = "auto"
    at_water.session_state["latest_pipeline_result"] = water_res
    at_water.session_state["selected_bbox"] = water_res.resolved_location.get("bbox")
    at_water.run(timeout=30)

    assert len(at_water.exception) == 0, f"Waterbody UI rendered with exception: {at_water.exception}"
    all_water_md = " ".join(m.value for m in at_water.markdown)
    assert "REAL SATELLITE PROVENANCE" in all_water_md
    assert "best_fusion_net.pth" in all_water_md

    # 2. Test Forest Cover Result Rendering
    forest_res = VoxelPipelineResult(
        query="Classify forest canopy cover in Jim Corbett, Uttarakhand",
        execution_mode="real_satellite",
        status="COMPLETE",
        is_synthetic=False,
        is_mock_prediction=False,
        data_source="Real Sentinel-2 L2A Scene (S2B_44RKT_20241211_0_L2A) [Cached on Disk] via Element84 STAC",
        resolved_location={"name": "Jim Corbett, Uttarakhand", "bbox": [78.83, 29.545, 78.855, 29.57]},
        contract={},
        task_type="forest_cover",
        num_classes=2,
        class_list=["Non-forest", "Forest"],
        input_tensor_shape=[13, 256, 256],
        selected_bands=["B02", "B04", "B06", "B11", "B12"],
        selected_indices=[1, 3, 5, 11, 12],
        band_reduction_pct=61.5,
        mask_provenance="precomputed_mask",
        model_checkpoint="checkpoints/real_forest_cover/best_fusion_net.pth",
        model_provenance="genuine_trained_model",
        prediction={
            "predicted_class_id": 1,
            "predicted_class_name": "Forest",
            "confidence_pct": 54.88,
            "all_probabilities": {"Non-forest": 45.12, "Forest": 54.88},
            "provenance": "real_inference",
        },
        diagnostics={
            "scene_id": "S2B_44RKT_20241211_0_L2A",
            "actual_processed_bbox": [78.83, 29.545, 78.855, 29.57],
            "actual_processed_tile_count": 1,
            "calibrated_tensor": np.ones((13, 256, 256), dtype=np.float32) * 0.2,
            "module1_metadata": {
                "datetime": "2024-12-11T05:30:51.361000+00:00",
                "cloud_cover": 0.0006,
                "stac_provider": "AWS Earth Search (Element84)",
            },
        },
    )

    at_forest = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
    at_forest.session_state["active_page"] = "🔍 Satellite Analysis"
    at_forest.session_state["top_nav_bar"] = "🔍 Satellite Analysis"
    at_forest.session_state["task_profile_radio"] = "auto"
    at_forest.session_state["active_task_mode"] = "auto"
    at_forest.session_state["latest_pipeline_result"] = forest_res
    at_forest.session_state["selected_bbox"] = forest_res.resolved_location.get("bbox")
    at_forest.run(timeout=30)

    assert len(at_forest.exception) == 0, f"Forest cover UI rendered with exception: {at_forest.exception}"
    all_forest_md = " ".join(m.value for m in at_forest.markdown)
    assert "REAL SATELLITE PROVENANCE" in all_forest_md
    assert "best_fusion_net.pth" in all_forest_md

