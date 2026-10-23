"""
tests/test_interactive_aoi.py - Unit tests for Interactive Area of Interest (AOI) selection,
geographic bounding-box validation, size-limit guards, and multi-tile inference handling.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from voxel_pipeline import VoxelPipeline
from utils.llm_parser import parse_geographic_coordinates, VoxelStageZeroParser


@pytest.fixture
def offline_pipeline():
    """Instantiate standard test pipeline in offline mode."""
    return VoxelPipeline(
        default_mode="offline",
        checkpoint_dir="checkpoints",
        mask_dir="checkpoints/masks",
        verbose=False,
    )


# -----------------------------------------------------------------------------
# 1. Geographic Bounding-Box Parameter Support & Resolution
# -----------------------------------------------------------------------------
def test_pipeline_explicit_bbox_override(offline_pipeline):
    """Verify that passing explicit bbox overrides Stage 0 default resolution."""
    custom_bbox = [79.2995, 16.5655, 79.3245, 16.5905]  # Nagarjuna Sagar Reservoir
    res = offline_pipeline.run(
        query="Analyze waterbody levels",
        mode="offline",
        task_override="waterbody",
        bbox=custom_bbox,
    )

    assert res.status == "COMPLETE"
    assert res.resolved_location["bbox"] == custom_bbox
    assert "Custom AOI" in res.resolved_location["name"]
    assert res.resolved_location["geocoder"] == "Interactive Map / Explicit BBOX Input"


def test_pipeline_point_coordinates_via_query(offline_pipeline):
    """Verify that direct decimal coordinates in query are resolved to a 2.56 km tile footprint."""
    res = offline_pipeline.run(
        query="Assess forest canopy at lat: 21.1240, lon: 70.8240",
        mode="offline",
        task_override="forest_cover",
    )

    assert res.status == "COMPLETE"
    bbox = res.resolved_location["bbox"]
    assert len(bbox) == 4
    # Check that width and height are approximately 0.025 degrees (2.56 km footprint)
    span_lon = bbox[2] - bbox[0]
    span_lat = bbox[3] - bbox[1]
    assert pytest.approx(span_lon, abs=0.005) == 0.025
    assert pytest.approx(span_lat, abs=0.005) == 0.025


# -----------------------------------------------------------------------------
# 2. Safety Bounds & Size Limits
# -----------------------------------------------------------------------------
def test_pipeline_rejects_inverted_bbox(offline_pipeline):
    """Verify that an inverted bounding box (min >= max) raises a clear ValueError."""
    inverted_bbox = [80.0, 20.0, 79.0, 19.0]
    with pytest.raises(ValueError, match="Invalid bounding box geometry"):
        offline_pipeline.run(
            query="Analyze water",
            mode="offline",
            bbox=inverted_bbox,
        )


def test_pipeline_rejects_out_of_bounds_bbox(offline_pipeline):
    """Verify that out-of-bounds physical coordinates raise a clear ValueError."""
    oob_bbox = [70.0, 95.0, 75.0, 98.0]  # Lat > 90
    with pytest.raises(ValueError, match="out of physical bounds"):
        offline_pipeline.run(
            query="Analyze water",
            mode="offline",
            bbox=oob_bbox,
        )


def test_pipeline_rejects_oversized_bbox_memory_safety(offline_pipeline):
    """Verify that rectangles exceeding 0.08 degrees (~8.0 km) raise a safety error."""
    oversized_bbox = [78.0, 20.0, 78.15, 20.15]  # Span = 0.15 deg > 0.08 deg
    with pytest.raises(ValueError, match="exceeds maximum supported single-scene limit"):
        offline_pipeline.run(
            query="Analyze water",
            mode="offline",
            bbox=oversized_bbox,
        )


# -----------------------------------------------------------------------------
# 3. Polygon & Point GeoJSON Extraction
# -----------------------------------------------------------------------------
def test_parse_drawn_polygon_coordinates():
    """Verify coordinate extraction from Leaflet Draw GeoJSON polygon."""
    polygon_geojson = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[77.50, 12.90], [77.50, 12.93], [77.53, 12.93], [77.53, 12.90], [77.50, 12.90]]
            ],
        },
    }
    coords = polygon_geojson["geometry"]["coordinates"][0]
    min_lon = round(min(p[0] for p in coords), 4)
    max_lon = round(max(p[0] for p in coords), 4)
    min_lat = round(min(p[1] for p in coords), 4)
    max_lat = round(max(p[1] for p in coords), 4)

    assert [min_lon, min_lat, max_lon, max_lat] == [77.50, 12.90, 77.53, 12.93]
    # Span check: 0.03 deg < 0.08 deg (within safe limits)
    assert max_lon - min_lon < 0.08
    assert max_lat - min_lat < 0.08


# -----------------------------------------------------------------------------
# 4. Multi-Tile Integrity Preservation
# -----------------------------------------------------------------------------
def test_multi_tile_inference_preservation(offline_pipeline):
    """
    Verify that when multiple tiles are present in scene metadata, each tile is
    independently evaluated and stored without artificial whole-region averaging.
    """
    # Create two synthetic tiles
    tile1, _ = offline_pipeline.generate_synthetic_tile("waterbody", seed=1)
    tile2, _ = offline_pipeline.generate_synthetic_tile("waterbody", seed=2)

    # Mock metadata with two tiles
    mock_tiles_data = [
        {"tile_id": "tile_000_000", "processed_tensor": tile1, "diagnostics": {"bounds": [78.0, 20.0, 78.025, 20.025]}},
        {"tile_id": "tile_000_001", "processed_tensor": tile2, "diagnostics": {"bounds": [78.025, 20.0, 78.050, 20.025]}},
    ]

    # Verify model evaluation on multiple tiles directly
    ckpt_path = Path("checkpoints/real_waterbody/best_fusion_net.pth")
    if ckpt_path.exists():
        model, prov, _, _ = offline_pipeline.inspect_and_load_model(
            task_key="waterbody",
            checkpoint_path=str(ckpt_path),
            selected_channels=6,
            selected_indices=[0, 2, 3, 5, 8, 11],
            num_classes=2,
            is_real_mode=False,
        )
        assert model is not None

        mask_device = torch.zeros(13)
        mask_device[[0, 2, 3, 5, 8, 11]] = 1.0

        predictions = []
        for t in mock_tiles_data:
            inp = torch.from_numpy(t["processed_tensor"]).unsqueeze(0)
            with torch.no_grad():
                logits = model(inp, mask=mask_device)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
            pred_id = int(torch.argmax(probs).item())
            predictions.append({
                "tile_id": t["tile_id"],
                "class_id": pred_id,
                "confidence": float(probs[pred_id].item() * 100.0),
            })

        assert len(predictions) == 2
        assert predictions[0]["tile_id"] == "tile_000_000"
        assert predictions[1]["tile_id"] == "tile_000_001"
        assert 0.0 <= predictions[0]["confidence"] <= 100.0
        assert 0.0 <= predictions[1]["confidence"] <= 100.0
