"""
Unit and Integration Tests for tools/validate_integration.py.
Validates the Integration Contract Validator against fake datasets, edge cases,
failure modes, and advisory warning conditions.
"""

import os
import sys
import gc
import json
import tempfile
import subprocess
import pytest
import pandas as pd
import numpy as np
import torch

from tools.validate_integration import validate_integration


@pytest.fixture(scope="module")
def fake_dataset_paths():
    """Ensures synthetic fake dataset and example mask exist."""
    manifest_path = "data/fake_dataset/manifest.csv"
    mask_path = "config/example_band_mask.json"
    if not os.path.exists(manifest_path):
        from tools.make_fake_dataset import generate_fake_dataset
        generate_fake_dataset(output_dir="data/fake_dataset", num_samples=40)
    assert os.path.exists(manifest_path), f"Manifest missing: {manifest_path}"
    assert os.path.exists(mask_path), f"Mask missing: {mask_path}"
    return manifest_path, mask_path


def test_validate_integration_fake_dataset_passes(fake_dataset_paths):
    """Verifies that the reference fake dataset and example mask pass all contract checks."""
    manifest_path, mask_path = fake_dataset_paths
    results = validate_integration(
        manifest_path=manifest_path,
        mask_path=mask_path,
        sample_size=10,
    )

    assert results["overall_ready"] is True, f"Expected READY, got checks: {results['checks']}"
    for check in results["checks"]:
        assert check["status"] in ["PASS", "WARN"], f"Check failed: {check}"

    assert results["manifest_res"]["status"] == "PASS"
    assert results["tiles_res"]["status"] == "PASS"
    assert results["spectral_res"]["status"] in ["PASS", "WARN"]
    assert results["labels_res"]["status"] in ["PASS", "WARN"]
    assert results["spatial_res"]["status"] == "PASS"
    assert results["mask_res"]["status"] == "PASS"
    assert results["model_res"]["status"] == "PASS"

    # Verify requirement 3: says "ready for integration with Module 3" and does NOT claim branch merge into main
    assert "ready for integration with Module 3" in results["teammate_message"]
    assert "READY FOR MERGE into main" not in results["teammate_message"]
    # Verify no duplicated phrase in spatial summary line
    assert "Zero spatial leakage across splits (Zero spatial leakage" not in results["teammate_message"]


@pytest.mark.parametrize("bad_bands", [12, 14])
def test_validate_integration_fails_wrong_band_count(bad_bands, fake_dataset_paths):
    """(a) Verifies that tiles with wrong band count (12 or 14 bands) cause validation to FAIL."""
    manifest_path, mask_path = fake_dataset_paths

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tile_path = os.path.join(tmpdir, f"tile_bands_{bad_bands}.npy")
        np.save(tile_path, np.random.uniform(0.1, 0.4, (bad_bands, 256, 256)).astype(np.float32))

        bad_manifest = os.path.join(tmpdir, "manifest_bad_bands.csv")
        pd.DataFrame([{"tile_path": tile_path, "label": 0}]).to_csv(bad_manifest, index=False)

        results = validate_integration(
            manifest_path=bad_manifest,
            mask_path=mask_path,
            sample_size=1,
        )

        assert results["overall_ready"] is False, f"Expected NOT READY for tile with {bad_bands} bands"
        tiles_check = [c for c in results["checks"] if "Tiles" in c["check"]][0]
        assert tiles_check["status"] == "FAIL"

        details_lower = tiles_check["details"].lower()
        # Must name the problem
        assert "band count" in details_lower or str(bad_bands) in details_lower or "13 bands" in details_lower
        assert "REQUIRED FIXES" in results["teammate_message"]
        gc.collect()


@pytest.mark.parametrize("bad_label", [99, "AlienSurfaceTerrain"])
def test_validate_integration_fails_unknown_label(bad_label, fake_dataset_paths):
    """(b) Verifies that unknown label ID or name causes validation to FAIL."""
    manifest_path, mask_path = fake_dataset_paths
    df = pd.read_csv(manifest_path).copy()
    df["label"] = df["label"].astype(object)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        bad_manifest = os.path.join(tmpdir, "manifest_bad_label.csv")
        df.loc[0, "label"] = bad_label
        df.to_csv(bad_manifest, index=False)

        results = validate_integration(
            manifest_path=bad_manifest,
            mask_path=mask_path,
            data_dir="data/fake_dataset",
            sample_size=2,
        )

        assert results["overall_ready"] is False, f"Expected NOT READY for unknown label {bad_label}"
        labels_check = [c for c in results["checks"] if "Class Labels" in c["check"]][0]
        assert labels_check["status"] == "FAIL"

        details_lower = labels_check["details"].lower()
        # Must name the problem
        assert "cannot be mapped" in details_lower or str(bad_label).lower() in details_lower
        assert "REQUIRED FIXES" in results["teammate_message"]
        gc.collect()


@pytest.mark.parametrize("bad_len", [5, 10, 15])
def test_validate_integration_fails_mask_wrong_length(bad_len, fake_dataset_paths):
    """(c) Verifies that a mask of wrong length causes validation to FAIL."""
    manifest_path, _ = fake_dataset_paths

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        bad_mask_path = os.path.join(tmpdir, f"mask_{bad_len}.json")
        with open(bad_mask_path, "w", encoding="utf-8") as f:
            json.dump([1] * bad_len, f)

        results = validate_integration(
            manifest_path=manifest_path,
            mask_path=bad_mask_path,
            sample_size=2,
        )

        assert results["overall_ready"] is False, f"Expected NOT READY for mask length {bad_len}"
        mask_check = [c for c in results["checks"] if "Mask Specification" in c["check"]][0]
        assert mask_check["status"] == "FAIL"

        details_lower = mask_check["details"].lower()
        # Must name the problem
        assert "length" in details_lower or "13" in details_lower
        assert "REQUIRED FIXES" in results["teammate_message"]
        gc.collect()


def test_validate_integration_fails_non_binary_mask(fake_dataset_paths):
    """(d) Verifies that a non-binary mask causes validation to FAIL."""
    manifest_path, _ = fake_dataset_paths

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        bad_mask_path = os.path.join(tmpdir, "mask_non_binary.json")
        # 13 elements, but contains non-binary float 0.5
        bad_mask = [0, 1, 0.5, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        with open(bad_mask_path, "w", encoding="utf-8") as f:
            json.dump(bad_mask, f)

        results = validate_integration(
            manifest_path=manifest_path,
            mask_path=bad_mask_path,
            sample_size=2,
        )

        assert results["overall_ready"] is False, "Expected NOT READY for non-binary mask"
        mask_check = [c for c in results["checks"] if "Mask Specification" in c["check"]][0]
        assert mask_check["status"] == "FAIL"

        details_lower = mask_check["details"].lower()
        # Must name the problem
        assert "binary" in details_lower or "0.5" in details_lower
        assert "REQUIRED FIXES" in results["teammate_message"]
        gc.collect()


def test_validate_integration_warns_on_suspicious_band_order(fake_dataset_paths):
    """Verifies that nearly identical band means or inverted B08/B04 emits a WARNING without a hard FAIL."""
    manifest_path, mask_path = fake_dataset_paths

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        flat_tile_path = os.path.join(tmpdir, "flat_tile.npy")
        # All 13 bands have flat 0.25 value -> nearly identical band means
        np.save(flat_tile_path, np.full((13, 256, 256), 0.25, dtype=np.float32))

        flat_manifest = os.path.join(tmpdir, "flat_manifest.csv")
        # Generate 20 records across 10 classes so that batch size > 1 and all classes exist
        records = [{"tile_path": flat_tile_path, "label": i % 10} for i in range(20)]
        pd.DataFrame(records).to_csv(flat_manifest, index=False)

        results = validate_integration(
            manifest_path=flat_manifest,
            mask_path=mask_path,
            sample_size=4,
        )

        # Should be an advisory warning, NOT a hard fail
        spectral_check = [c for c in results["checks"] if "Spectral Bands" in c["check"]][0]
        assert spectral_check["status"] == "WARN"
        assert "Suspicious" in spectral_check["details"] or "nearly identical" in spectral_check["details"]
        # Advisory warning does not block readiness if everything else passes
        assert results["overall_ready"] is True
        gc.collect()


def test_validate_integration_detects_spatial_leakage(fake_dataset_paths):
    """Verifies that spatial data leakage across train and val splits is flagged as FAIL."""
    manifest_path, mask_path = fake_dataset_paths
    df = pd.read_csv(manifest_path).copy()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        leak_manifest = os.path.join(tmpdir, "manifest_leak.csv")
        # Force tile 0 (val) and tile 400 (train) to share the same location ID
        shared_loc = df.loc[df["split"] == "train", "location_id"].iloc[0]
        val_idx = df[df["split"] == "val"].index[0]
        df.loc[val_idx, "location_id"] = shared_loc
        df.to_csv(leak_manifest, index=False)

        results = validate_integration(
            manifest_path=leak_manifest,
            mask_path=mask_path,
            data_dir="data/fake_dataset",
            sample_size=4,
        )

        assert results["overall_ready"] is False
        spatial_check = [c for c in results["checks"] if c["check"] == "Spatial Leakage (location_id)"][0]
        assert spatial_check["status"] == "FAIL"
        assert "Spatial data leakage detected" in spatial_check["details"]
        assert "REQUIRED FIXES" in results["teammate_message"]
        gc.collect()


def test_validate_integration_detects_missing_tile_files(fake_dataset_paths):
    """Verifies that non-existent tile paths in the manifest are caught and flagged."""
    _, mask_path = fake_dataset_paths

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        fake_manifest = os.path.join(tmpdir, "manifest_missing.csv")
        bad_df = pd.DataFrame([
            {"tile_path": "non_existent_folder/tile_ghost.npy", "label": 0}
        ])
        bad_df.to_csv(fake_manifest, index=False)

        results = validate_integration(
            manifest_path=fake_manifest,
            mask_path=mask_path,
            sample_size=1,
        )

        assert results["overall_ready"] is False
        manifest_check = [c for c in results["checks"] if c["check"] == "Manifest & Disk Existence"][0]
        assert manifest_check["status"] == "FAIL"
        assert "do not exist on disk" in manifest_check["details"]
        gc.collect()


def test_validate_integration_cli_execution():
    """Runs tools/validate_integration.py via Python CLI subprocess and asserts returncode 0."""
    cmd = [
        sys.executable,
        "tools/validate_integration.py",
        "--manifest",
        "data/fake_dataset/manifest.csv",
        "--mask_path",
        "config/example_band_mask.json",
        "--sample",
        "5",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI exited with {proc.returncode}. Stderr: {proc.stderr}"
    assert "READY FOR INTEGRATION WITH MODULE 3" in proc.stdout
    assert "ready for integration with Module 3" in proc.stdout
    assert "READY FOR MERGE into main" not in proc.stdout
