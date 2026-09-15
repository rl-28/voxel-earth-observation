"""
Tests for the synthetic dataset generator.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from module1_ingestion.synthetic_dataset import (
    generate_synthetic_tile,
    generate_synthetic_dataset
)
from module1_ingestion.config import EXPECTED_SHAPE, OUTPUT_DTYPE

def test_generate_synthetic_tile_shape_dtype_finiteness():
    """Verify one generated tile has correct shape, dtype, and finite values."""
    tile = generate_synthetic_tile(seed=123)
    
    assert tile.shape == EXPECTED_SHAPE
    assert tile.dtype == OUTPUT_DTYPE
    assert np.isfinite(tile).all()
    # Contract validation happens implicitly inside generate_synthetic_tile

def test_generate_synthetic_tile_reproducibility():
    """Verify the same seed produces reproducible output."""
    tile1 = generate_synthetic_tile(seed=42)
    tile2 = generate_synthetic_tile(seed=42)
    
    np.testing.assert_array_equal(tile1, tile2)

def test_generate_synthetic_tile_variation():
    """Verify different seeds produce different output."""
    tile1 = generate_synthetic_tile(seed=42)
    tile2 = generate_synthetic_tile(seed=43)
    
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(tile1, tile2)

def test_generate_synthetic_dataset_integration():
    """
    Verify the dataset generator creates the expected number of files,
    uses correct filenames, and writes valid metadata.
    """
    # Use a temporary directory to avoid generating large artifacts in real data dir
    with tempfile.TemporaryDirectory() as tmpdir:
        num_tiles = 3
        generate_synthetic_dataset(num_tiles=num_tiles, seed=10, output_dir=tmpdir)
        
        # Check files
        tmp_path = Path(tmpdir)
        npy_files = list(tmp_path.glob("tile_*.npy"))
        assert len(npy_files) == num_tiles
        
        # Check specific filename conventions
        assert (tmp_path / "tile_000.npy").exists()
        assert (tmp_path / "tile_001.npy").exists()
        assert (tmp_path / "tile_002.npy").exists()
        
        # Check metadata
        metadata_file = tmp_path / "metadata.json"
        assert metadata_file.exists()
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
        assert metadata["num_tiles"] == num_tiles
        assert metadata["dataset_type"] == "synthetic"
        assert metadata["random_seed"] == 10
        assert metadata["dtype"] == str(OUTPUT_DTYPE)
        
        # Load one tile to ensure it's valid
        tile = np.load(tmp_path / "tile_000.npy")
        assert tile.shape == EXPECTED_SHAPE
        assert tile.dtype == OUTPUT_DTYPE
