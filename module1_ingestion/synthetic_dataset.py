"""
Phase 2 Synthetic Dataset Generator.

Generates mock spatial-spectral tiles to establish a development baseline
for downstream modules. Ensures every generated tile strictly satisfies the
spatial-spectral data contract.
"""

import json
import numpy as np
from pathlib import Path

from .config import (
    TILE_SIZE,
    NUM_BANDS,
    OUTPUT_DTYPE,
    EXPECTED_SHAPE,
    SYNTHETIC_DATA_DIR
)
from .data_contract import validate_spatial_spectral_tensor

def generate_synthetic_tile(seed=None):
    """
    Generates a single synthetic spatial-spectral tile.
    
    Args:
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        np.ndarray: A float32 NumPy array of shape (256, 256, 13)
                    that passes the Module 1 data contract.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Generate structured synthetic data:
    # A combination of smooth background variation and local noise.
    
    # 1. Base variation (spatial gradients)
    x = np.linspace(-1, 1, TILE_SIZE)
    y = np.linspace(-1, 1, TILE_SIZE)
    xx, yy = np.meshgrid(x, y)
    
    # Base gradient pattern for the tile
    base_pattern = np.sin(xx * 3.14) * np.cos(yy * 3.14)
    
    # 2. Spectral variation (simulate band correlations)
    # We create a random spectral signature vector for the base pattern
    spectral_signature = np.random.randn(NUM_BANDS).astype(OUTPUT_DTYPE)
    
    # Broadcast to shape (256, 256, 13)
    base_tensor = base_pattern[..., np.newaxis] * spectral_signature
    
    # 3. Add high-frequency local noise (e.g., representing fine details)
    noise = np.random.randn(*EXPECTED_SHAPE) * 0.1
    
    tile = (base_tensor + noise).astype(OUTPUT_DTYPE)
    
    # Validate the generated tile against the contract
    validate_spatial_spectral_tensor(tile)
    
    return tile

def generate_synthetic_dataset(num_tiles=100, seed=42, output_dir=SYNTHETIC_DATA_DIR):
    """
    Generates a dataset of synthetic spatial-spectral tiles and metadata.
    
    Args:
        num_tiles (int): Number of tiles to generate.
        seed (int): Base random seed.
        output_dir (Path or str): Output directory for the dataset.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set the base seed
    np.random.seed(seed)
    # Generate unique seeds for each tile to ensure variety while keeping reproducibility
    tile_seeds = np.random.randint(0, 1000000, size=num_tiles)
    
    for i in range(num_tiles):
        tile_seed = tile_seeds[i]
        tile = generate_synthetic_tile(seed=tile_seed)
        
        # Format filename like tile_000.npy, tile_001.npy, etc.
        filename = f"tile_{i:03d}.npy"
        filepath = output_path / filename
        
        # Save tile and release reference
        np.save(filepath, tile)
        del tile
        
    # Generate metadata
    metadata = {
        "dataset_name": "voxel_synthetic_baseline",
        "dataset_type": "synthetic",
        "num_tiles": num_tiles,
        "tile_height": EXPECTED_SHAPE[0],
        "tile_width": EXPECTED_SHAPE[1],
        "num_bands": EXPECTED_SHAPE[2],
        "dtype": str(OUTPUT_DTYPE),
        "axis_order": ["height", "width", "spectral_bands"],
        "random_seed": seed,
        "description": "Synthetic 100-tile spatial-spectral dataset for downstream development."
    }
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Generated {num_tiles} synthetic tiles in {output_path}")

if __name__ == "__main__":
    generate_synthetic_dataset()
