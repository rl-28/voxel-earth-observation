"""
VOXEL Fake Dataset Generator (tools/make_fake_dataset.py)
Generates on-disk synthetic Sentinel-2 tiles in .npy and multi-band GeoTIFF (.tif)
formats with a corresponding manifest.csv for end-to-end dataset pipeline validation.
The labels are deterministically tied to a spectral band pattern so training can learn it.
"""

import os
import sys

# Ensure repository root is in Python sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import argparse
import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from config.constants import NUM_BANDS, NUM_CLASSES, TILE_SIZE, LAND_COVER_CLASSES


def generate_fake_dataset(
    output_dir: str = "data/fake_dataset",
    num_samples: int = 40,
    seed: int = 42,
):
    """
    Generates synthetic on-disk tiles and manifest.csv.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    tiles_npy_dir = os.path.join(output_dir, "tiles_npy")
    tiles_tif_dir = os.path.join(output_dir, "tiles_tif")
    os.makedirs(tiles_npy_dir, exist_ok=True)
    os.makedirs(tiles_tif_dir, exist_ok=True)

    records = []
    print(f"[TOOLS] Generating {num_samples} fake Sentinel-2 tiles in '{output_dir}'...")

    for i in range(num_samples):
        # Target class in [0, NUM_CLASSES - 1]
        label = i % NUM_CLASSES
        class_name = LAND_COVER_CLASSES[label]

        # Generate base random reflectance in [0, 1]
        tile = np.random.uniform(0.1, 0.4, (NUM_BANDS, TILE_SIZE, TILE_SIZE)).astype(np.float32)

        # Learnable spectral pattern: Band 7 (NIR, index 7) has signal proportional to label
        signal_level = float((label + 1) / (NUM_CLASSES + 1))
        tile[7, :, :] = signal_level + np.random.normal(0, 0.02, (TILE_SIZE, TILE_SIZE)).astype(np.float32)
        tile = np.clip(tile, 0.0, 1.0)

        # Alternate between .npy (first half) and .tif (second half)
        use_npy = (i < num_samples // 2) or not HAS_RASTERIO

        # Stratified split allocation ensuring all classes exist proportionally across train, val, and test
        total_instances = max(3, num_samples // NUM_CLASSES)
        val_instances = max(1, int(total_instances * 0.15))
        test_instances = max(1, int(total_instances * 0.15))

        class_instance = i // NUM_CLASSES
        if class_instance < val_instances:
            split = "val"
        elif class_instance < (val_instances + test_instances):
            split = "test"
        else:
            split = "train"

        # Stratified location assignment ensuring regions never cross split boundaries
        val_cutoff = val_instances * NUM_CLASSES
        test_cutoff = (val_instances + test_instances) * NUM_CLASSES
        if split == "val":
            loc_idx = i // 4
        elif split == "test":
            loc_idx = 20 + (i - val_cutoff) // 4
        else:
            loc_idx = 40 + (i - test_cutoff) // 4
        location_id = f"LOC_REGION_{loc_idx:02d}"

        if use_npy:
            tile_filename = f"tile_{i:03d}.npy"
            full_tile_path = os.path.join(tiles_npy_dir, tile_filename)
            rel_tile_path = os.path.join("tiles_npy", tile_filename).replace("\\", "/")

            # Test channel-last on a subset to verify auto-detection
            if i % 4 == 0:
                save_tile = np.transpose(tile, (1, 2, 0))  # (256, 256, 13)
            else:
                save_tile = tile  # (13, 256, 256)

            np.save(full_tile_path, save_tile)

        else:
            tile_filename = f"tile_{i:03d}.tif"
            full_tile_path = os.path.join(tiles_tif_dir, tile_filename)
            rel_tile_path = os.path.join("tiles_tif", tile_filename).replace("\\", "/")

            # GeoTIFF transform (dummy georeferencing)
            transform = from_origin(72.8, 19.0, 10.0, 10.0)
            with rasterio.open(
                full_tile_path,
                "w",
                driver="GTiff",
                height=TILE_SIZE,
                width=TILE_SIZE,
                count=NUM_BANDS,
                dtype="float32",
                transform=transform,
            ) as dst:
                dst.write(tile)

        records.append(
            {
                "tile_path": rel_tile_path,
                "label": label,
                "class_name": class_name,
                "split": split,
                "location_id": location_id,
            }
        )

    # Save manifest.csv
    manifest_df = pd.DataFrame(records)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[TOOLS] Saved manifest.csv ({len(manifest_df)} records) to '{manifest_path}'.")

    # Save a sample label_map.json
    label_map = {class_name: idx for idx, class_name in enumerate(LAND_COVER_CLASSES)}
    label_map_path = os.path.join(output_dir, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"[TOOLS] Saved label_map.json to '{label_map_path}'.")

    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/fake_dataset")
    parser.add_argument("--num_samples", type=int, default=40)
    args = parser.parse_args()

    generate_fake_dataset(output_dir=args.output_dir, num_samples=args.num_samples)
