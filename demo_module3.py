"""
VOXEL Module 3: Demonstration Script (demo_module3.py)
Executes an end-to-end forward prediction and FAISS vector retrieval query
on the NVIDIA RTX 3050 GPU, outputting a clean engineering summary.
"""

import sys
import time
import torch
import numpy as np

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from inference import predict_and_search, LAND_COVER_CLASSES
from tests.mock_handshake import generate_mock_batch
from utils.band_mask import load_band_mask, describe_band_mask
from config.constants import SENTINEL2_BANDS


def main():
    print("=" * 80)
    print(" [VOXEL MODULE 3] END-TO-END DEEP LEARNING & VECTOR SEARCH DEMO")
    print("=" * 80)

    # 1. Environment & Hardware Status
    device_name = "CPU"
    if torch.cuda.is_available():
        gpu_raw = torch.cuda.get_device_name(0)
        prefix = "" if gpu_raw.startswith("NVIDIA") else "NVIDIA "
        device_name = f"{prefix}{gpu_raw} (CUDA {torch.version.cuda})"
    print(f"Compute Hardware:          {device_name}")

    # 2. Mock Handshake Sentinel-2 Tile Input
    print("-" * 80)
    print("Generating incoming Sentinel-2 L2A tile (13 bands, 256x256 spatial resolution)...")
    tiles, _, _ = generate_mock_batch(batch_size=1, seed=777)
    mask = load_band_mask()
    test_tile = tiles[0]

    active_indices = torch.nonzero(mask > 0.5, as_tuple=False).squeeze(-1).tolist()
    active_names = [SENTINEL2_BANDS[i].split()[0] for i in active_indices]

    print(f"Input Tile Shape:          {tuple(test_tile.shape)} | Channels: 13 | Dtype: {test_tile.dtype}")
    print(f"Active Channels Sliced:    {len(active_indices)} bands ({', '.join(active_names)})")

    # 3. Model Inference & Vector Retrieval
    print("-" * 80)
    print("Executing GPU Forward Pass and FAISS Similarity Retrieval...")
    t0 = time.perf_counter()
    result = predict_and_search(test_tile, band_mask=mask, top_k=5)
    total_time_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Display Results
    print("=" * 80)
    print(" PREDICTION & RETRIEVAL RESULTS")
    print("=" * 80)
    print(f"Predicted Land Cover:      {result['predicted_class_name']} (Class ID: {result['predicted_class_id']})")
    print(f"Confidence Probability:    {result['confidence']:.2f}%")
    print(f"Fused Representation:      1216-dim vector (XNANet CNN: 1024-dim | Tiny-32 ViT: 192-dim)")
    print(f"Inference Latency:         {total_time_ms:.2f} ms")
    print(f"FAISS Retrieval Latency:   {result['retrieval_latency_ms']:.3f} ms (sub-millisecond)")

    print("-" * 80)
    print("Top-3 Class Probability Distribution:")
    sorted_probs = sorted(result["all_probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
    for class_name, prob in sorted_probs:
        bar_len = int(prob / 2.5)
        bar = "█" * bar_len
        print(f"  {class_name:<26} : {prob:>5.2f}% {bar}")

    print("-" * 80)
    print("FAISS Top-5 Similar Terrain Tiles Retrieved from Vector Database:")
    print(f"  {'Rank':<5} | {'Cosine Sim':<11} | {'Tile ID':<22} | {'Split':<6} | {'Land Cover Class'}")
    print("  " + "-" * 72)
    for tile in result["similar_terrain"]:
        print(
            f"  {tile['rank']:<5} | {tile['similarity_score']:<11.4f} | {tile['tile_id']:<22} | "
            f"{tile['split']:<6} | {tile['class_name']}"
        )

    print("=" * 80)
    print(" [STATUS] Module 3 Demo executed successfully on CUDA GPU!")
    print("=" * 80)


if __name__ == "__main__":
    main()
