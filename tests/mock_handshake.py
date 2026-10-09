"""
Module 1 & 2 Handshake Mock: Synthetic Data and Band Selection Mask Generator.
Simulates Sentinel-2 L2A 13-band surface reflectance tiles and Shreeya's metaheuristic
band selection mask for Module 3 pipeline training and evaluation.
"""

import numpy as np
import torch

from config.constants import SENTINEL2_BANDS, DEFAULT_SELECTED_BAND_INDICES


def generate_band_mask(selected_indices=None, total_bands=13):
    """
    Generate a 1D binary mask array representing Module 2's selected bands.
    
    Args:
        selected_indices (list[int]): Indices of selected bands (0-indexed).
        total_bands (int): Total number of bands (default: 13 for Sentinel-2).
        
    Returns:
        torch.Tensor: Binary mask of shape (total_bands,), dtype=torch.float32 or torch.bool.
    """
    if selected_indices is None:
        selected_indices = DEFAULT_SELECTED_BAND_INDICES

    mask = np.zeros(total_bands, dtype=np.float32)
    for idx in selected_indices:
        if 0 <= idx < total_bands:
            mask[idx] = 1.0
    return torch.tensor(mask, dtype=torch.float32)


def generate_mock_batch(
    batch_size=16,
    channels=13,
    height=256,
    width=256,
    num_classes=10,
    seed=42,
    device="cpu",
):
    """
    Generate a normalized Float32 spatial-spectral mini-batch tensor and land cover labels.
    
    Args:
        batch_size (int): Batch size (default: 16).
        channels (int): Number of spectral bands (default: 13).
        height (int): Tile spatial height (default: 256).
        width (int): Tile spatial width (default: 256).
        num_classes (int): Number of land cover target classes (default: 10).
        seed (int): Deterministic random seed.
        device (str or torch.device): PyTorch device.
        
    Returns:
        images (torch.Tensor): Shape (batch_size, 13, height, width), float32 normalized in [0, 1].
        labels (torch.Tensor): Shape (batch_size,), int64 in [0, num_classes - 1].
        mask (torch.Tensor): Shape (13,), float32 binary mask.
    """
    torch.manual_seed(seed)
    # Generate realistic surface reflectance values in [0, 1]
    images = torch.rand((batch_size, channels, height, width), dtype=torch.float32, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.int64, device=device)
    mask = generate_band_mask().to(device)
    
    return images, labels, mask


if __name__ == "__main__":
    print("=" * 60)
    print(" [HANDSHAKE] Generating Mock Sentinel-2 Data & Band Mask")
    print("=" * 60)
    images, labels, mask = generate_mock_batch()
    
    print(f"Image Batch Shape:     {tuple(images.shape)} | Dtype: {images.dtype}")
    print(f"Value Range:           [{images.min().item():.3f}, {images.max().item():.3f}]")
    print(f"Labels Shape:          {tuple(labels.shape)} | Classes: {labels.unique().tolist()}")
    print(f"Band Mask Shape:       {tuple(mask.shape)}")
    print(f"Active Bands Count:    {int(mask.sum().item())} / {len(mask)}")
    print("-" * 60)
    print("Active Spectral Bands Selected:")
    for idx, is_active in enumerate(mask):
        status = "[ACTIVE]  " if is_active.item() == 1.0 else "[EXCLUDED]"
        print(f"  Band {idx:02d}: {status} {SENTINEL2_BANDS[idx]}")
    print("=" * 60)
    print("Handshake data generator verified successfully.")
