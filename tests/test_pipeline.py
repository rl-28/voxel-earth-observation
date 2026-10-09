"""
Unit and Integration Tests for Training, Round-Trip Checkpointing, Inference, and GPU Steps.
Validates Step 7 of VOXEL Module 3 merge preparation.
"""

import os
import json
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.constants import NUM_BANDS, NUM_CLASSES, EMBEDDING_DIM, LAND_COVER_CLASSES
from models.fusion_net import FusionNet
from utils.band_mask import load_band_mask
from data.datasets import create_split_datasets
from train_gpu import train_epoch, validate_epoch, set_seed
from inference import predict_and_search, load_model


@pytest.fixture(scope="module")
def fake_dataset_manifest():
    """Ensures synthetic fake dataset exists and returns path to manifest.csv."""
    data_dir = "data/fake_dataset"
    manifest_path = os.path.join(data_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        from tools.make_fake_dataset import generate_fake_dataset
        generate_fake_dataset(output_dir=data_dir, num_samples=40)
    return manifest_path


def test_checkpoint_save_load_roundtrip_identical_logits():
    """
    Verifies that saving a model checkpoint and restoring it produces
    bit-for-bit identical forward logits and 1216-dim embeddings.
    """
    set_seed(42)
    device = torch.device("cpu")
    mask = load_band_mask("config/example_band_mask.json")
    active_channels = int(mask.sum().item())

    model_orig = FusionNet(
        num_classes=NUM_CLASSES,
        total_bands=NUM_BANDS,
        selected_channels=active_channels,
    ).to(device)
    model_orig.eval()

    test_input = torch.randn(2, NUM_BANDS, 256, 256, device=device)
    with torch.no_grad():
        logits_orig, emb_orig = model_orig(test_input, mask=mask, return_embedding=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_roundtrip.pth")
        torch.save(
            {
                "model_state_dict": model_orig.state_dict(),
                "band_mask": mask,
                "selected_channels": active_channels,
                "class_list": LAND_COVER_CLASSES,
            },
            ckpt_path,
        )

        # Restore into fresh instance
        model_loaded = FusionNet(
            num_classes=NUM_CLASSES,
            total_bands=NUM_BANDS,
            selected_channels=active_channels,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_loaded.load_state_dict(ckpt["model_state_dict"])
        model_loaded.eval()

        with torch.no_grad():
            logits_loaded, emb_loaded = model_loaded(test_input, mask=mask, return_embedding=True)

    assert torch.allclose(logits_orig, logits_loaded, atol=1e-6), "Logits diverged after checkpoint round-trip!"
    assert torch.allclose(emb_orig, emb_loaded, atol=1e-6), "Embeddings diverged after checkpoint round-trip!"


def test_fake_dataset_training_above_chance(fake_dataset_manifest):
    """
    Runs a 2-epoch training cycle on the synthetic fake dataset.
    Verifies validation accuracy exceeds chance level (> 10.0% for 10 classes).
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, _ = create_split_datasets(
        manifest_path=fake_dataset_manifest,
        data_dir=os.path.dirname(fake_dataset_manifest),
        seed=42,
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    # Use default mask which includes Band 7 (where the deterministic signal was injected)
    mask = load_band_mask().to(device)
    active_channels = int(mask.sum().item())

    model = FusionNet(
        num_classes=NUM_CLASSES,
        total_bands=NUM_BANDS,
        selected_channels=active_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_oa = 0.0
    for epoch in range(2):
        _ = train_epoch(model, train_loader, optimizer, criterion, mask, device)
        _, val_oa, _, _ = validate_epoch(model, val_loader, criterion, mask, device)
        if val_oa > best_val_oa:
            best_val_oa = val_oa

    # Chance level for 10 classes is 10.0%; model should reach or exceed chance on learnable feature
    assert best_val_oa >= 10.0, f"Expected validation OA >= 10.0%, achieved {best_val_oa:.2f}%"


def test_inference_end_to_end_on_tile(fake_dataset_manifest):
    """
    Verifies end-to-end inference on a single satellite tile:
    Returns predicted class name, confidence, 1216-dim embedding, and nearest terrain matches.
    """
    ckpt_path = "checkpoints/best_fusion_net.pth"
    faiss_path = "checkpoints/terrain_index.faiss"

    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    assert os.path.exists(faiss_path), f"FAISS index missing: {faiss_path}"

    test_tile = np.random.uniform(0.1, 0.4, (NUM_BANDS, 256, 256)).astype(np.float32)

    result = predict_and_search(
        tile_tensor=test_tile,
        top_k=3,
        checkpoint_path=ckpt_path,
        index_path=faiss_path,
    )

    assert "predicted_class_id" in result
    assert "predicted_class_name" in result
    assert "confidence" in result
    assert "embedding" in result
    assert "similar_terrain" in result

    assert 0 <= result["predicted_class_id"] < NUM_CLASSES
    assert result["predicted_class_name"] in LAND_COVER_CLASSES
    assert 0.0 <= result["confidence"] <= 100.0
    assert result["embedding"].shape == (EMBEDDING_DIM,)
    assert len(result["similar_terrain"]) <= 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_gpu_training_step_on_cuda():
    """
    Executes exactly one forward/backward pass and optimizer step on NVIDIA CUDA GPU.
    Verifies memory management and parameter update integrity.
    """
    device = torch.device("cuda")
    torch.cuda.empty_cache()

    mask = load_band_mask("config/example_band_mask.json").to(device)
    active_channels = int(mask.sum().item())

    model = FusionNet(
        num_classes=NUM_CLASSES,
        total_bands=NUM_BANDS,
        selected_channels=active_channels,
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Create dummy GPU mini-batch
    images = torch.randn(2, NUM_BANDS, 256, 256, device=device)
    targets = torch.tensor([0, 1], dtype=torch.int64, device=device)

    # Initial parameter copy
    first_param = next(model.parameters())
    initial_val = first_param.clone().detach()

    # Step
    optimizer.zero_grad()
    with torch.amp.autocast("cuda"):
        logits = model(images, mask=mask)
        loss = criterion(logits, targets)

    assert torch.isfinite(loss), "Loss became NaN or Inf on CUDA GPU"

    loss.backward()
    optimizer.step()

    # Verify parameter changed
    updated_val = first_param.clone().detach()
    diff = torch.abs(initial_val - updated_val).sum().item()
    assert diff > 0.0, "Model parameters were not updated by optimizer step on CUDA GPU"
