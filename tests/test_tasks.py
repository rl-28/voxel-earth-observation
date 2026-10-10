import os
import tempfile
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pytest

from config.constants import (
    TASKS,
    LAND_COVER_CLASSES,
    NUM_CLASSES,
    get_task_classes,
    get_task_num_classes,
)
from models.fusion_net import FusionNet
from data.datasets import RealSentinel2Dataset
from inference import load_model, predict_and_search


def test_task_constants_and_defaults():
    # 10-class default must remain intact
    assert len(LAND_COVER_CLASSES) == 10
    assert NUM_CLASSES == 10

    # Binary tasks configured
    assert "forest_cover" in TASKS
    assert TASKS["forest_cover"]["classes"] == ["Non-forest", "Forest"]
    assert get_task_num_classes("forest_cover") == 2
    assert get_task_classes("forest_cover") == ["Non-forest", "Forest"]

    assert "waterbody" in TASKS
    assert TASKS["waterbody"]["classes"] == ["Non-water", "Water"]
    assert get_task_num_classes("waterbody") == 2
    assert get_task_classes("waterbody") == ["Non-water", "Water"]

    # Drought and rainfall stubs marked "label source undefined"
    assert "drought" in TASKS
    assert "label source undefined" in str(TASKS["drought"]).lower()
    assert "rainfall" in TASKS
    assert "label source undefined" in str(TASKS["rainfall"]).lower()


def test_fusion_net_2class_forward_backward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionNet(
        num_classes=2,
        total_bands=13,
        selected_channels=6,
        cnn_feature_dim=1024,
        vit_feature_dim=192,
    ).to(device)
    model.train()

    # Input: batch of 2 tiles with 13 bands
    batch_size = 2
    inputs = torch.randn(batch_size, 13, 256, 256, device=device)
    mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=torch.float32, device=device)
    targets = torch.tensor([0, 1], dtype=torch.long, device=device)

    logits, embedding = model(inputs, mask=mask, return_embedding=True)

    assert logits.shape == (batch_size, 2), f"Expected logits shape (2, 2), got {logits.shape}"
    assert embedding.shape == (batch_size, 1216), f"Expected embedding shape (2, 1216), got {embedding.shape}"

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    loss.backward()

    # Verify gradients computed
    has_grad = False
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            break
    assert has_grad, "Expected backward pass to compute gradients on model parameters"


def test_checkpoint_roundtrip_2class():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionNet(
        num_classes=2,
        total_bands=13,
        selected_channels=6,
    ).to(device)
    model.eval()

    inputs = torch.randn(1, 13, 256, 256, device=device)
    mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=torch.float32, device=device)

    with torch.no_grad():
        original_logits, _ = model(inputs, mask=mask, return_embedding=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_2class.pth")
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "band_mask": mask,
                "selected_channels": 6,
                "task": "forest_cover",
                "num_classes": 2,
                "class_list": ["Non-forest", "Forest"],
            },
            ckpt_path,
        )

        # Load back with load_model
        loaded_model, loaded_mask, loaded_device = load_model(checkpoint_path=ckpt_path, device=str(device))
        assert loaded_model.classifier[-1].out_features == 2

        with torch.no_grad():
            loaded_logits, _ = loaded_model(inputs, mask=loaded_mask, return_embedding=True)

        assert torch.allclose(original_logits, loaded_logits, atol=1e-5), "Logits do not match after checkpoint round-trip"


def test_dataset_2class_mapping():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock tile
        tile_path = os.path.join(tmpdir, "mock_tile.npy")
        np.save(tile_path, np.zeros((13, 256, 256), dtype=np.float32))

        df = pd.DataFrame([
            {"tile_path": tile_path, "label": "Non-forest", "split": "train"},
            {"tile_path": tile_path, "label": "Forest", "split": "train"},
            {"tile_path": tile_path, "label": 0, "split": "val"},
            {"tile_path": tile_path, "label": 1, "split": "val"},
        ])

        ds = RealSentinel2Dataset(
            manifest_df=df,
            data_dir=tmpdir,
            task="forest_cover",
            num_classes=2,
            class_list=["Non-forest", "Forest"],
        )

        assert len(ds) == 4
        _, y0 = ds[0]
        _, y1 = ds[1]
        _, y2 = ds[2]
        _, y3 = ds[3]
        assert y0.item() == 0
        assert y1.item() == 1
        assert y2.item() == 0
        assert y3.item() == 1
