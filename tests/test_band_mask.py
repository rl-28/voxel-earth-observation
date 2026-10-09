"""
Unit and Integration Tests for Band Mask Loader and Variable Mask Sizes in FusionNet.
Validates Step 2 of Module 3 merge preparation.
"""

import os
import tempfile
import json
import pytest
import numpy as np
import torch

from config.constants import NUM_BANDS, NUM_CLASSES, EMBEDDING_DIM
from utils.band_mask import (
    default_band_mask,
    describe_band_mask,
    load_band_mask,
    validate_band_mask,
)
from models.fusion_net import FusionNet


def test_default_band_mask():
    """Verify default placeholder band mask loads with correct shape and warning."""
    with pytest.warns(UserWarning, match="Using default placeholder band mask"):
        mask = default_band_mask()
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (NUM_BANDS,)
    assert mask.dtype == torch.float32
    assert int(mask.sum().item()) == 6
    desc = describe_band_mask(mask)
    assert len(desc) == 6


def test_load_band_mask_json():
    """Verify loading from standard JSON file."""
    mask_path = "config/example_band_mask.json"
    assert os.path.exists(mask_path)
    mask = load_band_mask(mask_path)
    assert mask.shape == (NUM_BANDS,)
    assert int(mask.sum().item()) == 6
    active_indices = torch.nonzero(mask > 0.5, as_tuple=False).squeeze(-1).tolist()
    assert active_indices == [1, 2, 3, 7, 11, 12]


def test_load_band_mask_formats():
    """Verify loading across .json, .npy, and .pt formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Custom JSON format
        json_path = os.path.join(tmpdir, "custom_mask.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"selected_indices": [0, 4, 8]}, f)
        m_json = load_band_mask(json_path)
        assert int(m_json.sum().item()) == 3

        # 2. NumPy array
        npy_path = os.path.join(tmpdir, "mask.npy")
        npy_arr = np.zeros(NUM_BANDS, dtype=np.float32)
        npy_arr[[1, 3, 5, 7]] = 1.0
        np.save(npy_path, npy_arr)
        m_npy = load_band_mask(npy_path)
        assert int(m_npy.sum().item()) == 4

        # 3. PyTorch file
        pt_path = os.path.join(tmpdir, "mask.pt")
        pt_tensor = torch.zeros(NUM_BANDS, dtype=torch.float32)
        pt_tensor[[0, 1, 2, 3, 4]] = 1.0
        torch.save({"band_mask": pt_tensor}, pt_path)
        m_pt = load_band_mask(pt_path)
        assert int(m_pt.sum().item()) == 5


def test_validate_band_mask_errors():
    """Verify strict validation catches bad lengths, non-binary values, and empty masks."""
    # Bad length
    with pytest.raises(ValueError, match="Band mask length must be exactly 13"):
        validate_band_mask(torch.zeros(10))

    # Non-binary value
    with pytest.raises(ValueError, match="must contain only binary values"):
        bad_vals = torch.zeros(NUM_BANDS)
        bad_vals[0] = 0.5
        validate_band_mask(bad_vals)

    # Empty mask (0 bands active)
    with pytest.raises(ValueError, match="Band mask is empty"):
        validate_band_mask(torch.zeros(NUM_BANDS))


@pytest.mark.parametrize("num_selected", [3, 6, 9, 13])
def test_fusion_net_variable_mask_sizes(num_selected):
    """
    Verify FusionNet executes cleanly for variable mask sizes (3, 6, 9, 13 bands)
    while strictly preserving the 1216-dimensional embedding and 10-class head.
    """
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate custom mask with `num_selected` active bands
    mask_arr = np.zeros(NUM_BANDS, dtype=np.float32)
    selected_indices = list(range(num_selected))
    mask_arr[selected_indices] = 1.0
    mask = torch.tensor(mask_arr, dtype=torch.float32, device=device)

    # Instantiate FusionNet configured for the active band count
    model = FusionNet(
        num_classes=NUM_CLASSES,
        total_bands=NUM_BANDS,
        selected_channels=num_selected,
    ).to(device)

    # Mock 13-band Sentinel-2 tile batch
    x = torch.randn(batch_size, NUM_BANDS, 256, 256, device=device)

    # Forward pass
    logits, embedding = model(x, mask=mask, return_embedding=True)

    # Assert exact dimensional invariants
    assert logits.shape == (batch_size, NUM_CLASSES), f"Logits shape {logits.shape} != ({batch_size}, {NUM_CLASSES})"
    assert embedding.shape == (batch_size, EMBEDDING_DIM), f"Embedding shape {embedding.shape} != ({batch_size}, {EMBEDDING_DIM})"

    # Verify backward pass / gradient flow
    loss = logits.sum()
    loss.backward()

    # Check gradients exist
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grads, "Gradients failed to propagate during backward pass"
