"""
VOXEL Module 3: Explainability & SHAP Engine
Implements Permutation SHAP channel attribution analysis on the trained FusionNet
to quantify and visualize feature importance across all active spectral bands.
"""

import os
import sys
from typing import Optional, Tuple, Dict, List, Any
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

# Ensure Windows UTF-8 stdout encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config.constants import SENTINEL2_BANDS, NUM_BANDS, NUM_CLASSES, LAND_COVER_CLASSES


def load_model_and_mask(
    checkpoint_path: str = "checkpoints/best_fusion_net.pth",
    device: Optional[str] = None,
):
    """Loads FusionNet model and registered band_mask dynamically from checkpoint."""
    from models.fusion_net import FusionNet

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    mask = checkpoint.get("band_mask", None)
    if mask is not None:
        mask = mask.to(device_obj)
        active_channels = checkpoint.get("selected_channels", int(mask.sum().item()))
    else:
        active_channels = checkpoint.get("selected_channels", 6)

    class_list = checkpoint.get("class_list", LAND_COVER_CLASSES)
    model = FusionNet(
        num_classes=len(class_list),
        total_bands=NUM_BANDS,
        selected_channels=active_channels,
        cnn_feature_dim=1024,
        vit_feature_dim=192,
    ).to(device_obj)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, mask, device_obj


def compute_permutation_shap_attributions(
    model: torch.nn.Module,
    sample_images: torch.Tensor,
    mask: torch.Tensor,
    num_permutations: int = 10,
    device: str = "cuda",
) -> dict:
    """
    Computes Permutation SHAP channel attribution scores across active spectral bands.
    
    Args:
        model: Trained FusionNet model in eval mode.
        sample_images: Input tensor (B, 13, H, W) on device.
        mask: 1D binary mask (13,) on device.
        num_permutations: Number of permutation trials per channel.
        device: 'cuda' or 'cpu'.
        
    Returns:
        dict: Mapping from band_index -> dict with 'name', 'score', and 'percentage'.
    """
    model.eval()
    sample_images = sample_images.to(device)
    mask = mask.to(device)

    # 1. Baseline model prediction
    with torch.no_grad():
        baseline_logits = model(sample_images, mask=mask)
        baseline_probs = torch.softmax(baseline_logits, dim=-1)
        pred_classes = torch.argmax(baseline_probs, dim=-1)  # (B,)
        baseline_conf = baseline_probs.gather(1, pred_classes.unsqueeze(1)).squeeze(1)  # (B,)

    active_indices = torch.nonzero(mask > 0.5, as_tuple=False).squeeze(-1).tolist()
    channel_importance = {}

    # 2. Permutation attribution per active channel
    for band_idx in active_indices:
        band_name = SENTINEL2_BANDS[band_idx]
        delta_scores = []

        for _ in range(num_permutations):
            # Clone input and permute the spatial pixels / tile values for this specific band
            permuted_images = sample_images.clone()
            b, c, h, w = permuted_images.shape
            
            # Shuffle spatial patches across the batch to destroy channel-specific feature correlations
            perm_idx = torch.randperm(b, device=device)
            permuted_images[:, band_idx, :, :] = sample_images[perm_idx, band_idx, :, :]

            with torch.no_grad():
                perm_logits = model(permuted_images, mask=mask)
                perm_probs = torch.softmax(perm_logits, dim=-1)
                perm_conf = perm_probs.gather(1, pred_classes.unsqueeze(1)).squeeze(1)

            # Attribution is the expected reduction in top-class confidence
            impact = (baseline_conf - perm_conf).clamp(min=0.0).cpu().numpy()
            delta_scores.append(np.mean(impact))

        raw_importance = float(np.mean(delta_scores)) if delta_scores else 0.001
        channel_importance[band_idx] = {
            "band_idx": band_idx,
            "name": band_name,
            "raw_score": max(raw_importance, 1e-4),
        }

    # 3. Normalize into relative percentages
    total_raw = sum(v["raw_score"] for v in channel_importance.values())
    for band_idx, data in channel_importance.items():
        data["percentage"] = (data["raw_score"] / total_raw) * 100.0

    return channel_importance


def plot_band_importance(
    channel_importance: dict,
    output_path: str = "checkpoints/shap_band_importance.png",
):
    """
    Renders and saves a publication-quality SHAP channel attribution bar chart.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    sorted_bands = sorted(channel_importance.values(), key=lambda x: x["percentage"], reverse=True)
    names = [b["name"] for b in sorted_bands]
    percentages = [b["percentage"] for b in sorted_bands]

    # Earth observation visual theme (blues, greens, deep cyans)
    colors = plt.cm.viridis(np.linspace(0.4, 0.85, len(names)))

    plt.figure(figsize=(9, 5.5), dpi=300)
    bars = plt.barh(names[::-1], percentages[::-1], color=colors, edgecolor="#2c3e50", height=0.6, alpha=0.9)

    plt.xlabel("SHAP Relative Importance Attribution (%)", fontsize=11, fontweight="bold", labelpad=8)
    plt.ylabel("Active Spectral Band", fontsize=11, fontweight="bold", labelpad=8)
    plt.title("VOXEL: Explainable AI Spectral Band Attribution (SHAP Analysis)\nFeature Importance of Shreeya's Metaheuristic Selected Bands", fontsize=12, fontweight="bold", pad=12)

    # Annotate percentage labels on each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#1a252f",
        )

    plt.xlim(0, max(percentages) + 8)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" [SHAP] Attribution plot saved successfully: {output_path}")


def run_explainability_pipeline(
    model: torch.nn.Module,
    sample_images: torch.Tensor,
    mask: torch.Tensor,
    output_plot: str = "checkpoints/shap_band_importance.png",
    device: str = "cuda",
) -> dict:
    """
    High-level entry point to compute attributions and save plot.
    """
    print("-" * 75)
    print(" [SHAP] Executing Permutation SHAP Channel Attribution Analysis...")
    attributions = compute_permutation_shap_attributions(
        model=model,
        sample_images=sample_images,
        mask=mask,
        num_permutations=10,
        device=device,
    )

    print(f"{'Band Index':<12} | {'Spectral Band Name':<22} | {'SHAP Attribution (%)'}")
    print("-" * 75)
    for band_idx, info in sorted(attributions.items(), key=lambda x: x[1]["percentage"], reverse=True):
        print(f"Band {band_idx:<7} | {info['name']:<22} | {info['percentage']:.2f}%")

    plot_band_importance(attributions, output_path=output_plot)
    return attributions


def run_explainability_from_loader(
    model: torch.nn.Module,
    dataloader: Any,
    mask: torch.Tensor,
    output_plot: str = "checkpoints/shap_band_importance.png",
    num_samples: int = 16,
    num_permutations: int = 10,
    device: str = "cuda",
) -> dict:
    """
    Computes Permutation SHAP channel attribution scores using real validation tiles from a DataLoader.
    """
    sample_images_list = []
    collected = 0
    for imgs, _ in dataloader:
        sample_images_list.append(imgs)
        collected += imgs.size(0)
        if collected >= num_samples:
            break

    if not sample_images_list:
        raise ValueError("DataLoader is empty, cannot compute SHAP attributions.")

    sample_images = torch.cat(sample_images_list, dim=0)[:num_samples]
    return run_explainability_pipeline(
        model=model,
        sample_images=sample_images,
        mask=mask,
        output_plot=output_plot,
        device=device,
    )


if __name__ == "__main__":
    import argparse
    from data.datasets import create_split_datasets
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="VOXEL Module 3: SHAP Spectral Attribution")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_fusion_net.pth", help="Checkpoint path")
    parser.add_argument("--manifest", type=str, default=None, help="Manifest CSV path")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--mask_path", type=str, default=None, help="Mask path")
    parser.add_argument("--output_plot", type=str, default="checkpoints/shap_band_importance.png", help="Output PNG")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")
    args = parser.parse_args()

    device_str = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    model, mask, dev = load_model_and_mask(args.checkpoint, device=device_str)

    if args.manifest and os.path.exists(args.manifest):
        _, val_ds, _ = create_split_datasets(manifest_path=args.manifest, data_dir=args.data_dir)
        loader = DataLoader(val_ds, batch_size=8, shuffle=False)
        run_explainability_from_loader(
            model=model,
            dataloader=loader,
            mask=mask,
            output_plot=args.output_plot,
            device=device_str,
        )
    else:
        print("[SHAP] Manifest not provided. Please provide --manifest <path> to run SHAP on satellite validation tiles.")
