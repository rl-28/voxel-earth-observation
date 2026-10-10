"""
VOXEL Module 3: Build Explainability & Retrieval Artifacts (build_explainability.py)
Regenerates:
  1. checkpoints/shap_band_importance.png (Permutation SHAP spectral attribution)
  2. checkpoints/terrain_index.faiss (1216-dim spatial-spectral similarity index)
  3. checkpoints/terrain_index.faiss.meta.json (Scene metadata mapping)

Consumes real validation tiles via --manifest or synthetic mock tiles via --mock.
"""

import os
import sys
import argparse
from typing import Optional

import torch
from torch.utils.data import DataLoader

# Ensure Windows UTF-8 stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from inference import load_model
from utils.band_mask import load_band_mask
from data.datasets import create_split_datasets
from train_gpu import MockSentinel2Dataset
from explainability.shap_engine import run_explainability_from_loader
from explainability.faiss_search import build_faiss_index_from_dataloader


def build_all_explainability_artifacts(
    checkpoint_path: str = "checkpoints/best_fusion_net.pth",
    manifest_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    mask_path: Optional[str] = None,
    label_map_path: Optional[str] = None,
    output_dir: str = "checkpoints",
    batch_size: int = 8,
    device_name: str = "auto",
    mock: bool = False,
    task: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=" * 80)
    print(" [VOXEL MODULE 3] BUILD EXPLAINABILITY & VECTOR RETRIEVAL ARTIFACTS")
    print("=" * 80)
    print(f"Device:             {device.type.upper()}")
    print(f"Checkpoint Source:  {checkpoint_path}")
    print(f"Output Directory:   {output_dir}")

    # 1. Load model and registered band mask from checkpoint
    model, mask, device_obj = load_model(checkpoint_path, device=device.type)

    if mask_path is not None and os.path.exists(mask_path):
        print(f"Overriding mask from CLI path: {mask_path}")
        mask = load_band_mask(mask_path).to(device)

    active_bands = int(mask.sum().item())
    print(f"Active Channels:    {active_bands} spectral bands")

    # 2. Setup Validation Dataset & DataLoader
    num_classes = getattr(model, "num_classes", 10)
    if not mock and manifest_path and os.path.exists(manifest_path):
        print(f"Loading real satellite validation tiles via manifest: {manifest_path}")
        splits_save_path = os.path.join(output_dir, "splits.json")
        _, val_ds, _ = create_split_datasets(
            manifest_path=manifest_path,
            data_dir=data_dir,
            label_map_path=label_map_path,
            splits_save_path=splits_save_path,
            task=task,
            num_classes=num_classes,
        )
    else:
        print(f"Using synthetic Mock Sentinel-2 validation dataset (--mock, num_classes={num_classes})...")
        val_ds = MockSentinel2Dataset(num_samples=32, num_classes=num_classes, augment=False, seed=123)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Validation Tiles:   {len(val_ds)} scenes available for explanation & indexing")

    # 3. Generate Permutation SHAP Spectral Importance Plot
    print("-" * 80)
    shap_plot_path = os.path.join(output_dir, "shap_band_importance.png")
    print(f"[1/2] Computing Permutation SHAP band attributions -> {shap_plot_path}...")
    run_explainability_from_loader(
        model=model,
        dataloader=val_loader,
        mask=mask,
        output_plot=shap_plot_path,
        num_samples=min(16, len(val_ds)),
        num_permutations=10,
        device=device.type,
    )

    # 4. Generate FAISS 1216-dim Vector Database & Scene Metadata
    print("-" * 80)
    faiss_index_path = os.path.join(output_dir, "terrain_index.faiss")
    print(f"[2/2] Extracting 1216-dim embeddings & indexing terrain database -> {faiss_index_path}...")
    build_faiss_index_from_dataloader(
        model=model,
        dataloader=val_loader,
        mask=mask,
        device=device,
        index_save_path=faiss_index_path,
        dim=1216,
    )

    print("=" * 80)
    print(" [ARTIFACT GENERATION COMPLETE]")
    print(f"  1. SHAP Plot:     {shap_plot_path}")
    print(f"  2. FAISS Index:   {faiss_index_path}")
    print(f"  3. FAISS Meta:    {faiss_index_path}.meta.json")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Build VOXEL Module 3 Explainability & FAISS Artifacts")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_fusion_net.pth", help="Trained model checkpoint")
    parser.add_argument("--manifest", type=str, default=None, help="Satellite dataset manifest CSV")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory for satellite tiles")
    parser.add_argument("--mask_path", type=str, default=None, help="Path to spectral band mask")
    parser.add_argument("--label_map", type=str, default=None, help="Path to label map JSON")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Artifacts directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--mock", action="store_true", help="Force synthetic mock validation tiles")
    parser.add_argument("--task", type=str, default=None, help="Specialized task profile ('forest_cover', 'waterbody')")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_all_explainability_artifacts(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        data_dir=args.data_dir,
        mask_path=args.mask_path,
        label_map_path=args.label_map,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_name=args.device,
        mock=args.mock,
        task=args.task,
    )
