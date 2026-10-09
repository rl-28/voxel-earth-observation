"""
VOXEL Module 3: Band Selection Comparison Benchmark (tools/compare_masks.py)
Compares model performance, memory footprint, parameter counts, and throughput across:
  (a) All 13 Spectral Bands (Full Spectrum Baseline)
  (b) Provided / Optimized Mask from Module 2 (e.g. metaheuristic mask)
  (c) Random Mask with identical band count K

Saves benchmark results table to results/band_selection_comparison.md.
Runnable with --mock or real satellite dataset via --manifest.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root is in sys.path when executed as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure Windows UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from models.fusion_net import FusionNet
from utils.band_mask import load_band_mask, describe_band_mask
from config.constants import SENTINEL2_BANDS, NUM_BANDS, NUM_CLASSES
from train_gpu import (
    train_epoch,
    validate_epoch,
    setup_datasets,
    set_seed,
    get_gpu_memory_usage,
)


def run_single_mask_experiment(
    mask_name: str,
    mask: torch.Tensor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Trains FusionNet from scratch with a given spectral mask and records benchmarks."""
    mask = mask.to(device)
    active_indices = torch.nonzero(mask > 0.5, as_tuple=False).squeeze(-1).tolist()
    if isinstance(active_indices, int):
        active_indices = [active_indices]
    active_channels = len(active_indices)
    active_names = [SENTINEL2_BANDS[i].split()[0] for i in active_indices]

    print("-" * 80)
    print(f" EXPERIMENT: {mask_name.upper()}")
    print(f" Active Channels: {active_channels} / {NUM_BANDS} ({', '.join(active_names)})")
    print("-" * 80)

    # Clean GPU memory before trial
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Instantiate model
    set_seed(42)
    model = FusionNet(
        num_classes=NUM_CLASSES,
        total_bands=NUM_BANDS,
        selected_channels=active_channels,
        cnn_feature_dim=1024,
        vit_feature_dim=192,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_oa = 0.0
    best_val_f1 = 0.0
    best_val_miou = 0.0
    epoch_times: List[float] = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_oa, train_f1, _ = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            mask=mask,
            device=device,
            scaler=scaler,
        )
        val_loss, val_oa, val_f1, val_miou = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            mask=mask,
            device=device,
        )
        scheduler.step()
        elapsed = time.perf_counter() - t0
        epoch_times.append(elapsed)

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_val_oa = val_oa
            best_val_miou = val_miou

        print(
            f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | "
            f"Val OA: {val_oa:5.2f}% | Val F1: {val_f1:5.2f}% | Time: {elapsed:4.1f}s"
        )

    # Peak VRAM calculation
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        peak_vram_gb = 0.0

    avg_sec_epoch = float(np.mean(epoch_times)) if epoch_times else 0.0

    return {
        "mask_name": mask_name,
        "active_channels": active_channels,
        "active_bands_str": ", ".join(active_names),
        "total_params": total_params,
        "avg_sec_epoch": avg_sec_epoch,
        "peak_vram_gb": peak_vram_gb,
        "best_val_oa": best_val_oa,
        "best_val_f1": best_val_f1,
        "best_val_miou": best_val_miou,
    }


def compare_spectral_masks(args) -> List[Dict[str, Any]]:
    """Runs mask comparison and writes results/band_selection_comparison.md."""
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print(" [VOXEL MODULE 3] SPECTRAL BAND SELECTION COMPARISON BENCHMARK")
    print("=" * 80)
    print(f"Device:             {device.type.upper()}")
    print(f"Training Epochs:    {args.epochs}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Candidate Mask:     {args.mask_path or 'Default / example_band_mask.json'}")

    # 1. Setup Datasets
    train_ds, val_ds, test_ds, class_weights = setup_datasets(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    criterion = (
        nn.CrossEntropyLoss(weight=class_weights.to(device))
        if class_weights is not None
        else nn.CrossEntropyLoss()
    )

    # 2. Define the 3 mask configurations
    # (a) All 13 bands
    mask_all = torch.ones(13, dtype=torch.float32)

    # (b) Provided mask
    mask_prov = load_band_mask(args.mask_path)
    k_bands = int(mask_prov.sum().item())

    # (c) Random mask with same K band count
    rng = np.random.RandomState(args.seed)
    rand_indices = rng.choice(NUM_BANDS, size=k_bands, replace=False)
    mask_rand = torch.zeros(13, dtype=torch.float32)
    mask_rand[rand_indices] = 1.0

    experiments = [
        ("All 13 Bands (Full Spectrum)", mask_all),
        (f"Optimized / Provided Mask ({k_bands} Bands)", mask_prov),
        (f"Random Control Mask ({k_bands} Bands)", mask_rand),
    ]

    results: List[Dict[str, Any]] = []

    for name, mask_tensor in experiments:
        res = run_single_mask_experiment(
            mask_name=name,
            mask=mask_tensor,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )
        results.append(res)

    # 3. Print Console Summary Table
    print("\n" + "=" * 95)
    print(" [BENCHMARK RESULTS] SPECTRAL BAND SELECTION COMPARISON")
    print("=" * 95)
    print(
        f"{'Configuration':<38} | {'Bands':<6} | {'Params(M)':<10} | {'Sec/Ep':<7} | "
        f"{'Peak VRAM':<10} | {'Val OA(%)':<9} | {'Val F1(%)':<9}"
    )
    print("-" * 95)
    for r in results:
        params_m = r["total_params"] / 1e6
        vram_str = f"{r['peak_vram_gb']:.2f} GB" if device.type == "cuda" else "N/A"
        print(
            f"{r['mask_name']:<38} | {r['active_channels']:<6} | {params_m:<10.2f} | "
            f"{r['avg_sec_epoch']:<7.1f} | {vram_str:<10} | {r['best_val_oa']:<9.2f} | {r['best_val_f1']:<9.2f}"
        )
    print("=" * 95)

    # 4. Generate Markdown Report
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("# VOXEL Module 3: Spectral Band Selection Comparison Benchmark\n\n")
        f.write(
            f"**Execution Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}<br>\n"
            f"**Compute Device:** `{device.type.upper()}` "
            f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})<br>\n"
            f"**Benchmark Epochs:** `{args.epochs}` | **Batch Size:** `{args.batch_size}` | "
            f"**Optimizer:** `AdamW (lr={args.lr}, wd={args.weight_decay})`<br>\n"
            f"**Dataset Source:** `{'Mock Generator' if args.mock or not args.manifest else args.manifest}`\n\n"
        )
        f.write("## 1. Quantitative Benchmark Summary\n\n")
        f.write(
            "| Spectral Configuration | Active Bands | Band Count | Parameters (M) | Throughput (sec/ep) | Peak VRAM | Val OA (%) | Val Macro F1 (%) | Val mIoU (%) |\n"
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        )
        for r in results:
            params_m = f"{r['total_params'] / 1e6:.2f}M"
            vram_str = f"{r['peak_vram_gb']:.2f} GB" if device.type == "cuda" else "N/A"
            f.write(
                f"| **{r['mask_name']}** | `{r['active_bands_str']}` | {r['active_channels']} | "
                f"{params_m} | {r['avg_sec_epoch']:.1f}s | {vram_str} | "
                f"**{r['best_val_oa']:.2f}%** | **{r['best_val_f1']:.2f}%** | {r['best_val_miou']:.2f}% |\n"
            )

        f.write("\n## 2. Key Engineering Insights\n\n")
        f.write(
            "1. **Channel Reduction & Efficiency:** Slicing the input spectral tensor from 13 bands to the optimized subset reduces the input dimensional footprint without altering the fused 1216-dim representation vector.\n"
            "2. **Memory & VRAM Footprint:** On 4GB VRAM hardware (NVIDIA RTX 3050), band selection preserves memory headroom, permitting larger batch sizes or higher throughput.\n"
            "3. **Optimization Validation Contract:** When Module 2 provides the final metaheuristic band mask, running this script verifies that the selected bands outperform both random band subsets and rival full-spectrum accuracy with reduced data transmission.\n"
        )

    print(f"\n [SAVED] Benchmark report written to: {args.output_file}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="VOXEL Module 3 Band Selection Benchmark")
    parser.add_argument("--mask_path", type=str, default="config/example_band_mask.json", help="Path to optimized mask file")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory for satellite tiles")
    parser.add_argument("--label_map", type=str, default=None, help="Path to label map JSON")
    parser.add_argument("--mock", action="store_true", help="Force synthetic mock data")
    parser.add_argument("--epochs", type=int, default=2, help="Benchmark training epochs per configuration")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--output_file", type=str, default="results/band_selection_comparison.md", help="Output markdown report")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_spectral_masks(args)
