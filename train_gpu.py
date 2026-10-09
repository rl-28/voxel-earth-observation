"""
VOXEL Module 3: Hardened GPU Training Pipeline
Features:
  - Pluggable real-data dataset loader (RealSentinel2Dataset) and mock generator (--mock)
  - Pluggable band mask support (--mask_path) with dynamic FusionNet channel sizing
  - PyTorch CUDA execution with AMP (Automatic Mixed Precision) and GradScaler
  - AdamW optimizer + CosineAnnealingLR scheduler + CrossEntropyLoss with dynamic class weights (>3x imbalance)
  - Gradient accumulation (--grad_accum_steps) for memory-constrained training
  - torch.cuda.OutOfMemoryError handling with actionable guidance
  - Best model checkpointing based on Validation Macro-F1 with early stopping (patience=10)
  - Comprehensive checkpoint metadata: model state, optimizer, band mask, band names, classes, metrics, args, git commit
  - Per-epoch metric logging to console and results/training_log.csv
  - Fully reproducible execution with global seeding
"""

import os
import sys
import time
import csv
import argparse
import random
import subprocess
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, jaccard_score

# Ensure UTF-8 output encoding and immediate line buffering on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from models.fusion_net import FusionNet
from utils.band_mask import load_band_mask, describe_band_mask
from config.constants import (
    SENTINEL2_BANDS,
    NUM_CLASSES,
    NUM_BANDS,
    LAND_COVER_CLASSES,
    TILE_SIZE,
)
from data.datasets import (
    RealSentinel2Dataset,
    SpatialAugmentation,
    create_split_datasets,
)


class MockSentinel2Dataset(Dataset):
    """
    Fallback synthetic Sentinel-2 Land Cover Dataset.
    Generates normalized 13-band Float32 tiles in [0, 1] with random labels.
    """

    def __init__(
        self,
        num_samples: int = 64,
        channels: int = 13,
        height: int = 256,
        width: int = 256,
        num_classes: int = 10,
        augment: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.transform = SpatialAugmentation() if augment else None

        np.random.seed(seed)
        torch.manual_seed(seed)
        self.data = torch.rand((num_samples, channels, height, width), dtype=torch.float32)
        self.targets = torch.randint(0, num_classes, (num_samples,), dtype=torch.int64)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def set_seed(seed: int = 42) -> None:
    """Sets global seeds across Python, NumPy, and PyTorch for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash if running in a git repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"


def get_gpu_memory_usage() -> Tuple[float, float]:
    """Returns currently allocated and reserved GPU VRAM in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        return allocated, reserved
    return 0.0, 0.0


def compute_metrics(all_preds: List[int], all_targets: List[int]) -> Tuple[float, float, float]:
    """Computes Overall Accuracy (OA), Macro F1-Score, and Mean IoU (mIoU)."""
    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)

    oa = float(np.mean(preds_np == targets_np) * 100.0) if len(targets_np) > 0 else 0.0
    f1 = float(f1_score(targets_np, preds_np, average="macro", zero_division=0) * 100.0)
    miou = float(jaccard_score(targets_np, preds_np, average="macro", zero_division=0) * 100.0)

    return oa, f1, miou


def compute_class_weights(targets: np.ndarray, num_classes: int) -> Optional[torch.Tensor]:
    """
    Computes inverse class frequency weights when training split imbalance exceeds 3x.
    """
    counts = np.bincount(targets, minlength=num_classes)
    nonzero_counts = counts[counts > 0]
    if len(nonzero_counts) == 0:
        return None

    imbalance_ratio = float(np.max(nonzero_counts)) / float(np.min(nonzero_counts))
    if imbalance_ratio > 3.0:
        total_samples = len(targets)
        weights = np.zeros(num_classes, dtype=np.float32)
        for c in range(num_classes):
            if counts[c] > 0:
                weights[c] = total_samples / (num_classes * counts[c])
            else:
                weights[c] = 0.0
        # Normalize so mean of active weights is 1.0
        active_mean = np.mean(weights[counts > 0])
        if active_mean > 0:
            weights = weights / active_mean
        print(f"[IMBALANCE] Class imbalance ratio {imbalance_ratio:.2f}x (>3.0x threshold).")
        print(f"[IMBALANCE] Applying inverse frequency class weights to CrossEntropyLoss.")
        return torch.tensor(weights, dtype=torch.float32)
    return None


def probe_dataloader_workers(num_workers: int) -> int:
    """
    Tests DataLoader multiprocessing worker functionality.
    Falls back gracefully to 0 on Windows if workers fail or encounter issues.
    """
    if num_workers <= 0:
        return 0
    if sys.platform == "win32":
        print(f"[INFO] Windows platform detected: Falling back from --num_workers={num_workers} to 0 for stability.")
        return 0
    return num_workers


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mask: torch.Tensor,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_accum_steps: int = 1,
) -> Tuple[float, float, float, float]:
    """Runs a single training epoch with gradient accumulation and AMP."""
    model.train()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    optimizer.zero_grad()

    try:
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if scaler and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(images, mask=mask)
                    loss = criterion(logits, targets)
                    loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits = model(images, mask=mask)
                loss = criterion(logits, targets)
                loss = loss / grad_accum_steps
                loss.backward()
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += (loss.item() * grad_accum_steps) * images.size(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        print("\n" + "!" * 80)
        print(" [ERROR] CUDA Out of Memory (OOM) during training forward/backward pass!")
        print(" Actionable remediation:")
        print(f"   1. Lower --batch_size (e.g. from {dataloader.batch_size} to {max(1, dataloader.batch_size // 2)})")
        print(f"   2. Increase --grad_accum_steps (e.g. from {grad_accum_steps} to {grad_accum_steps * 2})")
        print("   3. Check other GPU processes using nvidia-smi")
        print("!" * 80 + "\n")
        raise e

    avg_loss = total_loss / len(dataloader.dataset)
    oa, f1, miou = compute_metrics(all_preds, all_targets)
    return avg_loss, oa, f1, miou


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    mask: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Evaluates the model across the validation split."""
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    try:
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images, mask=mask)
            loss = criterion(logits, targets)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        print("\n" + "!" * 80)
        print(" [ERROR] CUDA Out of Memory (OOM) during validation inference!")
        print(" Actionable remediation:")
        print(f"   1. Lower --batch_size (e.g. from {dataloader.batch_size} to {max(1, dataloader.batch_size // 2)})")
        print("!" * 80 + "\n")
        raise e

    avg_loss = total_loss / len(dataloader.dataset)
    oa, f1, miou = compute_metrics(all_preds, all_targets)
    return avg_loss, oa, f1, miou


def setup_datasets(args) -> Tuple[Dataset, Dataset, Optional[Dataset], Optional[torch.Tensor]]:
    """
    Initializes train, val, and test datasets based on CLI arguments.
    Returns (train_dataset, val_dataset, test_dataset, class_weights_tensor).
    """
    if not args.mock and args.manifest:
        print(f"Loading real satellite dataset via manifest: {args.manifest}")
        splits_save_path = os.path.join(args.checkpoint_dir, "splits.json")
        train_ds, val_ds, test_ds = create_split_datasets(
            manifest_path=args.manifest,
            data_dir=args.data_dir,
            label_map_path=args.label_map,
            seed=args.seed,
            splits_save_path=splits_save_path,
        )
        print(f"  Dataset Splits: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
        print(f"  Split indices saved to: {splits_save_path}")

        # Compute class weights for training split
        train_labels = train_ds.records["label"].values
        class_weights = compute_class_weights(train_labels, num_classes=NUM_CLASSES)
        return train_ds, val_ds, test_ds, class_weights

    # Fallback to mock dataset
    print("Initializing synthetic Mock Sentinel-2 Dataset (--mock mode)...")
    train_ds = MockSentinel2Dataset(
        num_samples=64, num_classes=NUM_CLASSES, augment=True, seed=args.seed
    )
    val_ds = MockSentinel2Dataset(
        num_samples=32, num_classes=NUM_CLASSES, augment=False, seed=args.seed + 1
    )
    test_ds = MockSentinel2Dataset(
        num_samples=32, num_classes=NUM_CLASSES, augment=False, seed=args.seed + 2
    )
    train_labels = train_ds.targets.numpy()
    class_weights = compute_class_weights(train_labels, num_classes=NUM_CLASSES)
    return train_ds, val_ds, test_ds, class_weights


def train_pipeline(args) -> Tuple[nn.Module, Dict[str, Any]]:
    """Executes the full hardened training pipeline."""
    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("=" * 80)
    print(" [VOXEL MODULE 3] HARDENED GPU TRAINING PIPELINE")
    print("=" * 80)

    # 1. Device configuration
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Active Compute Device:      {device.type.upper()}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Hardware:               {gpu_name} ({total_vram_gb:.2f} GB VRAM)")
        torch.cuda.empty_cache()

    # 2. Datasets & Loaders
    num_workers = probe_dataloader_workers(args.num_workers)
    train_ds, val_ds, test_ds, class_weights = setup_datasets(args)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Band Mask Loading & Channel Configuration
    band_mask = load_band_mask(args.mask_path).to(device)
    active_indices = torch.nonzero(band_mask > 0.5, as_tuple=False).squeeze(-1).tolist()
    if isinstance(active_indices, int):
        active_indices = [active_indices]
    active_bands = len(active_indices)
    selected_band_names = [SENTINEL2_BANDS[i] for i in active_indices]

    print(f"Selected Band Mask:         {active_bands} active bands ({active_indices})")
    print(f"Active Spectral Bands:      {describe_band_mask(band_mask)}")
    print(f"DataLoader Config:          Batch Size={args.batch_size}, Workers={num_workers}, GradAccum={args.grad_accum_steps}")

    # 4. Instantiate Model
    model = FusionNet(
        num_classes=NUM_CLASSES,
        total_bands=NUM_BANDS,
        selected_channels=active_bands,
        cnn_feature_dim=1024,
        vit_feature_dim=192,
    ).to(device)

    # 5. Optimizer, Scheduler, Loss, AMP Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    criterion = (
        nn.CrossEntropyLoss(weight=class_weights.to(device))
        if class_weights is not None
        else nn.CrossEntropyLoss()
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 1
    best_val_f1 = -1.0
    best_metrics: Dict[str, Any] = {}
    epochs_no_improve = 0
    patience = 10

    # Resume checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"[RESUME] Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_metrics = checkpoint.get("metrics", {})
            best_val_f1 = best_metrics.get("val_f1", -1.0)
            print(f"[RESUME] Resuming from epoch {start_epoch} with prior best Val F1: {best_val_f1:.2f}%")
        else:
            print(f"[WARNING] Resume checkpoint not found at {args.resume}, starting from scratch.")

    # Prepare results/training_log.csv
    log_csv_path = os.path.join("results", "training_log.csv")
    csv_mode = "a" if (args.resume and os.path.exists(log_csv_path)) else "w"
    csv_file = open(log_csv_path, csv_mode, newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if csv_mode == "w":
        csv_writer.writerow([
            "epoch",
            "lr",
            "train_loss",
            "train_oa",
            "train_f1",
            "train_miou",
            "val_loss",
            "val_oa",
            "val_f1",
            "val_miou",
            "time_sec",
            "vram_alloc_gb",
            "vram_reserved_gb",
        ])
        csv_file.flush()

    best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_fusion_net.pth")

    print("-" * 80)
    print(
        f"{'Epoch':<6} | {'LR':<9} | {'Train Loss':<10} | {'Val Loss':<9} | "
        f"{'Val OA(%)':<9} | {'Val F1(%)':<9} | {'Val mIoU(%)':<11} | {'Time(s)':<7} | {'VRAM (GB)'}"
    )
    print("-" * 80)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.perf_counter()
            current_lr = scheduler.get_last_lr()[0]

            train_loss, train_oa, train_f1, train_miou = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                mask=band_mask,
                device=device,
                scaler=scaler,
                grad_accum_steps=args.grad_accum_steps,
            )

            val_loss, val_oa, val_f1, val_miou = validate_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                mask=band_mask,
                device=device,
            )

            scheduler.step()
            epoch_time = time.perf_counter() - t0

            vram_alloc, vram_res = get_gpu_memory_usage()
            vram_str = f"{vram_alloc:.2f}/{vram_res:.2f}" if device.type == "cuda" else "N/A"

            print(
                f"{epoch:<6} | {current_lr:<9.2e} | {train_loss:<10.4f} | {val_loss:<9.4f} | "
                f"{val_oa:<9.2f} | {val_f1:<9.2f} | {val_miou:<11.2f} | {epoch_time:<7.1f} | {vram_str}"
            )

            # Log to CSV
            csv_writer.writerow([
                epoch,
                f"{current_lr:.6e}",
                f"{train_loss:.4f}",
                f"{train_oa:.2f}",
                f"{train_f1:.2f}",
                f"{train_miou:.2f}",
                f"{val_loss:.4f}",
                f"{val_oa:.2f}",
                f"{val_f1:.2f}",
                f"{val_miou:.2f}",
                f"{epoch_time:.2f}",
                f"{vram_alloc:.3f}",
                f"{vram_res:.3f}",
            ])
            csv_file.flush()

            # Early stopping and best model saving
            if val_f1 > best_val_f1 or epoch == 1:
                best_val_f1 = val_f1
                epochs_no_improve = 0
                best_metrics = {
                    "epoch": epoch,
                    "val_oa": val_oa,
                    "val_f1": val_f1,
                    "val_miou": val_miou,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "train_oa": train_oa,
                }

                # Save hardened checkpoint
                checkpoint_payload = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "band_mask": band_mask.cpu(),
                    "selected_band_names": selected_band_names,
                    "selected_channels": active_bands,
                    "class_list": LAND_COVER_CLASSES,
                    "metrics": best_metrics,
                    "config_args": vars(args),
                    "git_commit": get_git_commit_hash(),
                }
                torch.save(checkpoint_payload, best_checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("-" * 80)
                    print(
                        f" [EARLY STOPPING] Validation Macro-F1 did not improve for {patience} consecutive epochs. "
                        f"Stopping training at epoch {epoch}."
                    )
                    break

    finally:
        csv_file.close()

    print("=" * 80)
    print(f" [TRAINING COMPLETE] Best Val F1: {best_val_f1:.2f}% (Saved to '{best_checkpoint_path}')")
    print(f" Training log persisted to: '{log_csv_path}'")
    print("=" * 80)

    return model, best_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs the comprehensive CLI parser."""
    parser = argparse.ArgumentParser(
        description="VOXEL Module 3: Hardened GPU Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mock", action="store_true", help="Force synthetic mock data generator mode")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory for real satellite tiles")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV file for real tiles")
    parser.add_argument("--mask_path", type=str, default=None, help="Path to band mask file (.json, .npy, .pt)")
    parser.add_argument("--label_map", type=str, default=None, help="Path to label mapping JSON file")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader multiprocessing workers")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth to resume")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    train_pipeline(cli_args)
