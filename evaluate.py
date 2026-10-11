"""
VOXEL Module 3: Evaluation Script (evaluate.py)
Loads a trained FusionNet checkpoint, evaluates once on the TEST split,
computes Overall Accuracy, Macro F1, Mean IoU, and per-class Precision/Recall/F1,
and saves a confusion matrix visualization to results/confusion_matrix.png.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure Windows UTF-8 stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from models.fusion_net import FusionNet
from config.constants import (
    LAND_COVER_CLASSES,
    NUM_BANDS,
    NUM_CLASSES,
    TASKS,
    get_task_classes,
    get_task_num_classes,
)
from data.datasets import create_split_datasets, RealSentinel2Dataset
from train_gpu import MockSentinel2Dataset


def load_evaluation_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[nn.Module, torch.Tensor, List[str], Dict[str, Any]]:
    """Loads model and metadata from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mask = checkpoint.get("band_mask", None)
    if mask is not None:
        mask = mask.to(device)
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
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, mask, class_list, checkpoint


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str = "results/confusion_matrix.png",
) -> None:
    """Renders a publication-ready annotated confusion matrix heatmap."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    fig.colorbar(cax, fraction=0.046, pad=0.04)

    # Class label ticks
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    ax.set_xlabel("Predicted Class", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Class", fontweight="bold", labelpad=10)
    ax.set_title("VOXEL Module 3: Confusion Matrix (Test Split Evaluation)", fontweight="bold", pad=20)

    # Annotate cell numbers
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" [EVAL] Confusion matrix visualization saved to: {output_path}")


def evaluate_model(
    checkpoint_path: str = "checkpoints/best_fusion_net.pth",
    manifest_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    label_map_path: Optional[str] = None,
    batch_size: int = 8,
    device_name: str = "auto",
    mock: bool = False,
    output_dir: str = "results",
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs test evaluation and outputs complete metrics summary."""
    os.makedirs(output_dir, exist_ok=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=" * 80)
    print(" [VOXEL MODULE 3] TEST SPLIT EVALUATION & METRIC VERIFICATION")
    print("=" * 80)
    print(f"Compute Device:            {device.type.upper()}")
    print(f"Checkpoint File:           {checkpoint_path}")

    model, mask, class_list, ckpt_meta = load_evaluation_checkpoint(checkpoint_path, device)
    effective_task = task or ckpt_meta.get("task")
    if task is not None:
        class_list = get_task_classes(task)
    num_classes = len(class_list)
    print(f"Task:                      {effective_task or 'land_cover (default 10-class)'}")
    print(f"Target Classes ({num_classes}):       {class_list}")

    # Initialize Test Dataset
    if not mock and manifest_path and os.path.exists(manifest_path):
        print(f"Loading real satellite test split from manifest: {manifest_path}")
        _, _, test_ds = create_split_datasets(
            manifest_path=manifest_path,
            data_dir=data_dir,
            label_map_path=label_map_path,
            splits_save_path=os.path.join(output_dir, "eval_splits.json"),
            task=effective_task,
            num_classes=num_classes,
            class_list=class_list,
        )
    else:
        print("Using synthetic Mock Sentinel-2 test dataset (--mock)...")
        test_ds = MockSentinel2Dataset(num_samples=32, num_classes=num_classes, augment=False, seed=999)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Test Samples:              {len(test_ds)} tiles")
    print(f"Active Spectral Channels:  {int(mask.sum().item())} bands")

    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images, mask=mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)

    # Overall Metrics
    oa = float(np.mean(preds_np == targets_np) * 100.0) if len(targets_np) > 0 else 0.0
    macro_f1 = float(f1_score(targets_np, preds_np, average="macro", zero_division=0) * 100.0)
    miou = float(jaccard_score(targets_np, preds_np, average="macro", zero_division=0) * 100.0)
    balanced_acc = float(balanced_accuracy_score(targets_np, preds_np) * 100.0) if len(targets_np) > 0 else 0.0

    # Majority Class Baseline
    majority_class_id = int(np.bincount(targets_np).argmax()) if len(targets_np) > 0 else 0
    majority_class_name = class_list[majority_class_id] if majority_class_id < len(class_list) else str(majority_class_id)
    majority_preds = np.full_like(targets_np, majority_class_id)
    baseline_oa = float(np.mean(majority_preds == targets_np) * 100.0) if len(targets_np) > 0 else 0.0
    baseline_f1 = float(f1_score(targets_np, majority_preds, average="macro", zero_division=0) * 100.0)

    # Per-Class Precision, Recall, F1
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        targets_np, preds_np, labels=list(range(num_classes)), zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(targets_np, preds_np, labels=list(range(num_classes)))
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(cm, class_list, output_path=cm_path)

    print("-" * 80)
    print(f"Overall Accuracy (OA):       {oa:.2f}%")
    print(f"Balanced Accuracy:           {balanced_acc:.2f}%")
    print(f"Macro F1-Score:              {macro_f1:.2f}%")
    print(f"Mean IoU (mIoU):             {miou:.2f}%")
    print(f"Majority-Class Baseline:     {baseline_oa:.2f}% OA | {baseline_f1:.2f}% F1 (Always '{majority_class_name}')")
    print("-" * 80)
    print(f"{'Class ID':<9} | {'Class Name':<28} | {'Precision(%)':<13} | {'Recall(%)':<11} | {'F1(%)':<9} | {'Support'}")
    print("-" * 80)
    for c in range(num_classes):
        p_pct = precisions[c] * 100.0
        r_pct = recalls[c] * 100.0
        f_pct = f1s[c] * 100.0
        sup = int(supports[c])
        print(f"{c:<9} | {class_list[c]:<28} | {p_pct:<13.2f} | {r_pct:<11.2f} | {f_pct:<9.2f} | {sup}")

    print("=" * 80)
    print(" [STATUS] Test evaluation completed successfully!")
    print("=" * 80)

    metrics_summary = {
        "oa": oa,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "miou": miou,
        "majority_baseline_oa": baseline_oa,
        "majority_baseline_f1": baseline_f1,
        "majority_class": majority_class_name,
        "per_class": {
            class_list[c]: {
                "precision": float(precisions[c]),
                "recall": float(recalls[c]),
                "f1": float(f1s[c]),
                "support": int(supports[c]),
            }
            for c in range(num_classes)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": cm_path,
    }

    metrics_json_path = os.path.join(output_dir, "evaluation_metrics.json")
    try:
        with open(metrics_json_path, "w") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f" [EVAL] Evaluation metrics saved to: {metrics_json_path}")
    except Exception as e:
        print(f" [WARN] Failed to write evaluation metrics json: {e}")

    return metrics_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VOXEL Module 3 on Test Split")
    parser.add_argument("--task", type=str, default=None, choices=list(TASKS.keys()), help="Specialized task profile ('forest_cover', 'waterbody')")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_fusion_net.pth", help="Model checkpoint path")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory for satellite tiles")
    parser.add_argument("--label_map", type=str, default=None, help="Path to label map JSON")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--mock", action="store_true", help="Force synthetic mock dataset")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        data_dir=args.data_dir,
        label_map_path=args.label_map,
        batch_size=args.batch_size,
        device_name=args.device,
        mock=args.mock,
        output_dir=args.output_dir,
        task=args.task,
    )
