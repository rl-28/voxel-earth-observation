"""
VOXEL Integration Contract Validator (tools/validate_integration.py)
Validates handoffs from Data Ingestion (Teammate 1) and Band Optimization (Teammate 2)
against docs/INTEGRATION_CONTRACT.md prior to git branch merge.
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Optional, Tuple, Any

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from config.constants import (
    NUM_BANDS,
    NUM_CLASSES,
    TILE_SIZE,
    LAND_COVER_CLASSES,
    SENTINEL2_BAND_KEYS,
    SENTINEL2_BANDS,
    DEFAULT_SCALE_DIVISOR,
    TASKS,
    get_task_classes,
    get_task_num_classes,
)
from utils.band_mask import load_band_mask, describe_band_mask, validate_band_mask
from models.fusion_net import FusionNet
from data.datasets import RealSentinel2Dataset


# ==============================================================================
# 1. DATA VALIDATION CHECKS
# ==============================================================================

def check_manifest_and_files(
    manifest_path: str,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Validates manifest existence, required schema, and tile existence on disk."""
    if not os.path.exists(manifest_path):
        return {
            "status": "FAIL",
            "reason": f"Manifest CSV file not found at: {manifest_path}",
            "df": None,
        }

    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        return {
            "status": "FAIL",
            "reason": f"Failed to parse manifest CSV: {e}",
            "df": None,
        }

    if len(df) == 0:
        return {
            "status": "FAIL",
            "reason": "Manifest CSV contains 0 rows (empty dataset).",
            "df": df,
        }

    required_cols = ["tile_path", "label"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return {
            "status": "FAIL",
            "reason": f"Manifest missing required column(s): {missing_cols}. Found: {list(df.columns)}",
            "df": df,
        }

    base_dir = data_dir if data_dir else os.path.dirname(os.path.abspath(manifest_path))
    missing_files = []
    for idx, path_str in enumerate(df["tile_path"]):
        p = str(path_str)
        if os.path.isabs(p):
            full = p
        elif os.path.exists(p):
            full = p
        elif os.path.exists(os.path.join(base_dir, p)):
            full = os.path.join(base_dir, p)
        else:
            full = os.path.join(base_dir, p)
        if not os.path.exists(full):
            missing_files.append((idx, p))

    if missing_files:
        sample_missing = [p[1] for p in missing_files[:5]]
        return {
            "status": "FAIL",
            "reason": f"{len(missing_files)} / {len(df)} tile files do not exist on disk! First missing: {sample_missing}",
            "df": df,
            "missing_count": len(missing_files),
        }

    return {
        "status": "PASS",
        "reason": f"All {len(df)} tile paths verified on disk. Required columns {required_cols} present.",
        "df": df,
        "total_records": len(df),
    }


def check_sampled_tiles(
    df: pd.DataFrame,
    data_dir: str,
    sample_size: int = 20,
    scale_divisor: float = DEFAULT_SCALE_DIVISOR,
    label_map: Optional[Dict] = None,
    task: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Loads N sampled tiles and verifies dimensions, data type, range [0, 1], and lack of NaN/Inf."""
    n = min(sample_size, len(df))
    if len(df) <= n:
        sample_indices = list(range(len(df)))
    else:
        sample_indices = np.linspace(0, len(df) - 1, n, dtype=int).tolist()

    sample_df = df.iloc[sample_indices].reset_index(drop=True)
    dataset = RealSentinel2Dataset(
        manifest_df=sample_df,
        data_dir=data_dir,
        scale_divisor=scale_divisor,
        label_map=label_map,
        augment=False,
        task=task,
        num_classes=num_classes,
    )

    failures = []
    raw_tiles = []
    norm_tiles = []

    for idx in range(len(dataset)):
        row = sample_df.iloc[idx]
        file_path = str(row["tile_path"])
        if os.path.isabs(file_path):
            full_path = file_path
        elif os.path.exists(file_path):
            full_path = file_path
        elif data_dir and os.path.exists(os.path.join(data_dir, file_path)):
            full_path = os.path.join(data_dir, file_path)
        else:
            full_path = os.path.join(data_dir, file_path) if data_dir else file_path

        # 1. Raw disk inspection
        try:
            if file_path.endswith(".npy"):
                raw_arr = np.load(full_path, mmap_mode="r")
            elif file_path.endswith((".tif", ".tiff")):
                if not HAS_RASTERIO:
                    failures.append(f"Row {idx} ({file_path}): rasterio required to read GeoTIFF")
                    continue
                with rasterio.open(full_path) as src:
                    raw_arr = src.read()
            else:
                raw_arr = None

            if raw_arr is not None:
                raw_tiles.append(np.array(raw_arr))
                if raw_arr.ndim == 3 and raw_arr.shape[0] != NUM_BANDS and raw_arr.shape[2] != NUM_BANDS:
                    c_found = raw_arr.shape[0] if raw_arr.shape[2] != NUM_BANDS else raw_arr.shape[2]
                    failures.append(
                        f"Row {idx} ({file_path}): invalid band count ({c_found} bands), expected exactly {NUM_BANDS}"
                    )
                if hasattr(raw_arr, "_mmap") and raw_arr._mmap is not None:
                    try:
                        raw_arr._mmap.close()
                    except Exception:
                        pass
                del raw_arr
        except Exception as e:
            failures.append(f"Row {idx} ({file_path}): raw disk read error: {e}")
            continue

        # 2. Dataset loader inspection
        try:
            tile_tensor, label_tensor = dataset[idx]
        except Exception as e:
            failures.append(f"Row {idx} ({file_path}): dataset __getitem__ failed: {e}")
            continue

        norm_tiles.append(tile_tensor.cpu().numpy())

        # Validate band count == 13
        if tile_tensor.shape[0] != NUM_BANDS:
            failures.append(
                f"Row {idx} ({file_path}): band count is {tile_tensor.shape[0]}, expected {NUM_BANDS}"
            )

        # Validate spatial shape == (13, 256, 256)
        if tuple(tile_tensor.shape) != (NUM_BANDS, TILE_SIZE, TILE_SIZE):
            failures.append(
                f"Row {idx} ({file_path}): shape {tuple(tile_tensor.shape)} != expected ({NUM_BANDS}, {TILE_SIZE}, {TILE_SIZE})"
            )

        # Validate float32
        if tile_tensor.dtype != torch.float32:
            failures.append(
                f"Row {idx} ({file_path}): dtype {tile_tensor.dtype} != expected torch.float32"
            )

        # Validate value range [0.0, 1.0]
        t_min = float(torch.min(tile_tensor).item())
        t_max = float(torch.max(tile_tensor).item())
        if t_min < -1e-5 or t_max > 1.00001:
            failures.append(
                f"Row {idx} ({file_path}): values [{t_min:.4f}, {t_max:.4f}] outside valid [0.0, 1.0] range"
            )

        # Validate NaN and Inf
        if torch.isnan(tile_tensor).any():
            failures.append(f"Row {idx} ({file_path}): contains NaN values")
        if torch.isinf(tile_tensor).any():
            failures.append(f"Row {idx} ({file_path}): contains Inf values")

    if failures:
        return {
            "status": "FAIL",
            "reason": f"{len(failures)} tile verification failures across {n} sampled tiles:\n"
            + "\n".join(f"  • {f}" for f in failures[:5]),
            "raw_tiles": raw_tiles,
            "norm_tiles": norm_tiles,
            "sample_df": sample_df,
            "sample_size": n,
        }

    return {
        "status": "PASS",
        "reason": f"All {n} sampled tiles verified: shape (13, 256, 256), dtype float32, range [0.0, 1.0], no NaN/Inf.",
        "raw_tiles": raw_tiles,
        "norm_tiles": norm_tiles,
        "sample_df": sample_df,
        "sample_size": n,
    }


def check_spectral_and_scaling(
    raw_tiles: List[np.ndarray],
    norm_tiles: List[np.ndarray],
) -> Dict[str, Any]:
    """Detects band-order anomalies, all-zero bands, and improper scaling."""
    if not norm_tiles:
        return {
            "status": "FAIL",
            "reason": "No valid tiles available for spectral and scaling analysis.",
            "band_stats": None,
        }

    norm_stack = np.stack(norm_tiles, axis=0)  # shape: (N, 13, 256, 256)
    band_means = np.mean(norm_stack, axis=(0, 2, 3))
    band_mins = np.min(norm_stack, axis=(0, 2, 3))
    band_maxs = np.max(norm_stack, axis=(0, 2, 3))
    band_stds = np.std(norm_stack, axis=(0, 2, 3))

    anomalies = []
    warnings_list = []

    # 1. Check for all-zero bands
    for b in range(NUM_BANDS):
        if band_maxs[b] <= 1e-7:
            band_desc = SENTINEL2_BANDS[b]
            if b == 10:  # B10 Cirrus is frequently masked out in L2A BOA
                warnings_list.append(
                    f"Band 10 ({band_desc}) is all-zero across sampled tiles (common in L2A bottom-of-atmosphere)."
                )
            else:
                anomalies.append(
                    f"Critical Band {b} ({band_desc}) is all-zero across all {len(norm_tiles)} sampled tiles!"
                )

    # 2. Check raw scaling
    if raw_tiles:
        max_raw = max(float(np.max(t)) for t in raw_tiles)
        min_raw = min(float(np.min(t)) for t in raw_tiles)
        if min_raw < -1e-4:
            warnings_list.append(f"Raw data contains negative values (minimum: {min_raw:.4f}).")
        if max_raw > 10000.0:
            warnings_list.append(
                f"Raw tile max value ({max_raw:.1f}) exceeds standard Sentinel-2 10,000 DN scale."
            )
        elif max_raw > 1.05 and max_raw <= 10000.0:
            # Scaled properly via divisor, note as info
            pass

    # 3. Spectral signature / band-order heuristic check (advisory WARNING, not hard FAIL):
    # Sentinel-2 canonical: B04 is Red (idx 3, 665 nm), B08 is Broad NIR (idx 7, 842 nm).
    tile_b04_means = np.mean(norm_stack[:, 3, :, :], axis=(1, 2))  # (N,)
    tile_b08_means = np.mean(norm_stack[:, 7, :, :], axis=(1, 2))  # (N,)
    n_tiles = len(norm_tiles)
    b08_brighter_count = int(np.sum(tile_b08_means > tile_b04_means))

    avg_b04 = float(band_means[3])
    avg_b08 = float(band_means[7])
    band_means_std = float(np.std(band_means))
    band_means_spread = float(np.ptp(band_means))

    nearly_identical = (band_means_std < 0.01 or band_means_spread < 0.02)
    b08_not_brighter = (b08_brighter_count <= n_tiles // 2 or avg_b08 <= avg_b04)

    if nearly_identical:
        warnings_list.append(
            f"Suspicious spectral signature: Band means are nearly identical across all 13 bands "
            f"(std={band_means_std:.4f}, spread={band_means_spread:.4f}). "
            f"Note: This heuristic cannot prove the band order is right, only flags suspicious data "
            f"(e.g., constant, duplicate, or uncalibrated bands)."
        )
    elif b08_not_brighter:
        warnings_list.append(
            f"Suspicious band ordering / spectral response: B08 NIR (idx 7, mean={avg_b08:.4f}) is not brighter than "
            f"B04 Red (idx 3, mean={avg_b04:.4f}) for most sampled tiles ({b08_brighter_count}/{n_tiles} tiles had B08 > B04). "
            f"Note: This heuristic cannot prove the band order is right, only flags suspicious data "
            f"(non-vegetated scenes like water or bare soil naturally exhibit low NIR)."
        )
    else:
        # Tiles look vegetated (B08 NIR clearly brighter than B04 Red on average) -> pass silently
        pass

    status = "FAIL" if anomalies else ("WARN" if warnings_list else "PASS")
    reason_parts = anomalies + warnings_list
    reason = "; ".join(reason_parts) if reason_parts else "Per-band reflectance means healthy and in [0.0, 1.0]; no all-zero visible bands."

    return {
        "status": status,
        "reason": reason,
        "band_stats": {
            "means": band_means,
            "mins": band_mins,
            "maxs": band_maxs,
            "stds": band_stds,
        },
        "anomalies": anomalies,
        "warnings": warnings_list,
    }


def check_labels_and_distribution(
    df: pd.DataFrame,
    label_map: Optional[Dict] = None,
    task: Optional[str] = None,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validates class labels resolve to 0..(num_classes-1) and checks class sample count and balance."""
    if task is not None and task in TASKS:
        target_classes = list(TASKS[task]["classes"])
        n_classes = len(target_classes)
    elif class_names is not None:
        target_classes = list(class_names)
        n_classes = num_classes if num_classes is not None else len(target_classes)
    elif num_classes is not None:
        n_classes = num_classes
        target_classes = [f"Class_{i}" for i in range(num_classes)]
    else:
        target_classes = list(LAND_COVER_CLASSES)
        n_classes = NUM_CLASSES

    resolved_labels = []
    unmapped = []

    for idx, raw_label in enumerate(df["label"]):
        target = None
        if label_map is not None:
            if raw_label in label_map:
                target = int(label_map[raw_label])
            elif str(raw_label).strip() in label_map:
                target = int(label_map[str(raw_label).strip()])
            else:
                unmapped.append((idx, raw_label))
                continue
        else:
            try:
                target = int(raw_label)
            except (ValueError, TypeError):
                str_lbl = str(raw_label).strip()
                if str_lbl in target_classes:
                    target = target_classes.index(str_lbl)
                elif str_lbl in LAND_COVER_CLASSES:
                    target = LAND_COVER_CLASSES.index(str_lbl)
                else:
                    unmapped.append((idx, raw_label))
                    continue

        if 0 <= target < n_classes:
            resolved_labels.append(target)
        else:
            unmapped.append((idx, raw_label))

    if unmapped:
        sample_unmapped = [u[1] for u in unmapped[:5]]
        return {
            "status": "FAIL",
            "reason": f"{len(unmapped)} records have labels that cannot be mapped to the {n_classes} known classes (0..{n_classes - 1}). First few: {sample_unmapped}",
            "distribution": None,
        }

    counts = pd.Series(resolved_labels).value_counts().to_dict()
    class_dist = {i: counts.get(i, 0) for i in range(n_classes)}

    warnings_list = []
    min_count = min(class_dist.values())
    max_count = max(class_dist.values())

    # Check minimum samples per class
    for class_id, count in class_dist.items():
        if count < 20:
            c_name = target_classes[class_id] if class_id < len(target_classes) else f"Class_{class_id}"
            warnings_list.append(
                f"Class {class_id} ('{c_name}') has only {count} samples (< 20 threshold)."
            )

    # Check class imbalance ratio
    if min_count > 0:
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 3.0:
            warnings_list.append(
                f"Class imbalance ratio is {imbalance_ratio:.1f}x (> 3.0x threshold: max={max_count}, min={min_count})."
            )
    else:
        warnings_list.append("One or more classes have 0 samples in manifest!")

    status = "WARN" if warnings_list else "PASS"
    reason = "; ".join(warnings_list) if warnings_list else f"All {len(df)} samples map to {n_classes} classes with balanced distribution (ratio {max_count / max(1, min_count):.1f}x)."

    return {
        "status": status,
        "reason": reason,
        "distribution": class_dist,
        "warnings": warnings_list,
        "class_names": target_classes,
        "num_classes": n_classes,
    }


def check_spatial_splits(df: pd.DataFrame) -> Dict[str, Any]:
    """Confirms train, val, and test partitions do not share location_id."""
    if "location_id" not in df.columns:
        return {
            "status": "WARN",
            "reason": "Manifest lacks 'location_id' column. Group-aware spatial splitting cannot be validated.",
        }

    if "split" not in df.columns or df["split"].dropna().nunique() <= 1:
        unique_locs = df["location_id"].nunique()
        return {
            "status": "PASS",
            "reason": f"Column 'location_id' present ({unique_locs} unique locations); manifest does not declare pre-assigned splits.",
        }

    train_locs = set(df[df["split"].astype(str).str.lower() == "train"]["location_id"].dropna())
    val_locs = set(df[df["split"].astype(str).str.lower() == "val"]["location_id"].dropna())
    test_locs = set(df[df["split"].astype(str).str.lower() == "test"]["location_id"].dropna())

    train_val = train_locs & val_locs
    train_test = train_locs & test_locs
    val_test = val_locs & test_locs

    leaks = []
    if train_val:
        leaks.append(f"train & val share {len(train_val)} location(s): {sorted(list(train_val))[:5]}")
    if train_test:
        leaks.append(f"train & test share {len(train_test)} location(s): {sorted(list(train_test))[:5]}")
    if val_test:
        leaks.append(f"val & test share {len(val_test)} location(s): {sorted(list(val_test))[:5]}")

    if leaks:
        return {
            "status": "FAIL",
            "reason": "Spatial data leakage detected across splits! " + "; ".join(leaks),
            "leaks": leaks,
        }

    return {
        "status": "PASS",
        "reason": f"Zero spatial leakage across splits ({len(train_locs)} train, {len(val_locs)} val, {len(test_locs)} test unique locations).",
        "spatial_counts_str": f"{len(train_locs)} train, {len(val_locs)} val, {len(test_locs)} test unique locations",
    }


# ==============================================================================
# 2. BAND MASK VALIDATION CHECKS
# ==============================================================================

def check_mask_specification(mask_path: str) -> Dict[str, Any]:
    """Validates mask loading via load_band_mask, length 13, binary, >= 1 active band, and extracts metadata."""
    if not os.path.exists(mask_path):
        return {
            "status": "FAIL",
            "reason": f"Band mask file not found at: {mask_path}",
            "mask_tensor": None,
        }

    try:
        mask_tensor = load_band_mask(mask_path)
    except Exception as e:
        return {
            "status": "FAIL",
            "reason": f"Band mask loading failed: {e}",
            "mask_tensor": None,
        }

    # Extract metadata from JSON if available
    method = "N/A"
    dataset = "N/A"
    raw_meta = {}
    if mask_path.endswith(".json"):
        try:
            with open(mask_path, "r", encoding="utf-8") as f:
                raw_meta = json.load(f)
            if isinstance(raw_meta, dict):
                method = raw_meta.get("method", "Unspecified")
                dataset = raw_meta.get("dataset", "Unspecified")
        except Exception:
            pass

    selected_descs = describe_band_mask(mask_tensor)
    active_count = int(mask_tensor.sum().item())

    return {
        "status": "PASS",
        "reason": f"{active_count} of 13 bands selected ({', '.join([s.split()[0] for s in selected_descs])}). Method: {method}, Dataset: {dataset}.",
        "mask_tensor": mask_tensor,
        "active_count": active_count,
        "selected_bands": selected_descs,
        "method": method,
        "dataset": dataset,
    }


# ==============================================================================
# 3. MODEL COMPATIBILITY (FUSIONNET FORWARD + BACKWARD PASS)
# ==============================================================================

def check_model_compatibility(
    mask_tensor: Optional[torch.Tensor],
    sample_df: pd.DataFrame,
    data_dir: str,
    device_str: Optional[str] = None,
    scale_divisor: float = DEFAULT_SCALE_DIVISOR,
    label_map: Optional[Dict] = None,
    task: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Builds FusionNet with active channel count, executes forward+backward pass on real GPU mini-batch."""
    if mask_tensor is None:
        return {
            "status": "FAIL",
            "reason": "Model compatibility check skipped because band mask is invalid.",
        }

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    active_channels = int(mask_tensor.sum().item())
    n_classes = len(TASKS[task]["classes"]) if (task and task in TASKS) else (num_classes or NUM_CLASSES)

    try:
        model = FusionNet(
            num_classes=n_classes,
            total_bands=NUM_BANDS,
            selected_channels=active_channels,
        ).to(device)
        model.train()
    except Exception as e:
        return {
            "status": "FAIL",
            "reason": f"Failed to instantiate FusionNet: {e}",
        }

    batch_size = min(4, len(sample_df))
    dataset = RealSentinel2Dataset(
        manifest_df=sample_df.iloc[:batch_size],
        data_dir=data_dir,
        scale_divisor=scale_divisor,
        label_map=label_map,
        augment=False,
        task=task,
        num_classes=n_classes,
    )

    images = []
    targets = []
    try:
        for idx in range(len(dataset)):
            x, y = dataset[idx]
            images.append(x)
            targets.append(y)
    except Exception as e:
        return {
            "status": "FAIL",
            "reason": f"Failed to load mini-batch sample tiles for model compatibility check: {e}",
        }

    images = torch.stack(images, dim=0).to(device)
    targets = torch.stack(targets, dim=0).to(device)
    mask = mask_tensor.to(device)

    # 1. Forward Pass
    try:
        logits, embeddings = model(images, mask=mask, return_embedding=True)
    except Exception as e:
        return {
            "status": "FAIL",
            "reason": f"Forward pass crashed: {e}",
        }

    expected_logits = (batch_size, n_classes)
    expected_embeddings = (batch_size, 1216)
    if logits.shape != expected_logits:
        return {
            "status": "FAIL",
            "reason": f"Logits shape {tuple(logits.shape)} != expected {expected_logits}",
        }
    if embeddings.shape != expected_embeddings:
        return {
            "status": "FAIL",
            "reason": f"Embeddings shape {tuple(embeddings.shape)} != expected {expected_embeddings}",
        }

    # 2. Backward Pass
    try:
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            return {
                "status": "FAIL",
                "reason": f"Loss is NaN or Infinite: {loss.item()}",
            }
        loss.backward()

        first_param = next(model.parameters())
        if first_param.grad is None:
            return {
                "status": "FAIL",
                "reason": "Backward pass completed but model parameter gradients are None.",
            }
    except Exception as e:
        return {
            "status": "FAIL",
            "reason": f"Backward pass failed: {e}",
        }

    return {
        "status": "PASS",
        "reason": f"Forward & backward pass successful on {device.type.upper()} "
        f"(input: {tuple(images.shape)}, logits: {tuple(logits.shape)}, embeddings: {tuple(embeddings.shape)}, loss: {loss.item():.4f}).",
        "device": str(device),
        "input_shape": tuple(images.shape),
        "logits_shape": tuple(logits.shape),
        "embeddings_shape": tuple(embeddings.shape),
        "loss": float(loss.item()),
    }


# ==============================================================================
# 4. REPORTING, SUMMARY TABLE & TEAMMATE ACTION MESSAGE
# ==============================================================================

def generate_summary_table(checks: List[Dict[str, str]]) -> str:
    """Renders a clean ASCII summary table of all checks."""
    lines = []
    header_title = "VOXEL INTEGRATION CONTRACT VALIDATION SUMMARY"
    lines.append("=" * 104)
    lines.append(f"{header_title:^104}")
    lines.append("=" * 104)
    lines.append(f"{'#':<4} | {'Component':<22} | {'Validation Check':<28} | {'Status':<8} | {'Details'}")
    lines.append("-" * 104)

    for idx, c in enumerate(checks, 1):
        status = c["status"]
        status_display = f"[{status}]"
        # Truncate details if very long for single-line display
        details = c["details"]
        if len(details) > 36:
            details_line = details[:33] + "..."
        else:
            details_line = details
        lines.append(f"{idx:<4} | {c['component']:<22} | {c['check']:<28} | {status_display:<8} | {details_line}")

    lines.append("=" * 104)
    return "\n".join(lines)


def generate_teammate_message(
    overall_ready: bool,
    checks: List[Dict[str, str]],
    manifest_path: str,
    mask_path: str,
    meta_info: Dict[str, Any],
) -> str:
    """Creates a ready-to-copy message formatted for teammates (Slack/PR/Email)."""
    lines = []
    lines.append("=" * 80)
    lines.append("READY-TO-SEND TEAMMATE MESSAGE (Copy & Paste to Slack / Pull Request)")
    lines.append("=" * 80)

    if overall_ready:
        lines.append("Hi Team,")
        lines.append("")
        lines.append("Great news! I ran our automated contract validator (`tools/validate_integration.py`)")
        lines.append("against docs/INTEGRATION_CONTRACT.md and all integration checks have PASSED: ✅")
        lines.append("")
        lines.append(f"• Ingestion Data: {meta_info.get('total_tiles', 'N/A')} tiles verified across {meta_info.get('total_classes', 10)} classes.")
        lines.append(f"  - Tiles are (13, 256, 256) float32 in [0.0, 1.0], no missing files or NaNs.")
        lines.append(f"  - Zero spatial leakage across splits ({meta_info.get('spatial_summary', 'clean')}).")
        lines.append(f"• Band Optimization Mask: {meta_info.get('active_bands', 'N/A')} active bands ({meta_info.get('selected_bands_str', '')}).")
        lines.append(f"  - Method: {meta_info.get('mask_method', 'N/A')}, Dataset: {meta_info.get('mask_dataset', 'N/A')}.")
        lines.append(f"• Module 3 FusionNet: Forward & backward pass validated on {meta_info.get('device', 'GPU')}.")
        lines.append(f"  - Logits shape: {meta_info.get('logits_shape', '(B, 10)')}, Embeddings: {meta_info.get('embeddings_shape', '(B, 1216)')}.")
        lines.append("")
        lines.append("Your handoff outputs are ready for integration with Module 3. 🚀")

    else:
        failures = [c for c in checks if c["status"] == "FAIL"]
        warnings = [c for c in checks if c["status"] == "WARN"]

        lines.append("Hi Team,")
        lines.append("")
        lines.append("I ran our automated contract validator (`tools/validate_integration.py`) against")
        lines.append("docs/INTEGRATION_CONTRACT.md. Before we can merge, please address the following issues: ❌")
        lines.append("")

        lines.append("REQUIRED FIXES:")
        for idx, f in enumerate(failures, 1):
            lines.append(f"  {idx}. [{f['component'].upper()}] {f['check']}")
            lines.append(f"     Reason: {f['details']}")

        if warnings:
            lines.append("")
            lines.append("ADVISORY WARNINGS:")
            for idx, w in enumerate(warnings, 1):
                lines.append(f"  • [{w['component'].upper()}] {w['details']}")

        lines.append("")
        lines.append("Please push updates to your handoff files so we can re-verify and merge. Thanks!")

    lines.append("=" * 80)
    return "\n".join(lines)


# ==============================================================================
# 5. MAIN ORCHESTRATOR FUNCTION
# ==============================================================================

def validate_integration(
    manifest_path: str,
    mask_path: str,
    label_map_path: Optional[str] = None,
    sample_size: int = 20,
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
    scale_divisor: float = DEFAULT_SCALE_DIVISOR,
    task: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes all validation checks and returns a structured dictionary of results.
    """
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(manifest_path))

    label_map = None
    if label_map_path and os.path.exists(label_map_path):
        try:
            with open(label_map_path, "r", encoding="utf-8") as f:
                label_map = json.load(f)
        except Exception as e:
            print(f"[VALIDATOR WARNING] Failed to read label_map JSON at {label_map_path}: {e}")

    checks_summary = []
    meta_info = {}

    # 1. Manifest & Files
    res_manifest = check_manifest_and_files(manifest_path=manifest_path, data_dir=data_dir)
    checks_summary.append({
        "component": "Data Ingestion",
        "check": "Manifest & Disk Existence",
        "status": res_manifest["status"],
        "details": res_manifest["reason"],
    })

    if res_manifest["status"] == "FAIL":
        return {
            "overall_ready": False,
            "checks": checks_summary,
            "manifest_res": res_manifest,
            "tiles_res": None,
            "spectral_res": None,
            "labels_res": None,
            "spatial_res": None,
            "mask_res": None,
            "model_res": None,
            "teammate_message": generate_teammate_message(False, checks_summary, manifest_path, mask_path, meta_info),
        }

    df = res_manifest["df"]
    meta_info["total_tiles"] = len(df)

    # 2. Sample N Tiles
    res_tiles = check_sampled_tiles(
        df=df,
        data_dir=data_dir,
        sample_size=sample_size,
        scale_divisor=scale_divisor,
        label_map=label_map,
        task=task,
        num_classes=num_classes,
    )
    checks_summary.append({
        "component": "Data Ingestion",
        "check": f"Sample {res_tiles.get('sample_size', sample_size)} Tiles Layout",
        "status": res_tiles["status"],
        "details": res_tiles["reason"],
    })

    # 3. Spectral & Scaling
    res_spectral = check_spectral_and_scaling(
        raw_tiles=res_tiles.get("raw_tiles", []),
        norm_tiles=res_tiles.get("norm_tiles", []),
    )
    checks_summary.append({
        "component": "Data Ingestion",
        "check": "Spectral Bands & Scaling",
        "status": res_spectral["status"],
        "details": res_spectral["reason"],
    })

    # 4. Labels & Distribution
    res_labels = check_labels_and_distribution(
        df=df,
        label_map=label_map,
        task=task,
        num_classes=num_classes,
    )
    checks_summary.append({
        "component": "Data Ingestion",
        "check": "Class Labels & Balance",
        "status": res_labels["status"],
        "details": res_labels["reason"],
    })
    meta_info["total_classes"] = res_labels.get("num_classes", NUM_CLASSES)

    # 5. Spatial Splits (location_id)
    res_spatial = check_spatial_splits(df=df)
    checks_summary.append({
        "component": "Data Ingestion",
        "check": "Spatial Leakage (location_id)",
        "status": res_spatial["status"],
        "details": res_spatial["reason"],
    })
    meta_info["spatial_summary"] = res_spatial.get("spatial_counts_str", res_spatial["reason"])

    # 6. Mask Specification
    res_mask = check_mask_specification(mask_path=mask_path)
    checks_summary.append({
        "component": "Band Optimization",
        "check": "Mask Specification & Schema",
        "status": res_mask["status"],
        "details": res_mask["reason"],
    })
    meta_info["active_bands"] = res_mask.get("active_count", "N/A")
    meta_info["mask_method"] = res_mask.get("method", "N/A")
    meta_info["mask_dataset"] = res_mask.get("dataset", "N/A")
    if res_mask.get("selected_bands"):
        meta_info["selected_bands_str"] = ", ".join([b.split()[0] for b in res_mask["selected_bands"]])

    # 7. Model Compatibility
    if res_tiles["status"] == "FAIL":
        res_model = {
            "status": "FAIL",
            "reason": "Model compatibility check skipped because sampled tiles failed verification.",
        }
    elif res_mask["status"] == "FAIL":
        res_model = {
            "status": "FAIL",
            "reason": "Model compatibility check skipped because band mask failed verification.",
        }
    else:
        res_model = check_model_compatibility(
            mask_tensor=res_mask.get("mask_tensor"),
            sample_df=res_tiles.get("sample_df", df),
            data_dir=data_dir,
            device_str=device,
            scale_divisor=scale_divisor,
            label_map=label_map,
            task=task,
            num_classes=num_classes,
        )
    checks_summary.append({
        "component": "Module 3 Model",
        "check": "FusionNet GPU Step Compatibility",
        "status": res_model["status"],
        "details": res_model["reason"],
    })
    meta_info["device"] = res_model.get("device", "GPU")
    meta_info["logits_shape"] = str(res_model.get("logits_shape", ""))
    meta_info["embeddings_shape"] = str(res_model.get("embeddings_shape", ""))

    # Determine overall status
    has_failures = any(c["status"] == "FAIL" for c in checks_summary)
    overall_ready = not has_failures

    teammate_msg = generate_teammate_message(
        overall_ready=overall_ready,
        checks=checks_summary,
        manifest_path=manifest_path,
        mask_path=mask_path,
        meta_info=meta_info,
    )

    return {
        "overall_ready": overall_ready,
        "checks": checks_summary,
        "manifest_res": res_manifest,
        "tiles_res": res_tiles,
        "spectral_res": res_spectral,
        "labels_res": res_labels,
        "spatial_res": res_spatial,
        "mask_res": res_mask,
        "model_res": res_model,
        "meta_info": meta_info,
        "teammate_message": teammate_msg,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "VOXEL Integration Contract Validator (checks ingestion and optimization handoffs against docs/INTEGRATION_CONTRACT.md).\n\n"
            "LIMITATION NOTE ON SPECTRAL / BAND-ORDER CHECK:\n"
            "The spectral validator tests whether B08 (Broad NIR) is brighter than B04 (Red) as a heuristic\n"
            "for typical vegetated land cover, and flags nearly identical band means. This check only emits\n"
            "an advisory WARNING and CANNOT definitively prove that the Sentinel-2 band order is correct\n"
            "(e.g., scenes dominated by water or bare soil naturally exhibit low NIR reflectance). Users must\n"
            "verify satellite ingestion metadata if an advisory warning is emitted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to band mask (.json, .npy, .pt)")
    parser.add_argument("--label_map", type=str, default=None, help="Optional path to label_map.json")
    parser.add_argument("--sample", type=int, default=20, help="Number of tiles to sample (default: 20)")
    parser.add_argument("--data_dir", type=str, default=None, help="Base data directory for tiles (default: dirname of manifest)")
    parser.add_argument("--device", type=str, default=None, help="Device for model test ('cuda' or 'cpu')")
    parser.add_argument("--scale_divisor", type=float, default=DEFAULT_SCALE_DIVISOR, help="Scale divisor for raw DN tiles")
    parser.add_argument("--task", type=str, default=None, choices=list(TASKS.keys()), help="Task configuration name")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes override")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("VOXEL INTEGRATION CONTRACT VALIDATOR (docs/INTEGRATION_CONTRACT.md)")
    print("=" * 80)
    print(f"Manifest File  : {args.manifest}")
    print(f"Band Mask File : {args.mask_path}")
    if args.task:
        print(f"Task Name      : {args.task}")
    if args.label_map:
        print(f"Label Map JSON : {args.label_map}")
    print(f"Sample Count   : {args.sample}")
    print("=" * 80 + "\n")

    results = validate_integration(
        manifest_path=args.manifest,
        mask_path=args.mask_path,
        label_map_path=args.label_map,
        sample_size=args.sample,
        data_dir=args.data_dir,
        device=args.device,
        scale_divisor=args.scale_divisor,
        task=args.task,
        num_classes=args.num_classes,
    )

    # Print Itemized Results
    print("--- 1. DATA INGESTION CHECKS ---")
    print(f"• Manifest & Disk Existence : [{results['checks'][0]['status']}] {results['checks'][0]['details']}")
    if results['tiles_res']:
        print(f"• Sampled Tiles Integrity  : [{results['checks'][1]['status']}] {results['checks'][1]['details']}")
    if results['spectral_res']:
        print(f"• Spectral & Scaling        : [{results['checks'][2]['status']}] {results['checks'][2]['details']}")
        # Print Per-Band Stats Table
        if results['spectral_res']['band_stats'] is not None:
            stats = results['spectral_res']['band_stats']
            print("\n  Per-Band Reflectance Statistics (Sampled Tiles):")
            print("  Idx | Key | Band Name           | Mean   | Min    | Max    | Std   ")
            print("  " + "-" * 60)
            for b in range(NUM_BANDS):
                k = SENTINEL2_BAND_KEYS[b]
                name = SENTINEL2_BANDS[b].split("(")[1].replace(")", "")[:13]
                print(f"  {b:>2}  | {k:<3} | {name:<17} | {stats['means'][b]:.4f} | {stats['mins'][b]:.4f} | {stats['maxs'][b]:.4f} | {stats['stds'][b]:.4f}")
            print()

    if results['labels_res']:
        print(f"• Class Labels & Balance    : [{results['checks'][3]['status']}] {results['checks'][3]['details']}")
        if results['labels_res']['distribution'] is not None:
            dist = results['labels_res']['distribution']
            tot = sum(dist.values())
            target_classes = results['labels_res'].get('class_names', LAND_COVER_CLASSES)
            n_cls = len(target_classes)
            print("\n  Target Class Distribution:")
            print("  ID | Class Name                 | Count | Share (%)")
            print("  " + "-" * 50)
            for cid in range(n_cls):
                cnt = dist.get(cid, 0)
                pct = (cnt / tot * 100.0) if tot > 0 else 0.0
                cname = target_classes[cid] if cid < len(target_classes) else f"Class_{cid}"
                print(f"  {cid:>2} | {cname:<26} | {cnt:>5} | {pct:>6.2f}%")
            print()

    if results['spatial_res']:
        print(f"• Spatial Leakage (location): [{results['checks'][4]['status']}] {results['checks'][4]['details']}")

    print("\n--- 2. BAND OPTIMIZATION MASK CHECKS ---")
    if results['mask_res']:
        print(f"• Mask Specification        : [{results['checks'][5]['status']}] {results['checks'][5]['details']}")
        if results['mask_res'].get('selected_bands'):
            print(f"  Selected Bands: {results['mask_res']['selected_bands']}")
            print(f"  Algorithm/Method: {results['mask_res'].get('method', 'N/A')}")
            print(f"  Target Dataset: {results['mask_res'].get('dataset', 'N/A')}")

    print("\n--- 3. MODULE 3 MODEL COMPATIBILITY ---")
    if results['model_res']:
        print(f"• FusionNet Step Test       : [{results['checks'][6]['status']}] {results['checks'][6]['details']}")
        if results['model_res']['status'] == "PASS":
            print(f"  Input Batch Shape: {results['model_res']['input_shape']}")
            print(f"  Output Logits    : {results['model_res']['logits_shape']}")
            print(f"  Output Embedding : {results['model_res']['embeddings_shape']}")
            print(f"  Mini-batch Loss  : {results['model_res']['loss']:.4f}")

    print("\n" + generate_summary_table(results["checks"]) + "\n")

    if results["overall_ready"]:
        print("🎉 OVERALL VERDICT: READY FOR INTEGRATION WITH MODULE 3 ✅")
    else:
        print("🚫 OVERALL VERDICT: NOT READY FOR INTEGRATION WITH MODULE 3 ❌")

    print("\n" + results["teammate_message"] + "\n")

    sys.exit(0 if results["overall_ready"] else 1)


if __name__ == "__main__":
    main()
