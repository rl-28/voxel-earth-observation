"""
VOXEL Module 3: Pluggable Sentinel-2 Dataset Infrastructure.
Provides lazy-loading datasets for real on-disk Sentinel-2 data (NumPy .npy and GeoTIFF .tif),
with automated channel detection, normalization, spatial augmentation, and group-aware splitting.
"""

import json
import os
import sys
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from config.constants import (
    DEFAULT_SCALE_DIVISOR,
    LAND_COVER_CLASSES,
    NUM_BANDS,
    NUM_CLASSES,
    TILE_SIZE,
)


class SpatialAugmentation:
    """Applies random horizontal flips, vertical flips, and 90-degree rotations to 13-band tiles."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is (C, H, W)
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-1])  # Horizontal flip
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-2])  # Vertical flip
        k = int(torch.randint(0, 4, (1,)).item())
        if k > 0:
            x = torch.rot90(x, k=k, dims=[-2, -1])  # 90-deg rotation
        return x


class BaseSentinel2Dataset(Dataset):
    """Abstract base dataset for Sentinel-2 satellite tiles."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MockSentinel2Dataset(BaseSentinel2Dataset):
    """
    Synthetic dataset for testing and headless verification.
    Generates normalized 13-band tiles and deterministic or random labels.
    """

    def __init__(
        self,
        num_samples: int = 64,
        channels: int = NUM_BANDS,
        height: int = TILE_SIZE,
        width: int = TILE_SIZE,
        num_classes: int = NUM_CLASSES,
        augment: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.augment = augment
        self.transform = SpatialAugmentation() if augment else None

        torch.manual_seed(seed)
        self.data = torch.rand((num_samples, channels, height, width), dtype=torch.float32)
        self.targets = torch.randint(0, num_classes, (num_samples,), dtype=torch.int64)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.targets[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class RealSentinel2Dataset(BaseSentinel2Dataset):
    """
    Production dataset that lazily loads real Sentinel-2 tiles from disk.
    Supports .npy arrays and multi-band GeoTIFF (.tif) files indexed via manifest.csv.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        data_dir: str = "",
        scale_divisor: float = DEFAULT_SCALE_DIVISOR,
        label_map: Optional[Dict[Union[str, int], int]] = None,
        split: Optional[str] = None,
        augment: bool = False,
        target_size: int = TILE_SIZE,
    ):
        """
        Args:
            manifest_df: DataFrame with at least ['tile_path', 'label'].
            data_dir: Base directory for relative tile paths.
            scale_divisor: Scaling divisor for raw digital numbers (default: 10000.0).
            label_map: Optional dictionary mapping raw label strings/IDs to [0, NUM_CLASSES - 1].
            split: Filter to a specific split ('train', 'val', 'test').
            augment: Whether to apply spatial augmentations (only applied if split == 'train').
            target_size: Expected spatial height and width (default: 256).
        """
        super().__init__()
        self.data_dir = data_dir
        self.scale_divisor = scale_divisor
        self.label_map = label_map
        self.target_size = target_size
        self._logged_scaling = False
        self._logged_resizing = False

        df = manifest_df.copy()

        # Filter by split if column exists and split specified
        if split is not None and "split" in df.columns:
            df = df[df["split"].astype(str).str.lower() == split.lower()].reset_index(drop=True)

        self.records = df
        self.split = split
        self.augment = augment and (split is None or split.lower() == "train")
        self.transform = SpatialAugmentation() if self.augment else None

        if len(self.records) == 0:
            warnings.warn(f"Dataset initialized with 0 samples for split='{split}'", UserWarning)

    def __len__(self) -> int:
        return len(self.records)

    def _load_tile_from_disk(self, file_path: str) -> np.ndarray:
        """Loads a tile from disk and auto-detects channel layout."""
        if not os.path.isabs(file_path) and self.data_dir:
            full_path = os.path.join(self.data_dir, file_path)
        else:
            full_path = file_path

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Satellite tile not found at: {full_path}")

        ext = os.path.splitext(full_path)[1].lower()

        if ext == ".npy":
            arr = np.load(full_path)
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D tile array, got {arr.ndim}D with shape {arr.shape} in {file_path}")

            # Auto-detect channel layout: (C, H, W) vs (H, W, C)
            if arr.shape[0] == NUM_BANDS:
                pass  # already channel-first
            elif arr.shape[2] == NUM_BANDS:
                arr = np.transpose(arr, (2, 0, 1))  # convert to (C, H, W)
            else:
                raise ValueError(
                    f"Tile has invalid band count: shape {arr.shape}. Expected {NUM_BANDS} bands (either (13,H,W) or (H,W,13))."
                )

        elif ext in [".tif", ".tiff"]:
            if not HAS_RASTERIO:
                raise ImportError("rasterio is required to load GeoTIFF tiles. Install via 'pip install rasterio'.")
            with rasterio.open(full_path) as src:
                if src.count != NUM_BANDS:
                    raise ValueError(
                        f"GeoTIFF has {src.count} bands; expected exactly {NUM_BANDS} bands in {file_path}"
                    )
                arr = src.read()  # rasterio read() returns (bands, height, width)

        else:
            raise ValueError(f"Unsupported tile file format '{ext}' in {file_path}. Must be .npy or .tif.")

        return arr.astype(np.float32)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalizes tile data to [0.0, 1.0]."""
        max_val = float(np.max(arr)) if arr.size > 0 else 0.0

        if max_val <= 1.05:
            # Already in [0, 1] range
            if not self._logged_scaling:
                print(f"[DATA] Tile values already normalized in [0.0, 1.0] (max={max_val:.3f}). Skipping divisor.")
                self._logged_scaling = True
            norm_arr = np.clip(arr, 0.0, 1.0)
        else:
            # Raw digital numbers (e.g. 0 - 10000)
            if not self._logged_scaling:
                print(f"[DATA] Applying L2A scale divisor {self.scale_divisor} (detected max={max_val:.1f}).")
                self._logged_scaling = True
            norm_arr = np.clip(arr / self.scale_divisor, 0.0, 1.0)

        return norm_arr

    def _ensure_spatial_size(self, tensor: torch.Tensor) -> torch.Tensor:
        """Resizes or crops/pads tile to target_size x target_size if dimensions differ."""
        _, h, w = tensor.shape
        if h != self.target_size or w != self.target_size:
            if not self._logged_resizing:
                warnings.warn(
                    f"[DATA WARNING] Tile spatial size ({h}, {w}) differs from expected ({self.target_size}, {self.target_size}). Resizing via bilinear interpolation.",
                    UserWarning,
                )
                self._logged_resizing = True
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.records.iloc[idx]
        tile_path = str(row["tile_path"])
        raw_label = row["label"]

        # Parse ground truth label
        if self.label_map is not None:
            if raw_label not in self.label_map:
                # Try string key
                str_key = str(raw_label).strip()
                if str_key in self.label_map:
                    target_class = int(self.label_map[str_key])
                else:
                    raise KeyError(
                        f"Unknown label '{raw_label}' in row {idx}. Label map keys: {list(self.label_map.keys())}"
                    )
            else:
                target_class = int(self.label_map[raw_label])
        else:
            try:
                target_class = int(raw_label)
            except ValueError:
                if str(raw_label).strip() in LAND_COVER_CLASSES:
                    target_class = LAND_COVER_CLASSES.index(str(raw_label).strip())
                else:
                    raise ValueError(
                        f"Label '{raw_label}' cannot be parsed as int. Provide --label_map JSON mapping."
                    )

        if not (0 <= target_class < NUM_CLASSES):
            raise ValueError(
                f"Target class {target_class} out of bounds [0, {NUM_CLASSES - 1}]. Check label or --label_map."
            )

        # Lazy load from disk
        raw_arr = self._load_tile_from_disk(tile_path)
        norm_arr = self._normalize(raw_arr)
        tile_tensor = torch.from_numpy(norm_arr).float()
        tile_tensor = self._ensure_spatial_size(tile_tensor)

        if self.transform is not None:
            tile_tensor = self.transform(tile_tensor)

        target_tensor = torch.tensor(target_class, dtype=torch.int64)
        return tile_tensor, target_tensor


def partition_manifest(
    manifest_df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    splits_output_path: Optional[str] = "checkpoints/splits.json",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions a manifest into reproducible train, val, and test splits.
    Handles explicit 'split' column or performs stratified / group-aware splitting.
    """
    df = manifest_df.copy()

    # 1. If explicit 'split' column exists
    if "split" in df.columns and df["split"].dropna().nunique() > 1:
        train_df = df[df["split"].astype(str).str.lower() == "train"].reset_index(drop=True)
        val_df = df[df["split"].astype(str).str.lower() == "val"].reset_index(drop=True)
        test_df = df[df["split"].astype(str).str.lower() == "test"].reset_index(drop=True)
        if len(val_df) == 0:
            train_df, val_df = train_test_split(train_df, test_size=val_ratio, random_state=seed)

    # 2. If 'location_id' exists, split by location group to prevent spatial data leakage
    elif "location_id" in df.columns and df["location_id"].nunique() > 3:
        unique_locs = df["location_id"].unique()
        np.random.seed(seed)
        np.random.shuffle(unique_locs)

        n_locs = len(unique_locs)
        n_test = max(1, int(n_locs * test_ratio))
        n_val = max(1, int(n_locs * val_ratio))

        test_locs = set(unique_locs[:n_test])
        val_locs = set(unique_locs[n_test : n_test + n_val])
        train_locs = set(unique_locs[n_test + n_val :])

        train_df = df[df["location_id"].isin(train_locs)].reset_index(drop=True)
        val_df = df[df["location_id"].isin(val_locs)].reset_index(drop=True)
        test_df = df[df["location_id"].isin(test_locs)].reset_index(drop=True)

    # 3. Standard stratified split by label
    else:
        labels = df["label"].values
        # Split out test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_ratio, random_state=seed, stratify=labels
        )
        # Split train and val
        train_val_labels = train_val_df["label"].values
        adjusted_val_ratio = val_ratio / (1.0 - test_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_ratio,
            random_state=seed,
            stratify=train_val_labels,
        )

    # Save split indices if path given
    if splits_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(splits_output_path)), exist_ok=True)
        splits_record = {
            "train_paths": train_df["tile_path"].tolist(),
            "val_paths": val_df["tile_path"].tolist(),
            "test_paths": test_df["tile_path"].tolist(),
            "train_count": len(train_df),
            "val_count": len(val_df),
            "test_count": len(test_df),
            "seed": seed,
        }
        with open(splits_output_path, "w", encoding="utf-8") as f:
            json.dump(splits_record, f, indent=2)

    return train_df, val_df, test_df
