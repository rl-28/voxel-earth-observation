"""
tools/optimize_india_bands.py

Runs Module 2 Metaheuristic Band Selection (BWOA) on real Indian satellite tiles
for downstream tasks: forest_cover and waterbody.

Constraints:
- Uses ONLY tiles from the TRAIN split (prevents data leakage from val/test AOIs).
- Balanced sampling across the two classes up to 400 tiles with fixed seed.
- Transposes tiles from (13, 256, 256) channels-first to (256, 256, 13) channels-last.
- Passes ground-truth task labels (0 and 1) broadcast across pixels (N, 256, 256).
- target_count=6, exclude_placeholder_b10=True.
- Exports mask JSON to checkpoints/masks/<task>_mask.json.
- Records search history, runtime, and baseline comparisons in results/masks/<task>_search.json.
"""

import os
import sys
import json
import time
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from config.constants import TASKS
from module2_optimization.config import INDEX_TO_BAND, TOTAL_BANDS
from module2_optimization.orchestrator import run_module2_optimization, save_band_mask
from module2_optimization.fitness_scorecard import FitnessEvaluator


def load_balanced_train_tensor(
    manifest_path: str,
    pos_label: str,
    neg_label: str,
    max_samples: int = 400,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """
    Loads up to max_samples balanced tiles from the TRAIN split only.
    Transposes from (13, 256, 256) to (256, 256, 13).
    Returns (tensor, labels_grid, tile_paths, tile_scalar_labels).
    """
    df = pd.read_csv(manifest_path)
    train_df = df[df["split"].astype(str).str.lower() == "train"].copy()

    df_pos = train_df[train_df["label"] == pos_label]
    df_neg = train_df[train_df["label"] == neg_label]

    n_pos = len(df_pos)
    n_neg = len(df_neg)
    half_samples = max_samples // 2
    per_class = min(half_samples, n_pos, n_neg)

    rng = np.random.default_rng(seed)
    idx_pos = rng.choice(n_pos, size=per_class, replace=False)
    idx_neg = rng.choice(n_neg, size=per_class, replace=False)

    sampled_pos = df_pos.iloc[idx_pos]
    sampled_neg = df_neg.iloc[idx_neg]

    sampled_df = pd.concat([sampled_pos, sampled_neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"  Train split available: {n_pos} {pos_label}, {n_neg} {neg_label}")
    print(f"  Balanced sample: {per_class} {pos_label} + {per_class} {neg_label} = {len(sampled_df)} tiles total")

    tiles = []
    scalar_labels = []
    tile_paths = []

    for _, row in sampled_df.iterrows():
        p = str(row["tile_path"])
        full_path = p if os.path.exists(p) else os.path.join("data", "india", p)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing tile file: {full_path}")

        arr = np.load(full_path, mmap_mode="r")
        # Ensure (13, 256, 256)
        if arr.shape != (13, 256, 256):
            raise ValueError(f"Unexpected tile shape {arr.shape} in {full_path}")

        # Transpose to channels-last: (256, 256, 13)
        arr_last = np.ascontiguousarray(np.transpose(arr, (1, 2, 0)), dtype=np.float32)
        tiles.append(arr_last)

        # Map binary label: positive=1, negative=0
        lbl = 1 if row["label"] == pos_label else 0
        scalar_labels.append(lbl)
        tile_paths.append(p)

    tensor = np.stack(tiles, axis=0)  # (N, 256, 256, 13)
    scalar_labels_arr = np.array(scalar_labels, dtype=np.int32)
    # Broadcast to (N, 256, 256) per pixel
    labels_grid = np.repeat(scalar_labels_arr[:, None, None], 256, axis=1).repeat(256, axis=2)

    return tensor, labels_grid, tile_paths, scalar_labels


def optimize_task(
    task_name: str,
    manifest_path: str,
    pos_label: str,
    neg_label: str,
    target_count: int = 6,
    max_samples: int = 400,
    seed: int = 42
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"RUNNING MODULE 2 OPTIMIZATION FOR TASK: {task_name.upper()}")
    print("=" * 80)

    # 1. Load data
    tensor, labels_grid, tile_paths, scalar_labels = load_balanced_train_tensor(
        manifest_path=manifest_path,
        pos_label=pos_label,
        neg_label=neg_label,
        max_samples=max_samples,
        seed=seed
    )
    print(f"  Input Tensor Shape : {tensor.shape} (dtype: {tensor.dtype})")
    print(f"  Labels Grid Shape  : {labels_grid.shape} (dtype: {labels_grid.dtype})")

    # Destination paths
    mask_dir = Path("checkpoints/masks")
    results_dir = Path("results/masks")
    mask_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    export_mask_path = mask_dir / f"{task_name}_mask.json"
    search_json_path = results_dir / f"{task_name}_search.json"

    # 2. Run Module 2 Optimization
    t0 = time.perf_counter()
    best_mask, selected_indices, opt_metrics = run_module2_optimization(
        input_tensor=tensor,
        target_count=target_count,
        labels=labels_grid,
        seed=seed,
        export_mask_path=export_mask_path,
        exclude_placeholder_b10=True,
    )
    runtime_sec = time.perf_counter() - t0

    # 3. Baseline Comparisons on the Same Sample using FitnessEvaluator
    evaluator = FitnessEvaluator(
        tensor=tensor,
        target_band_count=target_count,
        labels=labels_grid,
        seed=seed
    )

    # A. Selected optimal mask
    fit_sel, metrics_sel = evaluator.evaluate(best_mask)

    # B. All 13 bands mask
    all_13_mask = np.ones(13, dtype=np.int32)
    fit_all, metrics_all = evaluator.evaluate(all_13_mask)

    # C. Random 6-band mask (excluding placeholder B10)
    rng = np.random.default_rng(seed + 100)
    available_indices = [i for i in range(13) if i != 10]
    rand_selected = sorted(rng.choice(available_indices, size=target_count, replace=False).tolist())
    rand_mask = np.zeros(13, dtype=np.int32)
    rand_mask[rand_selected] = 1
    fit_rand, metrics_rand = evaluator.evaluate(rand_mask)

    selected_bands = [INDEX_TO_BAND[i] for i in selected_indices]
    rand_bands = [INDEX_TO_BAND[i] for i in rand_selected]
    all_bands = [INDEX_TO_BAND[i] for i in range(13)]

    # 4. Save search metadata
    search_record = {
        "task": task_name,
        "algorithm": "BWOA",
        "seed": seed,
        "target_count": target_count,
        "runtime_seconds": float(runtime_sec),
        "selected_indices": [int(i) for i in selected_indices],
        "selected_bands": selected_bands,
        "data_reduction_pct": opt_metrics["data_reduction_pct"],
        "flop_savings_pct": opt_metrics["flop_savings_pct"],
        "fitness_history": [float(h) for h in opt_metrics["convergence_history"]],
        "selected_mask_metrics": {
            "fitness": float(fit_sel),
            "accuracy": float(metrics_sel["accuracy"]),
            "jm_distance": float(metrics_sel["jm_distance"]),
            "jm_normalized": float(metrics_sel["jm_normalized"]),
            "band_penalty": float(metrics_sel["total_band_penalty"]),
        },
        "all_13_bands_metrics": {
            "indices": list(range(13)),
            "bands": all_bands,
            "fitness": float(fit_all),
            "accuracy": float(metrics_all["accuracy"]),
            "jm_distance": float(metrics_all["jm_distance"]),
            "jm_normalized": float(metrics_all["jm_normalized"]),
            "band_penalty": float(metrics_all["total_band_penalty"]),
        },
        "random_6_band_metrics": {
            "indices": [int(i) for i in rand_selected],
            "bands": rand_bands,
            "fitness": float(fit_rand),
            "accuracy": float(metrics_rand["accuracy"]),
            "jm_distance": float(metrics_rand["jm_distance"]),
            "jm_normalized": float(metrics_rand["jm_normalized"]),
            "band_penalty": float(metrics_rand["total_band_penalty"]),
        },
        "domain_distribution": opt_metrics["domain_distribution"]
    }

    with open(search_json_path, "w", encoding="utf-8") as f:
        json.dump(search_record, f, indent=2)

    print("\n" + "-" * 80)
    print(f"RESULTS FOR {task_name.upper()}:")
    print("-" * 80)
    print(f"  Runtime            : {runtime_sec:.2f} seconds")
    print(f"  Selected Bands (6) : {selected_bands} (indices: {selected_indices})")
    print(f"  Data Reduction     : {opt_metrics['data_reduction_pct']:.2f}% ({TOTAL_BANDS} -> {len(selected_indices)} bands)")
    print(f"  Exported Mask      : {export_mask_path}")
    print(f"  Search History JSON: {search_json_path}")
    print()
    print("  Comparison on Same Sample:")
    print("  Configuration      | Active Bands | Accuracy | JM Norm | Penalty | Fitness")
    print("  " + "-" * 70)
    print(f"  Optimal Mask (BWOA)| {len(selected_indices):>12} | {metrics_sel['accuracy']:>8.4f} | {metrics_sel['jm_normalized']:>7.4f} | {metrics_sel['total_band_penalty']:>7.4f} | {fit_sel:>7.4f}")
    print(f"  Random 6-Band Mask | {len(rand_selected):>12} | {metrics_rand['accuracy']:>8.4f} | {metrics_rand['jm_normalized']:>7.4f} | {metrics_rand['total_band_penalty']:>7.4f} | {fit_rand:>7.4f}")
    print(f"  All 13 Bands       | {13:>12} | {metrics_all['accuracy']:>8.4f} | {metrics_all['jm_normalized']:>7.4f} | {metrics_all['total_band_penalty']:>7.4f} | {fit_all:>7.4f}")
    print("-" * 80)

    delta_vs_random = fit_sel - fit_rand
    delta_vs_all13 = fit_sel - fit_all
    print(f"  Fitness Delta vs Random 6: {delta_vs_random:+.4f} ({'BETTER' if delta_vs_random > 0 else 'NOT BETTER'})")
    print(f"  Fitness Delta vs All 13  : {delta_vs_all13:+.4f} ({'BETTER' if delta_vs_all13 > 0 else 'NOT BETTER'})")
    print("=" * 80 + "\n")

    return search_record


def main():
    print("================================================================================")
    print("VOXEL MODULE 2 BAND OPTIMIZATION ON REAL INDIAN TILES")
    print("================================================================================")

    # 1. Forest Cover Task
    res_forest = optimize_task(
        task_name="forest_cover",
        manifest_path="data/india/manifest_forest_cover.csv",
        pos_label="Forest",
        neg_label="Non-forest",
        target_count=6,
        max_samples=400,
        seed=42
    )

    # 2. Waterbody Task
    res_water = optimize_task(
        task_name="waterbody",
        manifest_path="data/india/manifest_waterbody.csv",
        pos_label="Water",
        neg_label="Non-water",
        target_count=6,
        max_samples=400,
        seed=42
    )

    print("ALL MODULE 2 OPTIMIZATIONS COMPLETE.")


if __name__ == "__main__":
    main()
