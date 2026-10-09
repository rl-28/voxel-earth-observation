# VOXEL Module 3: Merge-Readiness Verification Report

> **Branch:** `feature/module3-gpu-training`  
> **Status:** **MERGE-READY (Pre-Merge Preparation Complete)**  
> **Verified Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM, CUDA 12.1)  
> **Python Runtime:** Python 3.10.11  

---

## 1. Step-by-Step Verification Checklist

| Step | Milestone | Status | Key Output / Verification Proof |
| :---: | :--- | :---: | :--- |
| **Step 1** | **Integration Contract** | **PASS** | Created `docs/INTEGRATION_CONTRACT.md`, extracted shared constants to `config/constants.py`, created `config/example_band_mask.json`. Unified all imports. |
| **Step 2** | **Pluggable Band Mask Loader** | **PASS** | Created `utils/band_mask.py` with multi-format support (`.json`, `.npy`, `.pt`), length/binary validation, descriptive logging, and warning fallback. Dynamic channel support in `models/fusion_net.py`. Passed 8 tests across 3, 6, 9, 13 bands. |
| **Step 3** | **Pluggable Real-Data Dataset** | **PASS** | Created `data/datasets.py` (`RealSentinel2Dataset`) with lazy loading, automated channel-first/last detection, scaling divisor auto-detection, spatial augmentation, label mapping, and stratified group splitting. Added synthetic generator `tools/make_fake_dataset.py`. |
| **Step 4** | **Hardened GPU Training Pipeline** | **PASS** | Refactored `train_gpu.py` into CLI: `--mock`, `--manifest`, `--mask_path`, `--epochs`, `--batch_size`, `--lr`, `--num_workers` (Windows fallback to 0), `--grad_accum_steps`, `--device`. Added AMP, CosineAnnealingLR, early stopping (patience 10), class imbalance weighting (>3x), and OOM handler. Saves full metadata in `checkpoints/best_fusion_net.pth` and logs to `results/training_log.csv`. |
| **Step 5** | **Evaluation & Mask Comparison** | **PASS** | Created `evaluate.py` (test split OA, Macro-F1, mIoU, per-class metrics, `results/confusion_matrix.png`). Created `tools/compare_masks.py` comparing 13 bands vs optimized vs random control mask (`results/band_selection_comparison.md`). |
| **Step 6** | **Decoupled Explainability & Retrieval** | **PASS** | Refactored `explainability/shap_engine.py` and `explainability/faiss_search.py` to be data-agnostic without internal mocks. Created `build_explainability.py` CLI to regenerate `checkpoints/shap_band_importance.png`, `checkpoints/terrain_index.faiss`, and metadata. Added `checkpoints/README.md`. |
| **Step 7** | **Pytest Automated Test Suite** | **PASS** | Complete automated test suite in `tests/test_band_mask.py`, `tests/test_dataset.py`, `tests/test_pipeline.py`. **18 of 18 tests passed** in 25.74s under `pytest -q`. |
| **Step 8** | **Repository Hygiene & Untracking** | **PASS** | Updated `.gitignore` to ignore `*.pth`, `*.pt`, `*.faiss`, `__pycache__/`, `data/`, `results/`. Untracked heavy binary checkpoints from Git index with `git rm --cached` while preserving local disk files. Pinned working dependencies + `pytest` in `requirements.txt`. |
| **Step 9** | **Merge-Readiness Report** | **PASS** | Final verification report documented below. `python train_gpu.py --mock` and `pytest -q` confirmed passing. |

---

## 2. Command Proof of Final Verification Runs

### A. Automated Test Suite (`pytest -q`)
```
> python -m pytest -q
..................                                                       [100%]
18 passed, 2 warnings in 25.74s
```

### B. Mock GPU Training Sanity Run (`python train_gpu.py --mock --epochs 2`)
```
================================================================================
 [VOXEL MODULE 3] HARDENED GPU TRAINING PIPELINE
================================================================================
Active Compute Device:      CUDA
GPU Hardware:               NVIDIA GeForce RTX 3050 4GB Laptop GPU (4.00 GB VRAM)
[INFO] Windows platform detected: Falling back from --num_workers=2 to 0 for stability.
Initializing synthetic Mock Sentinel-2 Dataset (--mock mode)...
Selected Band Mask:         6 active bands ([1, 2, 3, 7, 11, 12])
Active Spectral Bands:      ['B02 (Blue)', 'B03 (Green)', 'B04 (Red)', 'B08 (NIR)', 'B11 (SWIR 1)', 'B12 (SWIR 2)']
DataLoader Config:          Batch Size=8, Workers=0, GradAccum=1
--------------------------------------------------------------------------------
Epoch  | LR        | Train Loss | Val Loss  | Val OA(%) | Val F1(%) | Val mIoU(%) | Time(s) | VRAM (GB)
--------------------------------------------------------------------------------
1      | 1.00e-04  | 2.3421     | 2.3041    | 3.12      | 0.61      | 0.31        | 5.2     | 0.20/3.54
2      | 5.05e-05  | 2.3409     | 2.2946    | 9.38      | 1.71      | 0.94        | 3.5     | 0.20/3.54
================================================================================
 [TRAINING COMPLETE] Best Val F1: 1.71% (Saved to 'checkpoints\best_fusion_net.pth')
 Training log persisted to: 'results\training_log.csv'
================================================================================
```

---

## 3. Exact Commands to Run After Merging

Once Module 1 (Data Ingestion) and Module 2 (Band Optimization) branches arrive, run these exact commands without code modifications:

### 1. Train FusionNet with Real Ingested Tiles and Optimized Bands
```bash
python train_gpu.py \
  --manifest data/sentinel2_manifest.csv \
  --data_dir data/tiles \
  --mask_path config/optimized_band_mask.json \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --grad_accum_steps 1 \
  --checkpoint_dir checkpoints
```

### 2. Regenerate Real SHAP Attribution & FAISS Retrieval Database
```bash
python build_explainability.py \
  --checkpoint checkpoints/best_fusion_net.pth \
  --manifest data/sentinel2_manifest.csv \
  --data_dir data/tiles \
  --mask_path config/optimized_band_mask.json \
  --output_dir checkpoints
```

### 3. Evaluate Final Test Split Accuracy & Confusion Matrix
```bash
python evaluate.py \
  --checkpoint checkpoints/best_fusion_net.pth \
  --manifest data/sentinel2_manifest.csv \
  --data_dir data/tiles \
  --output_dir results
```

### 4. Benchmark Optimized Bands Against Full Spectrum (13 Bands)
```bash
python tools/compare_masks.py \
  --manifest data/sentinel2_manifest.csv \
  --mask_path config/optimized_band_mask.json \
  --epochs 10 \
  --batch_size 8 \
  --output_file results/band_selection_comparison.md
```

### 5. Run Live Inference and Similar Terrain Search
```bash
python demo_module3.py
```

---

## 4. Conflict Risk Analysis & Recommended Merge Order

### Suggested Merge Order
1. **Merge Module 1 (Data Ingestion) first:**  
   Ingestion creates data directories, STAC downloaders, and GeoTIFF tile stores. It does not touch model architectures or training loops.
2. **Merge Module 2 (Band Optimization) second:**  
   Optimization exports metaheuristic band masks (`.json` or `.npy`) and metric comparisons.
3. **Merge Module 3 (Deep Learning & Retrieval) third:**  
   Module 3 connects the ingestion pipeline (`--manifest`) with the optimization mask (`--mask_path`) to train and evaluate the model.

### Shared Files Modified in Module 3 (Heads-Up for Teammates)

| File | Change Description | Conflict Risk | Conflict Prevention Strategy |
| :--- | :--- | :---: | :--- |
| `models/fusion_net.py` | Added dynamic `selected_channels` parameter to support arbitrary band mask sizes while maintaining default 6 channels. | **Low** | Preserved exact class signature and 1216-dim embedding vector ($\text{CNN}_{1024} + \text{ViT}_{192}$). No changes made to `cnn_backbone.py` or `vit_transformer.py`. |
| `requirements.txt` | Pinned exact working versions and added `pytest`. | **Medium** | Teammates should accept Module 3's PyTorch 2.5 CUDA 12.1 and GDAL/rasterio version specifications. |
| `.gitignore` | Added `*.pth`, `*.pt`, `*.faiss`, `results/`, `data/`. | **Low** | Prevents GitHub pre-receive hook errors (>100MB blocks). |
| `README.md` | Added end-to-end CLI commands and architecture links. | **Low** | Can be merged cleanly as top-level documentation. |

---

## 5. Known Risks and Open Questions for Teammates

### Questions for Ingestion Teammate (Arya)
1. **Band Ordering:** Does the tile array follow the canonical Sentinel-2 13-band order:  
   `[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12]`?  
   *(Module 3 expects this exact order. If an order differs, provide a channel map in `manifest.csv`.)*
2. **Reflectance Scaling:** Are tile pixel values stored in raw L2A Digital Numbers (e.g. $[0, 10000]$) or already scaled to $[0.0, 1.0]$?  
   *(Module 3 auto-detects values already in $[0, 1]$ and skips division; if raw, it divides by `--scale_divisor 10000`.)*
3. **Class Labels:** Do ground truth labels match the 10 classes in `config/constants.py`? If numerical IDs differ, please provide a simple `label_map.json` mapping.

### Questions for Optimization Teammate (Shreeya)
1. **Mask Export Format:** Does the metaheuristic optimization script output the JSON schema defined in `config/example_band_mask.json` (`selected_indices`, `selected_bands`, `total_bands=13`)?  
   *(Module 3 also accepts `.npy` or `.pt` binary arrays of length 13.)*
2. **Index Base:** Confirm that selected band indices are **0-indexed** ($0 \le \text{idx} < 13$).
