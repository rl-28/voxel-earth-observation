# VOXEL Framework Integration Notes (`integration/real-data`)

**Date:** 2026-10-10  
**Integration Branch:** `integration/real-data`  
**Base Branch:** `feature/module3-gpu-training` (tagged `pre-integration-backup` at commit `6a9142e`)  
**Merged Ingestion Branch:** `origin/feature/ingestion-signal-processing` (merge commit `b19bc16`)  
**Merged Optimization Branch:** `origin/feature/optimization-engine` (merge commit `203dd92`)  
**Environment:** Certified `Python 3.10.11`, `torch 2.5.1+cu121` (CUDA True, RTX 3050 Laptop GPU), `numpy 2.2.6`, `rasterio 1.4.4`.

---

## 1. Executive Summary

Both remote feature branches were sequentially merged into the dedicated integration branch `integration/real-data` using `--no-ff` merge commits:
1. **Merge 1 (Ingestion):** Incorporated Module 1 (STAC downloading, raster window tiling, Z-score normalization, 3D-DWT Daubechies wavelets, albedo/reflection filtering, Streamlit visualizer).
2. **Merge 2 (Optimization):** Incorporated Stage 0 (Natural Language Query Compiler & OpenStreetMap Nominatim geocoding) and Module 2 (physics-guided spectral partitioning, Chebyshev chaotic population seeding, Binary Whale Optimization Algorithm, Green-AI multi-objective scorecard, and interactive terminal CLI bench).

Full automated verification succeeded: **153/153 unit and integration tests passed**, 1-epoch GPU mock training verified on CUDA, and `pip check` confirmed zero dependency conflicts.

---

## 2. Conflicts Encountered & Exact Resolutions

### A. `tests/test_pipeline.py` (Merge 1)
- **Problem:** Both Module 1 and Module 3 authored a test file at path `tests/test_pipeline.py`. Base tested FusionNet deep learning epochs, checkpoint save/restore, and CUDA training; Ingestion tested `Module1Pipeline` tile generation.
- **Resolution:** Retained Base's `tests/test_pipeline.py` intact. Extracted Ingestion's pipeline tests from `origin/feature/ingestion-signal-processing:tests/test_pipeline.py` and saved them as `tests/test_module1_pipeline.py`. Both suites run independently without collisions.

### B. `.gitignore` (Merge 1 & Merge 2)
- **Problem:** Base required keeping the source package `data/` tracked while ignoring `data/fake_dataset/`, `data/raw/`, and `data/processed/`. Teammates added broad ignores (`*.npy`, `*.tif`, `*.jp2`, `*.pt`, `.env`, `secrets.json`).
- **Resolution:** Formed a strict union. The `data/` package (`data/__init__.py`, `data/datasets.py`) remains tracked. Local generated datasets (`data/fake_dataset/`, `data/raw/`, `data/processed/`, `data/synthetic/*.npy`, `data/evaluation/**/*.npy`, `data/evaluation/**/*.tif`) are ignored. Model checkpoints (`*.pth`, `*.faiss`) and environment secrets (`.env`, `secrets.json`, `credentials.json`, `*.key`) are completely ignored while keeping `.env.example` tracked.

### C. `requirements.txt` (Merge 1 & Merge 2)
- **Problem:** Base strictly pins PyTorch CUDA, NumPy, Rasterio, SHAP, and FAISS. Teammates added unpinned packages (`PyWavelets`, `pystac-client`, `streamlit`, `google-generativeai`, `geopy`).
- **Resolution:** Preserved exact pins for the core training/inference stack:
  ```text
  torch>=2.5.1
  torchvision>=0.20.1
  timm==1.0.30
  numpy==2.2.6
  pandas==2.3.3
  scikit-learn==1.7.2
  scipy==1.15.3
  shap==0.49.1
  faiss-cpu==1.15.1
  rasterio==1.4.4
  pytest==9.1.1
  ```
  Appended teammates' dependencies with verified pinned versions:
  ```text
  PyWavelets==1.8.0
  pystac-client==0.9.0
  streamlit>=1.20.0
  google-generativeai>=0.8.0
  geopy>=2.4.0
  ```

### D. `utils/__init__.py` (Merge 2)
- **Problem:** Base exported band mask loader utilities (`load_band_mask`, `default_band_mask`, etc.). Optimization exported Stage 0 parser and API manager classes (`VoxelExecutionContract`, `APIManager`, etc.).
- **Resolution:** Merged `utils/__init__.py` to export both sets of public symbols without namespace collision.

### E. `README.md` (Merge 1 & Merge 2)
- **Problem:** Three competing overviews with diverging architectural focuses.
- **Resolution:** Authored a unified master documentation document structured across all four workflow stages:
  - Stage 0: Natural Language Query Compiler & Geocoding
  - Module 1: Ingestion & Signal Processing
  - Module 2: Physics-Guided Band Optimization (BWOA)
  - Module 3: Deep Learning, Explainability & FAISS Retrieval

---

## 3. Technical Audit Findings (Pre-Integration Analysis)

### Audit 1: Normalization Output vs. `data/datasets.py`
- **Mechanism:** In `module1_ingestion/normalization.py`, `normalize_zscore()` standardizes pixel reflectances using $Z = (X - \mu) / \sigma$. This routinely produces negative values (approximately 50% of valid pixels).
- **Behavior in Loader:** In `data/datasets.py`, `_normalize()` checks `max_val = np.max(arr)`.
  - For Z-scores, `max_val` typically falls between $2.0$ and $4.0$. Because `max_val > 1.05`, the loader assumes raw Digital Numbers ($0\text{--}10000$) and applies `np.clip(arr / 10000.0, 0.0, 1.0)`.
  - **Result:** All negative pixels are clipped to **0.0**, and positive pixels are reduced to infinitesimal values near $0.0003$.
  - If `max_val <= 1.05`, the loader applies `np.clip(arr, 0.0, 1.0)`, which still **destroys all negative values by clamping them to 0.0**.
- **Action Required:** Align normalization. Either Module 1 must offer a mode outputting $[0.0, 1.0]$ physical surface reflectance ($\text{DN} / 10000.0$), or Module 3's loader must detect Z-score normalized inputs and bypass clipping.

### Audit 2: 12-Band vs. 13-Band Interface Mismatch
- **Origin:** Sentinel-2 Level-2A (Bottom-of-Atmosphere surface reflectance) excludes Band 10 (cirrus, 1375 nm) because cirrus absorption is an atmospheric phenomenon not reflected on the ground.
- **Discrepancy Locations:**
  - **Module 1 (`module1_ingestion/config.py`):** Explicitly defines 12 spectral bands (`B01–B09, B8A, B11, B12`). Outputs arrays of shape `(256, 256, 12)`.
  - **Module 2 (`module2_optimization/config.py`):** Expects 13 channels (`EXPECTED_CHANNELS = 13`) with `B10` at index 10. `validate_input_tensor()` raises a `ValueError` if passed a 12-channel tensor.
  - **Module 3 (`config/constants.py`):** Canonical order includes `B10` at index 10 (`NUM_BANDS = 13`). `data/datasets.py` rejects tiles with $\ne 13$ bands.
- **Resolution Options:**
  1. *Ingestion Level-1C Fetch:* Ingestion queries the Sentinel-2 L1C product specifically for Band 10 and resamples it to 256×256. (True 13 bands).
  2. *Synthetic Placeholder:* Ingestion inserts a zero-filled channel at index 10. (Maintains 13 channels without architecture changes).
  3. *Project-Wide 12-Band Standardization:* Formally remove Band 10 across Module 2, Module 3, and constants. (Cleanest physically, but requires updating constants and re-validating model shapes).

---

## 4. Security Audit

- **Tracked Files Scan:** Searched git tree for `.env`, `secrets.json`, credentials, and private keys. Only the clean template `.env.example` is tracked.
- **Regex Secret Scan:** Scanned for Google Gemini (`AIzaSy...`), OpenAI (`sk-...`), and Groq (`gsk_...`) API keys across all tracked files. Zero hardcoded secrets were detected.
- **Credentials Infrastructure:** Credentials are exclusively read dynamically from environment variables (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`) via `utils.api_manager.APIManager`.

---

## 5. Verification Results

| Verification Check | Target / Command | Result |
| :--- | :--- | :--- |
| **Complete PyTest Suite** | `python -m pytest -q` | **153 passed**, 76 warnings in 171.63s |
| **CUDA GPU Mock Training** | `python train_gpu.py --mock --epochs 1` | **Passed on NVIDIA RTX 3050** (1 epoch completed, VRAM 0.20 GB, checkpoint verified) |
| **Environment Integrity** | `torch`, `numpy`, `rasterio` verification | `torch 2.5.1+cu121` (CUDA True), `numpy 2.2.6`, `rasterio 1.4.4` |
| **Dependency Consistency**| `python -m pip check` | **No broken requirements found** |
| **Git Working Tree** | `git status --short` | Clean |

---

## 6. What Is Still Missing Before Real Training Can Run

1. **Real Tile Data on Disk:** No real satellite imagery is committed (only `.gitkeep` placeholders exist). A pipeline run or staging step must populate `data/tiles/` with real tiles.
2. **Ground Truth Classification Labels:** Neither teammate produces land-cover labels. A labeled raster dataset (e.g. EuroSAT / Dynamic World) must be aligned with the tiles to populate integer labels ($0\text{--}9$).
3. **`manifest.csv` File:** A tabular index containing columns `tile_path`, `label`, `split`, and `location_id` must be generated.
4. **Exported Optimization Mask on Disk:** Module 2 executes in-memory. An exported mask file (`checkpoints/bwoa_selected_bands.json`) must be saved to disk for Module 3's `--mask_path` flag.
5. **Resolution on Band 10 / 12-Band Interface:** Team consensus needed on Option 1, 2, or 3 from Section 3.
6. **Resolution on Radiometric Scaling:** Decision needed on whether Module 3 trains on Z-score features or $[0.0, 1.0]$ surface reflectance.

---

## 7. Open Questions for Teammates

### For Arya (Teammate 1 — Ingestion & Signal Processing)
1. **Band 10 Resolution:** Sentinel-2 L2A STAC provides 12 bands, but Modules 2 and 3 expect 13. Will you fetch Band 10 from L1C, insert a placeholder channel, or should the project officially transition to 12 bands?
2. **Radiometric Scale:** Your normalization produces Z-scores centered around 0 with negative numbers, whereas Module 3's loader clips negative values to 0.0 assuming $[0.0, 1.0]$ surface reflectance. Can Module 1 output unclipped reflectances in $[0.0, 1.0]$ ($\text{DN} / 10000.0$), or should Module 3 adapt to Z-score inputs?
3. **Labels & Manifest:** Where will the EuroSAT ground-truth land-cover labels come from, and can Module 1 generate the standard `manifest.csv` referencing them?

### For Shreeya (Teammate 2 — Optimization Core)
1. **Mask Serialization:** Can you add a utility function in Module 2 that exports the optimal BWOA mask to a JSON file matching the schema in `config/example_band_mask.json`?
2. **Algorithm Roadmap:** The integration contract mentioned Jaya and RF-DE alongside BWOA. Are those algorithms still scheduled, or is BWOA the final selected method?
3. **12-Band vs. 13-Band Expectation:** If Module 1 supplies 12 bands, can Module 2's `physics_partitioning.py` and `config.py` be updated to operate on 12 bands directly?
