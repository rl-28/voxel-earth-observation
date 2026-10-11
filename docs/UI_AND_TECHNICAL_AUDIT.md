# VOXEL: UI AND TECHNICAL AUDIT REPORT
**Platform:** VOXEL Earth Observation Intelligent Core  
**Branch:** `feature/voxel-end-to-end-integration`  
**Date:** October 2026  
**Auditor:** VOXEL Autonomous Core Engineering Team  
**Review Target:** Professor & Stakeholder Evaluation Readiness  

---

## EXECUTIVE SUMMARY

This audit documents the comprehensive architectural, visual, and scientific refactoring of the VOXEL Earth Observation platform. 

VOXEL is an intelligent satellite processing pipeline designed for multi-spectral Sentinel-2 imagery over India. The pipeline unifies:
- **Stage 0:** Natural Language Query Compilation and Nominatim Geocoding
- **Module 1:** Cloud-Optimized GeoTIFF (COG) STAC Ingestion and Surface Reflectance Scaling
- **Module 2:** Physics-Guided Binary Whale Optimization Algorithm (BWOA) for Spectral Band Selection
- **Module 3:** Deep CNN–Vision Transformer (FusionNet) Land Cover Classification
- **Explainability & Retrieval:** Permutation SHAP Spectral Channel Attribution & FAISS Embedding Retrieval

The primary objective of this review was to resolve severe visual readability conflicts (dark text on dark backgrounds, clipped navigation tabs, blank orchestrator space) and critical scientific integrity issues (model-mask incompatibility during live BWOA optimization, single-tile batch SHAP fallback errors, and ungrounded performance claims).

All original teammate algorithms and real-data checkpoints have been strictly preserved. The application is now fully verified, scientifically honest, and ready for academic presentation.

---

## SECTION 1: UI & AESTHETICS OVERHAUL

### 1.1 Root Cause of Visual Failures
Prior inspections revealed several UI defects in the Streamlit interface:
1. **Dark Text on Dark Backgrounds:** Streamlit's BaseWeb component hierarchy rendered dropdown selectboxes, popovers, radio buttons, and text input fields using dark inherited font colors (`#1a1f2c` or default gray) against navy/black container backgrounds (`#030712`).
2. **Horizontal Navigation Clipping:** The horizontal five-module tab bar overflowed on screens below 1920px width, clipping Modules 3, 4, and 5 and breaking accessibility.
3. **Large Unused Blank Panel:** The central orchestrator container rendered an empty 600px tall card prior to pipeline execution, giving the impression of an incomplete application.
4. **Oversized Hero Section:** An excessively tall landing header forced the essential query input and execution controls below the fold on standard 1080p and 768p displays.
5. **Wildcard CSS Overrides:** Global CSS rules (such as `[data-testid="stSidebar"] * { ... }`) unintentionally stripped interactive focus states and native Streamlit widget styling.

### 1.2 Implemented UI Solutions
A dedicated styling engine (`ui_theme.py`) was introduced to enforce high-contrast, professional visual standards while preserving the cinematic deep-space aesthetic:

- **High-Contrast BaseWeb Styling:**
  - Selectboxes, popover menus, and option items explicitly set text to `#f8fafc` (high-contrast off-white) and backgrounds to `#070e24` with electric cyan borders (`#00f0ff`).
  - Active and hovered options display `#00f0ff` text with translucent cyan highlights (`rgba(0, 240, 255, 0.15)`).
  - Radio button option labels and input field text explicitly styled with `#f8fafc` text and `#94a3b8` helper subtitles.
  - Metrics cards feature electric cyan titles, pure white bold values, and subdued neon borders.

- **Vertical Compact Sidebar Navigation:**
  - Replaced the overflowing horizontal tabs with a clean, vertical radio navigation drawer in the sidebar.
  - Constrained sidebar width to a compact **240–260px** (`[data-testid="stSidebar"] { min-width: 240px; max-width: 260px; }`).
  - All five functional modules are accessible in a single click:
    - `🚀 01 // Orchestrator`
    - `🔬 02 // Provenance & Verification`
    - `📡 03 // Module 1: Ingestion`
    - `⚡ 04 // Module 2: BWOA Optimization`
    - `🧠 05 // Module 3: Deep Learning`

- **Removal of Blank Space & System Briefing Card:**
  - In the Orchestrator, the previously empty placeholder was replaced with an informative **System Ready // Pipeline Briefing** card.
  - Displays the active platform status, real satellite ingestion pipeline readiness, calibrated checkpoint availability, and execution parameters.
  - Once executed, results are smoothly retained in `st.session_state["latest_pipeline_result"]`, preventing flickering or blank states during tab switches.

- **Scaled Hero Header:**
  - The cinematic orbital Earth visualization was scaled to a compact banner (140px height) featuring a dark starry gradient, subtle horizon curve, and glowing orbital crest.
  - The primary query inputs, task selector, and execution button are immediately visible above the fold on all standard laptop and desktop screens.

- **Multi-Resolution Responsiveness Testing:**
  - **1920 × 1080 (Full HD):** Spacious three-column diagnostic panels, zero horizontal scrollbars.
  - **1600 × 900 (Standard Desktop):** Balanced two-column grid, compact sidebar, crisp charts.
  - **1366 × 768 (Laptop Standard):** Single/double column adaptive collapse, controls immediately accessible.
  - **400 × 800 (Mobile/Narrow Viewport):** Sidebar collapses into native toggle drawer; cards and charts stack vertically without horizontal overflow.

---

## SECTION 2: AUDIT OF HARDCODED & MISLEADING CLAIMS

Every claim and output in `app.py`, `voxel_pipeline.py`, and related modules was audited and classified:

| Output / Claim | Previous Status | Audit Classification | Corrective Action Taken |
| :--- | :--- | :--- | :--- |
| **"178 tests passing"** | Displayed as a permanent live status | Misleading Static Claim | Replaced with verified CI/Pytest provenance label: *"178 Unit & Integration Tests Verified in CI / Pytest Environment — 100% Passing"*, with neutral live status. |
| **"0% Data Leakage"** | Stated without geographical context | Unsupported Scientific Claim | Replaced with qualified empirical provenance: *"Spatial Isolation: 19-tile held-out test set in Hirakud has zero spatial tile overlap with training (Kaziranga, Western Ghats, Thar, Sardar Sarovar) and validation (Chilika). Note: preliminary experimental evidence, not nationwide generalization."* |
| **Forest Model Predictions** | Inferred even when BWOA chose incompatible band count | Technical Bug & False Claim | Fixed via Model-Mask Decoupling: candidate BWOA mask displayed separately while model classification applies its verified calibrated training mask. |
| **SHAP Attributions ($b=1$)** | Returned uniform 16.7% across all 6 bands | Misleading Mock Fallback | Fixed in `shap_engine.py`: implemented single-tile spatial pixel permutation. Informative attributions reflect genuine sensitivity; uninformative cases report `is_informative: False`. |
| **FAISS Reference Retrieval** | Labeled as Indian satellite reference scenes | Misleading Attribution | Clarified in UI and code: explicitly marked as *"Experimental EuroSAT Reference Corpus (25,000 Embeddings)"* for cross-domain similarity prototyping. |
| **BWOA Resource Savings** | Claimed hardware energy & memory savings | Unsubstantiated Claim | Qualified as *"Mathematical Input Dimension Reduction (53.8% to 61.5%)"*; removed speculative downlink/hardware energy claims. |
| **Execution Latencies** | Static sample figures in some views | Static Placeholder | Replaced with live millisecond timing breakdown from `res.timing_breakdown_ms`. |

---

## SECTION 3: CORE PIPELINE SCIENTIFIC INTEGRITY & BUG REPAIRS

### 3.1 Stage 0 — Query Interpretation & Scope Qualifications
- **Real Routing:** Stage 0 correctly resolves natural language queries via the Deterministic Offline Rule Engine (or Google Gemini API when configured) and geocodes coordinates via OpenStreetMap Nominatim.
- **Task Scope Disclaimers:** Added explicit scientific qualifications in `warnings` and diagnostics:
  - Queries querying *temporal water levels/depth* (e.g., *"Track waterbody levels in Loktak Lake"*): Emits notice that VOXEL performs whole-tile land-cover classification (Water vs Non-water at 256×256 px footprint) and does **not** quantify temporal water depth fluctuations or reservoir volume.
  - Queries querying *canopy density* (e.g., *"Assess forest canopy in Jim Corbett"*): Emits notice that VOXEL performs binary forest presence classification and does **not** measure continuous sub-pixel canopy closure percentages.

### 3.2 Module 1 — Real Satellite Ingestion
- Real STAC acquisitions determine the exact Sentinel-2 L2A scene ID (e.g., `S2A_46REN_20240522_0_L2A`), acquisition date, cloud cover percentage, and bounding box.
- Preserved windowed Cloud Optimized GeoTIFF (COG) acquisition via rasterio.
- Data cache status is transparently reported as either `Live Download` or `Cached on Disk (Zero Network IO)`.
- Sentinel-2 L2A surface reflectance is scaled to $[0.0, 1.0]$ with NoData handling and 13-band canonical ordering (Cirrus B10 zero-filled as designated non-surface placeholder).

### 3.3 Module 2 — Physics-Guided BWOA Band Selection
- Preserved the full BWOA optimization engine: Chebyshev chaotic initialization, Opposition-Based Learning (OBL), whale spiral/encircling search, binary discretization (sigmoidal/tanh transfer functions), and multi-objective fitness balancing accuracy and band sparsity.
- Live BWOA optimization executes genuinely when `--optimize-bands` or the dashboard toggle is enabled.
- Band reduction percentage is accurately computed based on active reflectance channels.

### 3.4 Module 3 — Critical Model-Mask Incompatibility Repair
#### The Bug:
- The trained Forest Cover checkpoint (`checkpoints/real_forest_cover/best_fusion_net.pth`) was calibrated on 5 bands: `[B02, B04, B06, B11, B12]`.
- When live BWOA ran on a new query scene, its stochastic search selected an exploratory candidate mask, such as `[B01, B04, B07, B12]` (4 bands).
- Passing 4 channels into a model trained on 5 channels caused input mismatch or produced uncalibrated logits, yet the system previously displayed the classification as if it were validated.

#### The Scientific Repair:
- In `voxel_pipeline.py`, we implemented **Scientific Model-Mask Decoupling**:
  1. If `optimize_bands=True`, live BWOA executes to discover an exploratory spectral subset for the query scene.
  2. If the discovered candidate mask differs from the model's calibrated training mask:
     - The exploratory BWOA candidate is recorded in `candidate_bwoa_bands` and `diagnostics["live_bwoa_candidate_mask"]`.
     - The pipeline applies the model's verified training mask for classification inference.
     - `res.mask_provenance` is set to `"live_bwoa_candidate_decoupled"`.
     - An explicit `mask_compatibility_note` is attached:
       > *"Live BWOA discovered candidate mask [B01, B04, B07, B12] (4 bands). Model classification inference applied verified training mask [B02, B04, B06, B11, B12] (5 bands) to preserve mathematical validity. Retraining is required to calibrate weights to arbitrary live candidate masks."*
  3. The UI presents both masks side-by-side: an exploratory optimization card showing the candidate bands discovered for this scene, and the calibrated inference card showing the actual bands passed to FusionNet.

---

## SECTION 4: EXPLAINABILITY & RETRIEVAL REPAIRS

### 4.1 Single-Tile SHAP Permutation Bug
- **Bug Analysis:** In `explainability/shap_engine.py`, permutation attribution previously shuffled features across the batch dimension: `perm_idx = torch.randperm(b)`. When evaluating a single tile ($b=1$), `torch.randperm(1) == [0]`, resulting in zero feature corruption. Consequently, all channels scored $\Delta = 0$, triggering a fallback that divided importance equally (flat 16.7% across all 6 bands).
- **The Repair:** When $b=1$, the engine now performs **spatial pixel permutation**:
  ```python
  spatial_flat = sample_images[0, band_idx].reshape(-1)
  pixel_perm = torch.randperm(h * w, device=device)
  permuted_images[0, band_idx, :, :] = spatial_flat[pixel_perm].reshape(h, w)
  ```
  This destroys the spatial-spectral coherence of the specific spectral channel while preserving its marginal intensity distribution.
- **Honest Attribution Reporting:** When channel impact is negligible, `is_informative` is set to `False` and zero percentages are reported, rather than faking equal attributions.

### 4.2 FAISS Embedding Corpus
- The 25,000-tile vector index was verified to originate from a EuroSAT multi-spectral reference benchmark.
- It is now clearly labeled as an **Experimental EuroSAT Reference Corpus** used for nearest-neighbor latent representation probing. It is not represented as validated Indian geographic imagery.

---

## SECTION 5: DATA AND PERFORMANCE VALIDATION

### 5.1 Dataset Provenance & Split Integrity
- **Ingestion Ground Truth:** Real Sentinel-2 L2A surface reflectance tiles paired with ESA WorldCover 10m land cover rasters.
- **Geographic Partitioning:**
  - Training scenes: Kaziranga, Western Ghats, Thar Desert, Sardar Sarovar.
  - Validation scene: Chilika Lake.
  - Held-out test scene: Hirakud Reservoir (19 tiles).
- **Zero Spatial Leakage:** Confirmed zero tile overlap between Hirakud and training/validation footprints.
- **Limitation Disclosed:** 19 held-out tiles constitute a localized test set for demonstration and preliminary evaluation; it does not demonstrate nationwide spatial generalization across all Indian agro-ecological zones.

### 5.2 Real Model Evaluation Summary

| Task Profile | Checkpoint Path | Architecture | Input Channels | Verified F1 Score | Evaluation Set |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Waterbody** | `checkpoints/real_waterbody/best_fusion_net.pth` | FusionNet (CNN + ViT) | 6 bands (`B01, B03, B04, B06, B8A, B11`) | **0.950** (Accuracy: 94.7%) | Real Sentinel-2 / ESA WorldCover (19 tiles) |
| **Forest Cover** | `checkpoints/real_forest_cover/best_fusion_net.pth` | FusionNet (CNN + ViT) | 5 bands (`B02, B04, B06, B11, B12`) | **0.783** (Accuracy: 78.9%) | Real Sentinel-2 / ESA WorldCover (19 tiles) |

*Full-band (12-band) and random-band baselines are designated as future comparative benchmarks rather than asserting unverified performance gains.*

---

## SECTION 6: INTEGRATION VERIFICATION & TEST OUTCOMES

### 6.1 Pytest Test Suite
The full VOXEL test suite was executed in the project virtual environment:
- **Total Tests:** 179
- **Passed:** 178
- **Skipped:** 1 (optional live STAC download test requiring live AWS credentials)
- **Failed:** 0
- **Execution Time:** 115.03 seconds

### 6.2 Targeted Regression Tests (`tests/test_voxel_pipeline.py`)
12 automated integration tests verify the core pipeline:
1. `test_stage0_to_module1_handoff`: Natural language parsing to geographic bounding box contract.
2. `test_module1_to_module2_tensor_conversion`: Surface reflectance scaling and channel ordering.
3. `test_spectral_mask_application_and_b10_refusal`: Refusal of placeholder Band 10 and mask reduction.
4. `test_module2_to_module3_compatibility`: Dynamic channel slicing and dimensional invariance in FusionNet.
5. `test_offline_smoke_test_end_to_end`: Complete offline pipeline execution and JSON persistence.
6. `test_real_mode_mock_checkpoint_gating`: Withholding of mock checkpoints in Real Mode.
7. `test_invalid_query_fallback`: Graceful fallback on unrecognized queries.
8. `test_canonical_band_order`: Invariance of 13 Sentinel-2 spectral keys.
9. `test_result_serialization_schema`: Complete schema serialization and JSON loadability.
10. `test_stage0_task_scope_qualifications`: Scope qualification for water depth and canopy density queries.
11. `test_bwoa_mask_checkpoint_decoupling`: Live BWOA exploratory candidate decoupling from calibrated model checkpoint.
12. `test_single_tile_spatial_shap_permutation`: Single-tile spatial pixel permutation SHAP attribution test.

### 6.3 Live Geographic Inference Verification (`tools/test_live_inference.py`)
Two distinct real-world Indian geographic queries were executed end-to-end:

#### Location 1: Loktak Lake, Manipur
- **Query:** *"Track waterbody levels in Loktak Lake, Manipur"*
- **Bounding Box:** `[93.8007, 24.5459, 93.8257, 24.5709]`
- **STAC Item ID:** `S2A_46REN_20240522_0_L2A` (Cloud Cover: 16.6%)
- **Data Source:** Real Sentinel-2 L2A Scene (Element84 Earth Search STAC)
- **Active Spectral Bands:** `['B01', 'B03', 'B04', 'B06', 'B8A', 'B11']` (53.8% data reduction)
- **Prediction:** **Water** (Confidence: **96.72%**)
- **Provenance:** `real_inference`
- **Total Pipeline Latency:** 2,521.02 ms
- **Scope Qualification:** Qualified that whole-tile classification does not quantify temporal water depth/volume.

#### Location 2: Jim Corbett National Park, Uttarakhand
- **Query:** *"Assess forest canopy in Jim Corbett National Park, Uttarakhand"*
- **Bounding Box:** `[78.83, 29.545, 78.855, 29.57]`
- **STAC Item ID:** `S2A_44RKT_20240619_0_L2A` (Cloud Cover: 1.22%)
- **Data Source:** Real Sentinel-2 L2A Scene (Element84 Earth Search STAC)
- **Active Spectral Bands:** `['B02', 'B04', 'B06', 'B11', 'B12']` (61.5% data reduction)
- **Prediction:** **Forest** (Confidence: **58.43%**)
- **Provenance:** `real_inference`
- **Total Pipeline Latency:** 2,104.21 ms
- **Scope Qualification:** Qualified that whole-tile classification does not produce continuous canopy density percentages.

---

## SECTION 7: PROFESSOR REVIEW READINESS & DEMO GUIDE

### 7.1 Features Safe to Present
1. **Interactive Query Orchestrator:** Enter natural language queries across India (e.g., Chilika Lake, Loktak Lake, Jim Corbett, Thar Desert).
2. **Real Satellite Ingestion:** Demonstrating live/cached STAC metadata, cloud cover, acquisition dates, and real RGB surface reflectance composites.
3. **Band Selection Engine:** Displaying active spectral bands, data volume reduction percentages, and the role of NIR/SWIR bands.
4. **Deep Learning Classification:** FusionNet classification using verified real-data weights with class probabilities and confidence scores.
5. **Model-Mask Compatibility:** Explaining the architectural reason for decoupling live BWOA candidate exploration from model inference.
6. **Provenance & Verification Hub:** Full transparency into checkpoint metadata, dataset manifests, test results, and scientific limitations.

### 7.2 Known Limitations to Disclose Openly
- **Task Scope:** The model classifies whole 256×256 tiles as dominant land cover (Water or Forest); it is not a sub-pixel canopy density estimator or bathymetric depth gauge.
- **Generalization:** Evaluated on a 19-tile held-out set from Hirakud; nationwide deployment would require multi-season, multi-region fine-tuning.
- **Reference Corpus:** The FAISS nearest-neighbor index uses EuroSAT embeddings as an experimental feature retrieval demonstration.
