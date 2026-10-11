# VOXEL: Task-Adaptive Earth Observation & Spectral Intelligence Pipeline

> Physics-Guided Band Selection and Hybrid CNN-ViT Multimodal Fusion for Satellite Earth Observation and Land Cover Classification.

---

## 🌍 Project Overview

**VOXEL** is an end-to-end spatial-spectral satellite data processing, feature optimization, and land cover classification system. Designed under strict hardware and computational efficiency constraints (16 GB RAM ceiling, edge-deployable training and inference), VOXEL ingests high-dimensional Sentinel-2 multispectral imagery, isolates task-relevant spectral bands using metaheuristic optimization, classifies terrain with a hybrid deep learning model, and provides similarity search with explainable AI attribution.

### End-to-End System Architecture

```text
               User Query / Task Definition ("Track waterbody levels in Loktak Lake, Manipur")
                                     ↓
               [Stage 0] Natural Language Query Compiler & Geospatial Geocoder (LLM + Nominatim)
                                     ↓ Execution Contract (Target Coordinates, Spectral Budget, Task Profile)
               [Module 1] Geospatial Ingestion, Provenance & Signal Processing
                   (STAC Search → L2A Surface Reflectance → Tile Crop → Z-Score → 3D-DWT → Albedo Filter)
                   • Calibrated RGB Composite (B04-Red, B03-Green, B02-Blue)
                   • Physical Spectral Indices: NDWI = (B03-B08)/(B03+B08) | NDVI = (B08-B04)/(B08+B04)
                   • Measured Spectral Reflectance Signature (443nm–2190nm Tile-Mean BOA ± 1σ Variance)
                                     ↓ Calibrated Spatial-Spectral Tensor (256, 256, 13)
               [Module 2] Physics-Guided Metaheuristic Band Optimization
                   (Spectral Domains → Chebyshev Chaotic Seeding → Binary Whale Optimization (BWOA))
                   • Task-Specific Saved Benchmarks (Waterbody: 6 Bands, Forest: 5 Bands)
                   • Spectral Mask Provenance: Saved Checkpoint-Compatible vs. Fresh Optimization
                                     ↓ 13-Element Binary Mask / Selected Band Indices
               [Module 3] Hybrid Deep Learning, Explainability & Retrieval
                   (FusionNet: XNANet CNN + Tiny-32 ViT → 1216-dim Embedding → Class Probabilities)
                   • Measured Pipeline Latency & Dynamic Inference Timing
                   • Mask-Compatibility Gating ("Prediction unavailable" on incompatible custom masks)
                   • SHAP Permutation Attribution & FAISS Cosine Similarity Vector Index
                                     ↓
               Outputs: Land Cover Classification + Spectral Diagnostics + Vector Nearest Neighbors
```

---

## 👥 Team Structure & Module Ownership

| Member | Module | Role | Core Responsibilities |
| :--- | :--- | :--- | :--- |
| **Arya Suryavanshi** | **Module 1** | Ingestion & Signal Processing Lead | STAC Sentinel-2 acquisition, raster tiling & CRS reprojection, Z-score normalization, 3D Discrete Wavelet Transform (3D-DWT), albedo/cloud filtering, calibrated spectral index calculation (NDWI/NDVI), and interactive locator interface. |
| **Shreeya Joshi** | **Module 2** | Optimization & Math Core Lead | Stage 0 query parser, physics-guided spectral partitioning, Chebyshev chaotic seeding with OBL, Binary Whale Optimization Algorithm (BWOA), Green-AI multi-objective fitness scorecard, and task-specific mask provenance. |
| **Riya Lepkar** | **Module 3** | Deep Learning & Search Architect | Hybrid FusionNet (CNN + ViT), dynamic spectral slicing, PyTorch training pipeline, measured latency telemetry, honest prediction gating, SHAP channel explainability, and FAISS vector search. |

---

## 🛰️ Architecture Breakdown by Module

### Stage 0: Natural Language Query Compiler & Geospatial Geocoder
- **Semantic Intent Extraction (`utils/llm_parser.py`)**: Translates unstructured, freeform human text into a strictly validated `VoxelExecutionContract`.
  - Primary LLM: **Google Gemini 1.5 Flash** with JSON mode.
  - Secondary Fallbacks: OpenAI (`gpt-4o-mini`) and Groq (`llama-3.1-8b-instant`).
  - Offline Fallback: Deterministic keyword-boundary regex classifier.
- **Spatial Geocoding to BBOX**: Geocodes location queries via `Nominatim` (OpenStreetMap) with a `0.0125°` coordinate offset around center lat/lon (~2.56 km × 2.56 km spatial footprint matching 256×256 pixels at 10m/pixel Sentinel-2 resolution).
- **Task Profiles & Spectral Band Budgets**:
  - `WATER_MOISTURE` (4–6 bands: SWIR-1 B11, SWIR-2 B12, NIR B08/B8A, Green B03, Coastal B01)
  - `CROP_HEALTH` / `FOREST_COVER` (5–6 bands: Red-Edge B05–B07, NIR B08, Red B04, Blue B02)
  - `URBAN_INFRASTRUCTURE` (5 bands: Visible B02–B04, SWIR B11)
  - `GENERAL_LANDCOVER` (8 bands: Visible, NIR, SWIR, Red-Edge)

### Module 1: Ingestion, Provenance & Signal Processing
- **STAC Ingestion & Provenance (`module1_ingestion/stac_downloader.py`)**: Automated search and ingestion of Sentinel-2 L2A tiles via Copernicus Data Space and AWS Earth Search (`pystac-client`). Recovers genuine Scene IDs (`S2A_46REN_...`), acquisition timestamps, and cloud-cover metadata.
- **Windowed Raster Tiling (`module1_ingestion/tile_processor.py`)**: Crops raw GeoTIFF assets into non-overlapping $256 \times 256$ tiles with bilinear CRS alignment (`rasterio`).
- **Physical Spectral Indices**: Calculates calibrated reflectance ratios from the identical ingested tensor:
  - **NDWI (Normalized Difference Water Index)**: $\frac{\text{B03} - \text{B08}}{\text{B03} + \text{B08}}$ (Waterbody tasks)
  - **NDVI (Normalized Difference Vegetation Index)**: $\frac{\text{B08} - \text{B04}}{\text{B08} + \text{B04}}$ (Forest-cover tasks)
  - Full numeric safeguarding against nodata values, division-by-zero, and out-of-gamut reflectance.
- **Measured Spectral Reflectance Signature**: Extracts tile-mean surface reflectance and spatial variance ($\pm 1\sigma$ envelope) across wavelengths (443 nm to 2190 nm), accurately reflecting surface biophysics while handling band spatial resampling ($10\text{m} / 20\text{m} / 60\text{m} \to 10\text{m}$) and explicitly refusing Band 10 (1375 nm atmospheric cirrus) to preserve bottom-of-atmosphere (BOA) surface reflectance integrity.
- **Z-Score Normalization (`module1_ingestion/normalization.py`)**: Per-band radiometric normalization: $Z = (X - \mu) / \sigma$.
- **3D Discrete Wavelet Transform (`module1_ingestion/wavelet_3d.py`)**: Multi-resolution spatial-spectral decomposition using Daubechies wavelets (`PyWavelets` `db1`).
- **Albedo & Reflection Filtering (`module1_ingestion/albedo_filter.py`)**: Masking of cloud glare, water specular reflection, and deep shadows based on reflectance thresholds.

### Module 2: Metaheuristic Optimization Core
- **Physics-Guided Spectral Partitioning (`module2_optimization/physics_partitioning.py`)**: Groups 13 Sentinel-2 bands into 4 physical domains:
  - `VISIBLE`: B02, B03, B04
  - `NIR_RED_EDGE`: B05, B06, B07, B08, B8A
  - `SWIR`: B11, B12
  - `AUXILIARY`: B01, B09, B10
- **Green-AI Multi-Objective Fitness Scorecard (`module2_optimization/fitness_scorecard.py`)**:
  $$\text{Fitness}(X) = w_{\text{acc}} \cdot \text{Accuracy} + w_{\text{jm}} \cdot \text{JM}_{\text{norm}} - w_{\text{band}} \cdot \left(\frac{\|X\|_1}{13}\right) - \text{OverBudgetPenalty} - \text{RAMPenalty}$$
  - Evaluates fast $k$-NN / surrogate accuracy, regularized Jeffries-Matusita (JM) distance, and penalizes band bloat.
  - Enforces hard step penalties (100.0) if estimated tensor memory exceeds the 16.0 GB hardware ceiling.
- **Chebyshev Chaotic Seeding & Opposition-Based Learning (`module2_optimization/chaotic_seeding.py`)**: Generates ergodic initial populations via Chebyshev map ($x_{k+1} = \cos(d \cdot \arccos(x_k))$) and applies Opposition-Based Learning ($X_{\text{opp}} = 1 - X$).
- **Binary Whale Optimization Algorithm (`module2_optimization/bio_search.py`)**: Metaheuristic discrete search with sigmoid transfer functions ($S(v) = \frac{1}{1 + e^{-v}}$), encircling prey, random exploration, and bubble-net spiral maneuvers.
- **Spectral Mask Provenance**: Distinguishes verified saved task masks (`checkpoints/masks/waterbody_mask.json` with 6 bands; `checkpoints/masks/forest_cover_mask.json` with 5 bands) from fresh optimization executions.

### Module 3: Deep Learning, Explainability & Vector Retrieval
- **Hybrid FusionNet (`models/fusion_net.py`)**: Multimodal dual-branch neural architecture:
  - **XNANet CNN Backbone (`models/cnn_backbone.py`)**: Extracts 1024-dimensional spatial and local textural features.
  - **Tiny-32 Vision Transformer (`models/vit_transformer.py`)**: Extracts 192-dimensional global spectral attention representations.
  - **Feature Fusion**: Concat-linear projection creates a rich **1216-dimensional embedding** followed by classification.
- **Dynamic Spectral Slicing**: Dynamically slices input channels based on the active BWOA mask.
- **Scientific Integrity & Prediction Gating**:
  - Replaces arbitrary latency claims with live **Measured Pipeline Latency** (e.g. `2.30s`).
  - Gated prediction safeguards: If a custom optimization mask produces band channel counts incompatible with trained checkpoints, the UI explicitly reports `"Prediction unavailable"` and `N/A` metrics rather than fabricating an inference result.
- **Permutation SHAP Attribution (`explainability/shap_engine.py`)**: Quantifies per-band importance scores to verify model reliance on physical spectral signatures.
- **FAISS Vector Index (`explainability/faiss_search.py`)**: Fast cosine similarity indexing of 1216-dim embeddings for real-time nearest-neighbor terrain search.

---

## 📊 Scientific Verification & Real-World Evaluation

The system has been trained and evaluated on real Sentinel-2 Level-2A imagery across Indian geographic regions. Evaluation metrics are loaded dynamically from verified test split evaluation artifacts:

| Task | Target Region | Active BWOA Bands | Spectral Reduction | Held-Out Test Accuracy | Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Waterbody Classification** | Loktak Lake, Manipur | B01, B03, B04, B06, B8A, B11 (6 bands) | **53.8%** | **73.68%** | **73.49%** |
| **Forest Cover Classification** | Jim Corbett National Park, UK | B02, B04, B06, B11, B12 (5 bands) | **61.5%** | **94.74%** | **94.74%** |

*Note: Held-out test performance is evaluated on a rigorous 19-tile geographic test split per task. Whole-tile predictions perform binary presence classification ($256 \times 256$ footprint) and do not claim continuous canopy density percentages or volumetric water depth measurements.*

---

## 🛠️ Environment Setup & Installation

- **Certified Environment:** `Python 3.10` / `Python 3.11`
- **Compute:** NVIDIA CUDA GPU (e.g. RTX 3050 Laptop GPU, CUDA 12.1) with CPU fallback

```bash
# 1. Clone repository
git clone https://github.com/rl-28/voxel-earth-observation.git
cd voxel-earth-observation

# 2. Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install unified dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution & Command Reference

### 1. Launching the VOXEL Dashboard
Launch the unified 5-tab scientific platform:
```bash
streamlit run app.py
```
Key Dashboard Capabilities:
- **Tab 1: Overview**: System architecture, data flow diagrams, hardware constraints, and team contributions.
- **Tab 2: Satellite Analysis**: Natural-language query interface with verified Indian research presets (Loktak Lake, Jim Corbett), interactive OpenStreetMap locator, real Sentinel-2 RGB composite, side-by-side NDWI/NDVI spectral indices, measured spectral reflectance profile, BWOA band matrix, and live FusionNet inference.
- **Tab 3: Spectral Band Optimization**: Standalone BWOA optimization lab with chaotic seeding, convergence curves, and spectral volume reduction analysis.
- **Tab 4: Model Intelligence**: Deep learning architecture inspectability, live channel slicing, SHAP channel attribution, and FAISS nearest-neighbor terrain retrieval.
- **Tab 5: Research & Evaluation**: Rigorous evaluation benchmarks loaded dynamically from held-out test artifacts, confusion matrices, and scientific disclosures.

### 2. Live Inference Verification via Script
Verify live inference on real cached Sentinel-2 Level-2A data:
```bash
python tools/test_live_inference.py
```

### 3. Stage 0 & Module 2 Interactive Test Bench
```bash
# Launch interactive console
python scripts/demo_cli.py

# Direct query execution
python scripts/demo_cli.py "Track waterbody levels in Loktak Lake, Manipur"
```

### 4. GPU Training (Module 3)
```bash
python train_gpu.py \
  --manifest data/fake_dataset/manifest.csv \
  --mask_path checkpoints/masks/waterbody_mask.json \
  --epochs 40 \
  --batch_size 8 \
  --lr 1e-4 \
  --amp
```

### 5. Running the Unified Test Suite
Run all unit, integration, and pipeline regression tests:
```bash
pytest -q
```
*(Certified: 181 passed tests across all pipeline modules).*

---

## 📜 Integration Contract Reference
For interface contracts, tensor shapes, manifest specifications, and handoff protocols between modules, see [docs/INTEGRATION_CONTRACT.md](file:///d:/Voxel/voxel-earth-observation/docs/INTEGRATION_CONTRACT.md).
