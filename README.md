# VOXEL: Task-Adaptive Earth Observation & Spectral Intelligence Pipeline

> Physics-Guided Band Selection and Hybrid CNN-ViT Multimodal Fusion for Satellite Earth Observation and Land Cover Classification.

---

## 🌍 Project Overview

**VOXEL** is an end-to-end spatial-spectral satellite data processing, feature optimization, and land cover classification system. Designed under strict hardware and computational efficiency constraints (16 GB RAM ceiling, edge-deployable training and inference), VOXEL compresses high-dimensional satellite imagery, isolates optimal task-relevant spectral bands, classifies terrain with a hybrid deep learning model, and provides sub-second similarity search with explainable AI attribution.

### End-to-End System Architecture

```text
               User Query / Task Definition
                            ↓
               [Stage 0] Natural Language Query Compiler & Geocoding (LLM + Nominatim)
                            ↓ Execution Contract
               [Module 1] Geospatial Ingestion & Signal Processing
                   (STAC → Reprojection → 256x256 Tiles → Z-Score → 3D-DWT → Albedo Filter)
                            ↓ Spatial-Spectral Tensor (256, 256, 13)
               [Module 2] Physics-Guided Metaheuristic Band Optimization
                   (Domains → Chaotic Seeding → Binary Whale Optimization → Scorecard)
                            ↓ 13-Element Binary Mask / Selected Indices
               [Module 3] Hybrid Deep Learning, Explainability & Retrieval
                   (FusionNet: XNANet CNN + Tiny-32 ViT → 1216-dim Embedding → 10 Classes)
                            ↓
               Outputs: Land Cover Class + SHAP Attribution + FAISS Vector Search
```

---

## 👥 Team Structure & Module Ownership

| Member | Module | Role | Core Responsibilities |
| :--- | :--- | :--- | :--- |
| **Arya Suryavanshi** | **Module 1** | Ingestion & Signal Processing Lead | STAC Sentinel-2 acquisition, raster tiling & CRS reprojection, Z-score normalization, 3D Discrete Wavelet Transform (3D-DWT), albedo/cloud filtering, Streamlit viewer. |
| **Shreeya Joshi** | **Module 2** | Optimization & Math Core Lead | Stage 0 query parser, physics-guided spectral partitioning, Chebyshev chaotic seeding with OBL, Binary Whale Optimization Algorithm (BWOA), Green-AI multi-objective fitness scorecard. |
| **Riya Lepkar** | **Module 3** | Deep Learning & Search Architect | Hybrid FusionNet (CNN + ViT), dynamic spectral slicing, PyTorch GPU training pipeline, SHAP channel explainability, FAISS cosine similarity vector search, integration contract. |

---

## 🛰️ Architecture Breakdown by Module

### Stage 0: Natural Language Query Compiler & Geospatial Geocoder
- **Semantic Intent Extraction (`utils/llm_parser.py`)**: Translates unstructured, freeform human text into a strictly validated `VoxelExecutionContract`.
  - Primary LLM: **Google Gemini 1.5 Flash** with JSON mode.
  - Secondary Fallbacks: OpenAI (`gpt-4o-mini`) and Groq (`llama-3.1-8b-instant`).
  - Offline Fallback: Deterministic keyword-boundary regex classifier.
- **Spatial Geocoding to BBOX**: Geocodes location queries via `Nominatim` (OpenStreetMap) with a `0.0125°` coordinate offset around center lat/lon (~2.56 km × 2.56 km spatial footprint matching 256×256 pixels at 10m/pixel Sentinel-2 resolution).
- **Task Profiles & Spectral Band Budgets**:
  - `WATER_MOISTURE` (4 bands: SWIR-1 B11, SWIR-2 B12, NIR B08, Green B03)
  - `CROP_HEALTH` (6 bands: Red-Edge B05–B07, NIR B08)
  - `URBAN_INFRASTRUCTURE` (5 bands: Visible B02–B04, SWIR B11)
  - `GENERAL_LANDCOVER` (8 bands: Visible, NIR, SWIR, Red-Edge)

### Module 1: Ingestion & Signal Processing
- **STAC Ingestion (`module1_ingestion/stac_downloader.py`)**: Automated search and downloading of Sentinel-2 L2A tiles via Copernicus Data Space (`pystac-client`).
- **Windowed Raster Tiling (`module1_ingestion/tile_processor.py`)**: Splits raw GeoTIFF assets into non-overlapping $256 \times 256$ tiles with bilinear CRS alignment (`rasterio`).
- **Z-Score Normalization (`module1_ingestion/normalization.py`)**: Per-band radiometric normalization: $Z = (X - \mu) / \sigma$.
- **3D Discrete Wavelet Transform (`module1_ingestion/wavelet_3d.py`)**: Multi-resolution spatial-spectral decomposition using Daubechies wavelets (`PyWavelets` `db1`).
- **Albedo & Reflection Filtering (`module1_ingestion/albedo_filter.py`)**: Masking of cloud glare, water specular reflection, and deep shadows based on reflectance thresholds.
- **Interactive UI (`app.py`)**: Streamlit interface for visual validation of tiles and wavelet sub-bands.

### Module 2: Metaheuristic Optimization Core
- **Physics-Guided Spectral Partitioning (`module2_optimization/physics_partitioning.py`)**: Groups 13 Sentinel-2 bands into 4 physical domains:
  - `VISIBLE`: B02, B03, B04
  - `NIR_RED_EDGE`: B05, B06, B07, B08, B8A
  - `SWIR`: B11, B12
  - `AUXILIARY`: B01, B09, B10
- **Green-AI Multi-Objective Fitness Scorecard (`module2_optimization/fitness_scorecard.py`)**:
  $$\text{Fitness}(X) = w_{\text{acc}} \cdot \text{Accuracy} + w_{\text{jm}} \cdot \text{JM}_{\text{norm}} - w_{\text{band}} \cdot \left(\frac{\|X\|_1}{13}\right) - \text{OverBudgetPenalty} - \text{RAMPenalty}$$
  - Fast $k$-NN / surrogate accuracy evaluated on subsampled train/test partitions.
  - Regularized Jeffries-Matusita (JM) distance ($\epsilon = 10^{-6}$) with determinant safeguards.
  - Deterministic $K$-Means ($k=6$) pseudo-labeling when external annotations are omitted.
  - Hard step penalty (100.0) applied if estimated tensor footprint exceeds the 16.0 GB hardware ceiling.
  - Memoization cache keyed by binary mask byte sequences.
- **Chebyshev Chaotic Seeding & Opposition-Based Learning (`module2_optimization/chaotic_seeding.py`)**:
  - Generates ergodic initial populations in $[-1, 1]$ via Chebyshev map ($x_{k+1} = \cos(d \cdot \arccos(x_k))$).
  - Enforces spectral budget constraints ($1 \le \|X\|_1 \le \text{target\_band\_count}$) via deterministic bit repair.
  - Applies Opposition-Based Learning ($X_{\text{opp}} = 1 - X$) and selects the top $N$ fittest individuals.
- **Binary Whale Optimization Algorithm (`module2_optimization/bio_search.py`)**:
  - Sigmoid transfer function $S(v) = \frac{1}{1 + e^{-v}}$ for discrete bit updates.
  - Encircling prey, random exploration ($|A| \ge 1$), and logarithmic bubble-net spiral attack.
  - Monotonic best fitness convergence tracking with patience-based early stopping.
- **Orchestrator & Physics Domain Balance Guard (`module2_optimization/orchestrator.py`)**:
  - Coordinates BWOA search, guarantees representation across `VISIBLE`, `NIR_RED_EDGE`, and `SWIR`, and calculates reduction metrics:
    $$\text{Data Reduction (\%)} = \left(\frac{13 - k_{\text{selected}}}{13}\right) \times 100\%$$
    $$\text{FLOP Savings (\%)} = \left(\frac{13 - k_{\text{selected}}}{13}\right) \times 100\%$$

### Module 3: Deep Learning, Explainability & Vector Retrieval
- **Hybrid FusionNet (`models/fusion_net.py`)**: Multimodal dual-branch neural architecture:
  - **XNANet CNN Backbone (`models/cnn_backbone.py`)**: Extracts 1024-dimensional spatial and local textural features.
  - **Tiny-32 Vision Transformer (`models/vit_transformer.py`)**: Extracts 192-dimensional global spectral attention representations.
  - **Feature Fusion**: Concat-linear projection creates a rich **1216-dimensional embedding** followed by a 10-class land cover head.
- **Dynamic Spectral Slicing**: Operates transparently on variable active channel counts (e.g. 4, 6, 8, or 13 bands) dictated by the Module 2 mask.
- **Permutation SHAP Attribution (`explainability/shap_engine.py`)**: Quantifies per-band importance scores to verify model reliance on physical spectral signatures.
- **FAISS Vector Index (`explainability/faiss_search.py`)**: Fast cosine similarity indexing of 1216-dim embeddings for real-time nearest-neighbor terrain search.

---

## 🛠️ Environment Setup & Installation

- **Certified Environment:** `Python 3.10.11`
- **Compute:** NVIDIA CUDA GPU (e.g. RTX 3050 Laptop GPU, CUDA 12.1) with CPU fallback

```bash
# 1. Clone repository
git clone https://github.com/your-org/voxel-earth-observation.git
cd voxel-earth-observation

# 2. Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install unified dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution & Command Reference

### 1. Stage 0 & Module 2 Interactive Test Bench
Run the interactive CLI REPL or test specific Earth observation queries:
```bash
# Launch interactive console
python scripts/demo_cli.py

# Run batch showcase across biomes (Agriculture, Urban, Water, Forest, Glaciers)
python scripts/demo_cli.py --batch

# Direct query execution
python scripts/demo_cli.py "Assess crop health in Punjab"
```

### 2. Module 1 Streamlit Interface
Launch the ingestion viewer to inspect Sentinel-2 scenes, tiling, and 3D wavelets:
```bash
streamlit run app.py
```

### 3. Synthetic Dataset Generation (Module 3)
Generate standardized spatial-spectral tiles and labels for testing:
```bash
python tools/make_fake_dataset.py --output_dir data/fake_dataset --num_samples 500
```

### 4. GPU Training (Module 3)
```bash
# Fast mock verification run
python train_gpu.py --mock --epochs 1 --checkpoint_dir checkpoints/_test_tmp

# Full GPU training on dataset with custom band mask
python train_gpu.py \
  --manifest data/fake_dataset/manifest.csv \
  --mask_path config/example_band_mask.json \
  --epochs 40 \
  --batch_size 8 \
  --lr 1e-4 \
  --amp
```

### 5. Explainability & FAISS Index Build
```bash
# Generate SHAP summary plots and build FAISS index from trained model
python build_explainability.py \
  --checkpoint checkpoints/best_fusion_net.pth \
  --manifest data/fake_dataset/manifest.csv \
  --output_dir checkpoints
```

### 6. Single-Tile Inference & Similar Terrain Retrieval
```bash
python inference.py \
  --input data/fake_dataset/tiles/tile_0000.npy \
  --checkpoint checkpoints/best_fusion_net.pth \
  --index checkpoints/terrain_index.faiss \
  --top_k 5
```

### 7. Running the Unified Test Suite
```bash
# Run all unit and integration tests across Module 1, 2, and 3
python -m pytest -q
```

---

## 📜 Integration Contract Reference
For interface contracts, tensor shapes, manifest specifications, and handoff protocols between modules, see [docs/INTEGRATION_CONTRACT.md](file:///c:/Users/Riya/VOXEL/voxel-earth-observation/docs/INTEGRATION_CONTRACT.md).
