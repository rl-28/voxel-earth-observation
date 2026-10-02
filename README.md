# VOXEL: Task-Adaptive Geospatial Feature Pipeline

## PROJECT OVERVIEW
VOXEL is a spatial-spectral satellite-data processing and land-cover analysis pipeline. The complete flow involves:
Satellite data → preprocessing → spectral band optimization → CNN-ViT modeling → explainability → FAISS retrieval.

The project focuses on:
- spatial-spectral processing
- spectral redundancy reduction
- classification
- computational efficiency
- explainability
- fast retrieval

There is a 16 GB hardware constraint as a project-level requirement.

## PROBLEM STATEMENT
Modern Earth observation produces large satellite datasets with many spectral bands, leading to redundant information. This poses challenges in computational cost and memory requirements. There is a critical need for efficient classification, explainability, and fast retrieval mechanisms.

## PROJECT OBJECTIVES
- satellite data acquisition
- preprocessing
- spatial tiling
- signal processing
- spectral band selection
- hybrid CNN-ViT classification
- explainability
- FAISS retrieval
- memory-aware processing

## SYSTEM ARCHITECTURE
```
Freeform Natural Language Query (e.g. "Assess flood risk in South Mumbai")
        ↓
Stage 0: Dynamic Intent & Spatial Geocoding Engine (Gemini / Nominatim)
        ↓
VoxelExecutionContract: [BBOX, Task Type, Target Band Budget, Scorecard Weights]
        ↓
Stage 1: Module 1 — Ingestion & Signal Processing (STAC, Tiling, Z-Score, 3D-DWT, Albedo)
        ↓
Standardized Tensor (256, 256, 13) Float32
        ↓
Stage 2: Module 2 — Metaheuristic Optimization Core (Physics Partitioning, Chaotic Seeding + OBL, BWOA)
        ↓
Reduced Tensor (256, 256, k) + 13-element Binary Mask (60-70% FLOP & Data Volume Reduction)
        ↓
Stage 3: Module 3 — Deep Learning & Neural Fusion (Hybrid CNN + Vision Transformer)
        ↓
Dense Feature Vector (1, 1216) + Explainable AI (SHAP / Integrated Gradients)
        ↓
Stage 4: FAISS High-Speed Vector Retrieval & Semantic Search
```

## TEAM STRUCTURE
| Member | Module | Role | Main Responsibilities |
|---|---|---|---|
| Arya Suryavanshi | Module 1 | Ingestion & Signal Processing Lead | STAC satellite acquisition, GeoTIFF processing, CRS alignment, 256x256 tiling, Z-score preprocessing, 3D-DWT signal processing, albedo/glare filtering |
| Shreeya Joshi | Module 2 | Optimization & Math Core Lead | Physics spectrum grouping, Chebyshev chaotic seeding, Opposition-Based Learning (OBL), Binary Whale Optimization Algorithm (BWOA), Green-AI fitness scorecard, domain balance guard |
| Riya Lepkar | Module 3 | Deep Learning & Search Architect | Hybrid CNN backbone, Tiny Vision Transformer, CNN-ViT neural fusion, SHAP explainability, FAISS indexing/retrieval |

## GIT BRANCH STRATEGY
The project uses separate feature branches for independent module development.

| Member | Module | Branch | Purpose |
|---|---|---|---|
| Arya | Module 1 | feature/ingestion-signal-processing | Ingestion & Signal Processing |
| Shreeya | Module 2 | feature/optimization-math-core | Optimization & Math Core |
| Riya | Module 3 | feature/deep-learning-search | Deep Learning & Search |

Workflow:
```
main
  │
  ├── feature/ingestion-signal-processing
  │
  ├── feature/optimization-math-core
  │
  └── feature/deep-learning-search
```
Each member works on their own branch, commits only their module changes, keeps their module independently testable, uses the agreed interface contract, pushes their branch, and creates a Pull Request when the module is ready. Integration is performed into main only after review/testing.

## MODULE INTERFACE CONTRACTS

### MODULE 1 → MODULE 2:
- **Type**: NumPy ndarray
- **dtype**: float32
- **Shape**: (256, 256, 13)
- **Axis Order**: (height, width, spectral_bands)
- **Values**: All values must be finite (no NaN or Inf).

This spatial-spectral data contract is the fixed interface between Module 1 and Module 2. Validation is explicitly enforced before handoff.

### MODULE 2 → MODULE 3:
- **Type**: Binary mask
- **Length**: 13
- **Values**: 0 or 1

### MODULE 3 → FINAL SYSTEM:
- **Type**: Dense feature vector
- **Shape**: (1, 1216)

## ARYA'S MODULE 1
### `stac_downloader.py`
**Purpose**: Programmatic satellite-data acquisition through the intended STAC workflow.
**Responsibilities**: STAC API connection, search/query, scene selection, asset selection, controlled download, metadata preservation.

### `tile_processor.py`
**Purpose**: Spatial preprocessing and tiling.
**Responsibilities**: GeoTIFF reading, CRS verification, CRS alignment/reprojection where required, non-overlapping 256×256 tiling, invalid-value handling, Z-score preprocessing, metadata handling.

### `wavelet_3d.py`
**Purpose**: 3D-DWT-based spatial-spectral signal processing.
**Responsibilities**: 3D transformation, multi-resolution analysis, coefficient handling, spatial/spectral detail extraction, validation/reconstruction experiments.

### `albedo_filter.py`
**Purpose**: Spatial filtering intended to suppress unwanted effects (solar reflection/glare, atmospheric scattering, cloud shadows) while preserving useful albedo-related information.

## MODULE 1 DATA FLOW
```
Raw Satellite Data
        ↓
STAC Downloader
        ↓
GeoTIFF / Raster
        ↓
CRS Verification / Alignment
        ↓
256×256 Non-overlapping Tiling
        ↓
Invalid Value Handling
        ↓
Z-score Processing
        ↓
3D-DWT
        ↓
Albedo Filtering
        ↓
Validation
        ↓
Float32
(256,256,13)
        ↓
Module 2
```

## PHASE 3 — STAC DOWNLOADER
Programmatic STAC-based ingestion of Sentinel-2 Level-2A surface reflectance data.
- **Provider**: Copernicus Data Space Ecosystem (configurable STAC endpoint).
- **Collection**: `sentinel-2-l2a`.
- **Search Capabilities**: Configurable AOI (GeoJSON), date interval, and cloud-cover threshold.
- **Asset Download**: Raw asset download functionality with memory-disciplined streaming.
- **Separation of Concerns**: The downloader discovers and downloads raw data; it does NOT crop, tile, or produce the (256, 256, 13) spatial tensor (handled by downstream preprocessing).
- **Security**: Credentials (if required) are injected via environment variables and never stored in Git.

> [!WARNING]
> Sentinel-2 L2A provides 12 surface-reflectance spectral bands. The project interface currently requires 13 channels. The mapping of the 13th channel is a project-level specification decision and must not be fabricated by the ingestion pipeline.
> - **L1C / MSI**: 13 spectral bands including B10.
> - **L2A**: 12 surface-reflectance spectral bands because B10 is excluded.

**Phase 3 Status**:
- STAC ingestion = implemented
- Copernicus L2A discovery = implemented
- L2A asset discovery = implemented
- 12-band L2A spectral mapping = verified
- 13-channel final mapping = pending project clarification

## PHASE 4 — GEOSPATIAL RASTER PROCESSING & TILING
Transforms raw geospatial raster assets into spatially aligned, non-overlapping 256×256 tiles, matching a specific reference grid. Maintains a 12-band L2A intermediate representation prior to the final 13-channel project specification.
- **GeoTIFF Inspection**: Metadata evaluation without full-scene loading.
- **CRS & Transform Handling**: Rasterio-driven coordinate reference alignment.
- **Resolution Harmonization**: Dynamic resampling via bilinear interpolation for heterogeneous resolutions (e.g., Sentinel-2 10m vs 20m vs 60m).
- **Reference Grid**: The pipeline aligns all inputs rigidly against a defined target grid (`B04` reference typically).
- **Tiling & Edge Policy**: Strict non-overlapping 256x256 window generation, discarding incomplete edge windows. Geospatial transforms are retained per tile.
- **L2A Intermediate State**: Generates a 12-band stack exactly preserving the L2A verified spectral bounds. Note: Normalization is deferred to later stages.

## PHASE 5 — Z-SCORE NORMALIZATION
Provides mathematically robust statistical standardisation across the spatial-spectral tile arrays, independent of source size.
- **Per-Band Standardisation**: Mean and standard deviation are calculated independently per spectral band ($Z = (X - \mu) / \sigma$).
- **Per-Tile Scope**: Baseline statistics are scoped to the tile level rather than a global scene.
- **Valid-Pixel Masking**: Nodata/invalid pixels are completely excluded from statistical calculations and Z-score transformations.
- **Zero-Valued Valid Pixels**: Zeros are treated correctly as physical measurements, not nodata, unless explicitly masked.
- **Constant Band Handling**: Handled strictly via numerical tolerance to prevent division by zero or NaN creation.
- **Output Contract**: Results are always finite, cast precisely to `float32`, returning an explicitly safe tensor and validity mask for downstream consumption.
- **Multi-Shape Support**: Agnostically supports both the 12-band real Sentinel-2 intermediate data and the final 13-channel project interface.

## PHASE 6 — 3D DISCRETE WAVELET TRANSFORM (3D-DWT)
Implements true 3D spatial-spectral wavelet decomposition to compress spectral bands and capture edge/textural information while preserving the original spatial scale.
- **Transform**: PyWavelets `wavedecn` over Height, Width, and Spectral axes.
- **Configuration**: Uses the lightweight `db1` (Haar) wavelet by default with a level-1 decomposition and `symmetric` boundary extension.
- **Spatial Preservation**: Reconstruction of the approximation back to the original `(256, 256, C)` shape is exposed to downstream users, guaranteeing spatial fidelity.
- **12/13-Band Agnosticism**: Supports 12-channel Sentinel-2 intermediate datasets and the 13-channel final project specification without assuming fixed spectral dimensionality or fabricating data.
- **Validation**: Enforces mathematical correctness via finite-input constraints and exact inverse reconstruction tests with errors close to numerical precision.

## PHASE 7 — ALBEDO / REFLECTION / ATMOSPHERIC FILTERING
Applies physically motivated reflection and contamination suppression to surface-reflectance tensors. This module identifies residual anomalies rather than replacing upstream Sentinel-2 L2A atmospheric correction processors.
- **Glare / Reflection Anomaly Detection**: Identifies unusually high broadband reflectance across visible (B03), NIR (B08), and SWIR (B11) channels.
- **Atmospheric Anomaly Detection**: Flags likely residual atmospheric scattering/haze based on isolated extreme coastal/blue responses (B01, B02).
- **Cloud-Shadow Anomaly Detection**: Detects shadow-like contamination via abnormally low (but non-zero) visible and NIR response.
- **Valid-Mask Handling**: Invalid pixels are safely excluded from thresholding and statistics.
- **Non-Destructive Masking**: Contaminated pixels are explicitly masked using `FILTER_FILL_VALUE`, while spatial scale `(256, 256, C)` is strictly preserved. Generates comprehensive `AlbedoFilterResult` diagnostics.
- **Scale-Aware Thresholds**: Dynamically applies appropriate thresholds based on whether the input is raw reflectance or Z-score normalized data.

## PHASE 8 — MODULE 1 END-TO-END PIPELINE INTEGRATION
Orchestrates the individual Module 1 components into a clean, testable processing pipeline:
- **Conceptual Flow**: GeoTIFF Ingestion -> CRS/Resolution Alignment -> 256x256 Tiling -> Valid Mask Extraction -> Z-Score Normalization -> 3D-DWT -> Albedo/Reflection Filtering -> Final Validation.
- **12/13-Band Support**: Explicitly handles both the real Sentinel-2 12-band intermediate data and the project's final 13-channel tensor contract, safely verifying dimensions and numeric bounds without hallucinating bands.
- **Rich Diagnostics**: Outputs a comprehensive `Module1Result` preserving all upstream statistics, contamination masks, processing timings, and structural metadata.
- **Tile-Oriented Memory Model**: Processes imagery incrementally tile-by-tile via a generator API to strictly respect the 16 GB hardware guardrail.

## PHASE 9 — OUT-OF-DOMAIN EVALUATION DATA PIPELINE
Establishes the robust data pipeline required to prepare geographically separated evaluation tiles (Serbia / Western Balkans) for out-of-domain evaluation. 
- **Separation**: Saves outputs to `data/evaluation` to strictly prevent data leakage into development/training datasets.
- **Unified Processing**: Reuses the exact same Module 1 preprocessing pipeline (CRS Alignment, Z-Score, 3D-DWT, Albedo Filtering).
- **Projection Verification**: Inspects and records geospatial metadata (CRS, bounds, transform) for every tile, verifying that it aligns with the expected output spatial grid.
- **CLI and Config**: Provides a reusable `module1_ingestion.evaluation_pipeline` to process evaluation regions without fabricating exact AOIs or coordinates (project team provides actual sources).

## PHASE 10 — STREAMLIT FRONTEND INTEGRATION
Provides a clean, interactive Streamlit frontend (`app.py`) exposing the full Module 1 pipeline to the end-user.
- **Input Modes**: Supports local `.zip` uploads containing Sentinel-2 GeoTIFFs, as well as configured evaluation region (Ext SR / Ext WB) execution.
- **Processing Diagnostics**: Displays real-time, interactive expandable diagnostics including contract validation type, resulting shapes, masking stats, and timing information.
- **Module 2 Handoff Awareness**: Explicitly surfaces the unresolved 12-channel vs 13-channel project specification gap, warning the user if the intermediate 12-channel output needs future projection prior to Module 2 integration.
- **Startup**: `streamlit run app.py`

## PHASE 2 — SYNTHETIC DEVELOPMENT DATASET
A synthetic dataset is established to provide structured input for downstream Module 2 and Module 3 development before real satellite ingestion is fully implemented.

- **Purpose**: Development baseline for hyperspectral/geospatial pipelines.
- **Dataset Size**: 100 mock tiles.
- **Data Properties**: 13 spectral bands, 256 × 256 spatial dimensions, `float32` representation.
- **Randomization**: Deterministic seed (default 42) ensures reproducible tile generation.
- **Git Strategy**: This is a development-only dataset; generated binary tiles are strictly ignored from Git tracking.
- **Usage**: Allows downstream modules to begin integration testing against the strict spatial-spectral data contract.

## DATA DIRECTORIES
- `data/raw/`
- `data/intermediate/`
- `data/processed/`
- `data/synthetic/`
- `data/evaluation/`
- `data/evaluation/serbia/`
- `data/evaluation/western_balkans/`

## TESTING STRATEGY
Planned testing for:
- STAC
- GeoTIFF loading
- CRS
- tiling
- normalization
- wavelet processing
- albedo filtering
- final pipeline

Final output must eventually satisfy:
shape == (256,256,13)
dtype == float32
all values finite

## PERFORMANCE / MEMORY
- tile-based processing
- controlled loading
- avoiding unnecessary array copies
- Float32 use
- intermediate data management
- memory monitoring

Mention the 16 GB hardware constraint.

## EVALUATION TARGETS
- stated spectral band range of 4–15
- stated minimum data-volume savings of 70%
- >99% classification accuracy target
- <16 GB hardware-memory guardrail
- explainability target involving meaningful spectral contributions, especially NIR/SWIR relevance for vegetation/crops

**SPECIFICATION CLARIFICATION REQUIRED**:
The brief defines the input as 13 bands but states a selection range extending to 15 bands. Additionally, selecting 4 of 13 bands corresponds to approximately 69.23% dimensional reduction.
The team/faculty should clarify the intended interpretation before final evaluation.

## FOUR-WEEK ROADMAP
**WEEK 1 (Arya)**:
- STAC setup
- tiling
- synthetic dataset

**WEEK 2 (Arya)**:
- 3D-DWT
- albedo filtering

**WEEK 3 (Arya)**:
- Serbia evaluation tiles
- Western Balkans evaluation tiles
- projection verification
- integration

**WEEK 4 (Arya)**:
- preprocessing integration
- Streamlit linkage

## MODULE 2: METAHEURISTIC OPTIMIZATION CORE
Module 2 (`module2_optimization/`) implements the metaheuristic band selection and spectral dimensionality reduction engine. It translates standardized 13-band Sentinel-2 L2A tensors (`(256, 256, 13)`) into task-adaptive reduced tensors (`(256, 256, k)`) optimized for downstream hybrid CNN-ViT inference.

### 1. Architectural Components
1. **Physics-Guided Spectral Partitioning (`physics_partitioning.py`)**:
   - Groups 13 Sentinel-2 channels into physical domains:
     - `VISIBLE`: B02, B03, B04
     - `NIR_RED_EDGE`: B05, B06, B07, B08, B8A
     - `SWIR`: B11, B12
     - `AUXILIARY`: B01, B09, B10
2. **Green-AI Multi-Objective Fitness Scorecard (`fitness_scorecard.py`)**:
   - Multi-objective fitness formulation:
     $$\text{Fitness}(X) = w_{\text{acc}} \cdot \text{Accuracy} + w_{\text{jm}} \cdot \text{JM}_{\text{norm}} - w_{\text{band}} \cdot \left(\frac{\|X\|_1}{13}\right) - \text{OverBudgetPenalty} - \text{RAMPenalty}$$
   - Fast $k$-NN / surrogate accuracy evaluated on subsampled train/test partitions.
   - Regularized Jeffries-Matusita (JM) distance ($\epsilon = 10^{-6}$) with determinant safeguards.
   - Deterministic $K$-Means ($k=6$) pseudo-labeling when external annotations are omitted.
   - Hard step penalty (100.0) applied if estimated tensor footprint exceeds the 16.0 GB hardware ceiling.
   - Memoization cache keyed by binary mask byte sequences.
3. **Chebyshev Chaotic Seeding & Opposition-Based Learning (`chaotic_seeding.py`)**:
   - Generates ergodic initial populations in $[-1, 1]$ via Chebyshev map ($x_{k+1} = \cos(d \cdot \arccos(x_k))$).
   - Enforces spectral budget constraints ($1 \le \|X\|_1 \le \text{target\_band\_count}$) via deterministic bit repair.
   - Applies Opposition-Based Learning ($X_{\text{opp}} = 1 - X$) and selects the top $N$ fittest individuals.
4. **Binary Whale Optimization Algorithm (`bio_search.py`)**:
   - Sigmoid transfer function $S(v) = \frac{1}{1 + e^{-v}}$ for discrete bit updates.
   - Encircling prey, random exploration ($|A| \ge 1$), and logarithmic bubble-net spiral attack.
   - Monotonic best fitness convergence tracking with patience-based early stopping.
5. **Orchestrator & Physics Domain Balance Guard (`orchestrator.py`)**:
   - `run_module2_optimization()`: Clamps target counts to $[4, 13]$ with warnings, coordinates BWOA search, ensures `VISIBLE`, `NIR_RED_EDGE`, and `SWIR` representation, and calculates computational reduction metrics.

### 2. Computational Reduction Metrics & Formulas
- **Data Volume Reduction**:
  $$\text{Data Reduction (\%)} = \left(\frac{13 - k_{\text{selected}}}{13}\right) \times 100\%$$
- **FLOP Savings**:
  In CNN-ViT architectures, early patch-embedding and initial 2D/3D convolutional layers scale linearly with input spectral channels $C_{\text{in}}$. Reducing from 13 to $k$ bands yields:
  $$\text{FLOP Savings (\%)} = \left(\frac{13 - k_{\text{selected}}}{13}\right) \times 100\%$$

### 3. Usage Example
```python
from module2_optimization import run_module2_optimization, apply_module2_band_selection
from utils.llm_parser import VoxelExecutionContract

# Direct Orchestrator Execution
optimal_mask, selected_indices, metrics = run_module2_optimization(
    input_tensor=tensor_13_bands,  # Shape: (256, 256, 13)
    target_count=4,                # Target band budget
    seed=42
)

# Pipeline Integration Handoff
reduced_tensor, indices, metrics = apply_module2_band_selection(
    tensor_13_bands=tensor_13_bands,
    contract=stage0_contract,      # VoxelExecutionContract
    enable_module2=True            # Feature flag
)
# Result: reduced_tensor has shape (256, 256, 4)
```

## STAGE 0: NATURAL LANGUAGE QUERY COMPILER & GEOSPATIAL COMPILER
Stage 0 (`utils/llm_parser.py`) translates unstructured, freeform human text into a strictly validated `VoxelExecutionContract` configuring downstream ingestion, optimization, and modeling.

### 1. Architectural Flow
1. **Semantic Intent Extraction**:
   - Primary LLM: **Google Gemini 1.5 Flash** with JSON mode.
   - Secondary Fallbacks: OpenAI (`gpt-4o-mini`) and Groq (`llama-3.1-8b-instant`).
   - Offline Fallback: Deterministic keyword-boundary regex classifier.
2. **Spatial Geocoding to BBOX**:
   - Geocodes location queries via `Nominatim` (OpenStreetMap) with a `0.0125°` coordinate offset around center lat/lon (~2.56 km × 2.56 km spatial footprint matching 256×256 pixels at 10m/pixel Sentinel-2 resolution).
   - Default offline fallback: South Mumbai, India (`[72.8075, 18.9375, 72.8325, 18.9625]`).
3. **Task Profiles & Spectral Band Budgets**:
   - `WATER_MOISTURE` (4 bands: SWIR-1 B11, SWIR-2 B12, NIR B08, Green B03)
   - `CROP_HEALTH` (6 bands: Red-Edge B05–B07, NIR B08)
   - `URBAN_INFRASTRUCTURE` (5 bands: Visible B02–B04, SWIR B11)
   - `GENERAL_LANDCOVER` (8 bands: Visible, NIR, SWIR, Red-Edge)

---

## API MANAGEMENT & ENVIRONMENT SECURITY
The centralized API Management system (`utils/api_manager.py`) provides secure credential sourcing, diagnostic inspection, and offline resilience:
- **Environment Template (`.env.example`)**: Clean configuration template for all credentials.
- **Git Security**: `.gitignore` strictly ignores `.env`, `.env.*`, `*.key`, `*.pem`, `secrets.json`, and geospatial binary rasters (`*.tif`, `*.jp2`, `*.nc`).
- **Singleton `APIManager`**:
  - `api_manager.get_gemini_key()`, `api_manager.has_gemini()`
  - `api_manager.health_check()`: Returns service capability reports with safely masked keys (`AIza...2345`).

---

## TERMINAL TEST BENCH & INTERACTIVE CLI TOOL
An interactive command-line test bench (`scripts/demo_cli.py`) is provided for testing arbitrary natural language queries, live OpenStreetMap/Google Maps geocoding, Stage 0 contracts, and Module 2 Whale Optimization directly from the terminal.

### 1. Launch Interactive REPL Mode (Type any query in real time):
```bash
python scripts/demo_cli.py
```
> Prompts you with an interactive loop (`Enter Earth Observation Query > `) where you can type ANY query worldwide, view live geocoded coordinates `[min_lon, min_lat, max_lon, max_lat]`, extracted band budgets, and real-time spectral band selections!
> Type `examples` inside the loop for query inspiration, `status` for key diagnostics, or `exit` to quit.

### 2. Direct Query CLI Execution:
You can pass any custom query directly as a CLI argument:
```bash
# Agriculture / NDVI in Punjab
python scripts/demo_cli.py "Assess wheat crop health, nitrogen stress, and vegetation indices across Punjab, India"

# Urban Heat & Building Sprawl in Tokyo
python scripts/demo_cli.py "Analyze urban building sprawl and concrete heat island expansion in Tokyo, Japan"

# Tropical Rainforest Deforestation in the Amazon
python scripts/demo_cli.py "Detect tropical rainforest deforestation and canopy loss in Amazon Basin, Para, Brazil"

# Glaciology & Snow Cover in the Swiss Alps
python scripts/demo_cli.py "Map snow cover extent and glacier retreat around Zermatt, Switzerland"

# Coastal Flooding & Water Accumulation
python scripts/demo_cli.py "Predict water accumulation, inundation zones, and flood risk around South Mumbai, India"
```

### 3. Run Global Multi-Region Benchmark Showcase:
```bash
python scripts/demo_cli.py --batch
```
> Automatically benchmarks 5 distinct global biomes & domains (Agriculture, Urban, Water, Forestry, Glaciology) to compare live coordinates and domain-specific spectral channel reductions in a summary table.

---

## TEST SUITE EXECUTION
Run the full unit and integration test suite via standard Python unittest:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## PROJECT STATUS
- Repository: FOUNDATION IN PROGRESS
- Git Branch: feature/optimization-engine
- Stage 0 — Query Parser & Geospatial Compiler: **COMPLETED**
- API Management & Security Infrastructure: **COMPLETED**
- Phase 1 — Spatial-Spectral Contract: **COMPLETED**
- Phase 2 — Synthetic Dataset: **COMPLETED**
- Phase 3 — STAC Downloader: **COMPLETED**
- Phase 4 — Geospatial Tiling & CRS Alignment: **COMPLETED**
- Phase 5 — Normalization: **COMPLETED**
- Phase 6 — 3D-DWT: **COMPLETED**
- Phase 7 — Cloud & Shadow Masking: **COMPLETED**
- Phase 8 — Module 1 Pipeline Integration: **COMPLETED**
- Phase 9 — Out-of-Domain Evaluation Data Pipeline: **COMPLETED**
- Phase 10 — Streamlit Frontend/Backend Integration: **COMPLETED**
- Module 2 — Metaheuristic Optimization Core: **COMPLETED**


