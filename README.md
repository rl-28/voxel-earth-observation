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
Raw Satellite Data
        ↓
Module 1 — Arya
Ingestion & Signal Processing
        ↓
(256,256,13) Float32
        ↓
Module 2 — Shreeya
Band Optimization
        ↓
13-element Binary Mask
        ↓
Module 3 — Riya
CNN + ViT + XAI + FAISS
        ↓
(1,1216) Dense Feature Vector
        ↓
Fast Retrieval
```

## TEAM STRUCTURE
| Member | Module | Role | Main Responsibilities |
|---|---|---|---|
| Arya Suryavanshi | Module 1 | Ingestion & Signal Processing Lead | satellite data acquisition, GeoTIFF processing, CRS alignment, spatial tiling, Z-score preprocessing, 3D-DWT processing, albedo-related filtering, standardized output generation |
| Shreeya Joshi | Module 2 | Optimization & Math Core Lead | spectral-band grouping, chaotic seeding, Opposition-Based Learning, Binary Whale Optimization / Enhanced Jaya, fitness scorecard, spectral band selection |
| Riya Lepkar | Module 3 | Deep Learning & Search Architect | CNN backbone, Tiny Vision Transformer, CNN-ViT fusion, SHAP explainability, FAISS indexing/search |

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

## SYNTHETIC DATA STRATEGY
Synthetic data will be used for independent development.
Target synthetic baseline:
- 100 mock tiles
- 13 spectral bands
- 256×256 spatial dimensions
- Float32 representation

Each tile:
(256,256,13)

This allows Shreeya and Riya to develop independently while Arya develops the real ingestion pipeline.

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

## PROJECT STATUS
- Repository: FOUNDATION IN PROGRESS
- Git Branch: feature/ingestion-signal-processing
- Module 1 Structure: SETUP
- Algorithms: NOT STARTED
- Synthetic Dataset: NOT STARTED
- Real Satellite Data: NOT STARTED
- Integration: PENDING
- Streamlit: PENDING
- Final Evaluation: PENDING
