# Module 1: Ingestion & Signal Processing Architecture

## 1. Purpose
This module handles satellite data acquisition, spatial preprocessing, and signal processing to produce normalized, standardized tensors for downstream evaluation.

## 2. Scope
Includes STAC search/download, GeoTIFF processing, CRS management, Z-score normalization, 3D-DWT processing, and albedo filtering.

## 3. Input
Raw satellite data accessed via STAC API (e.g., Sentinel-2 Level-2A).

## 4. Output
A normalized spatial-spectral tensor of shape `(256, 256, 13)` and dtype `float32`.

## 5. STAC Ingestion
Query, search, and controlled download of satellite scenes, preserving essential metadata.
The STAC downloader specifically discovers and downloads raw assets but **does NOT produce the final (256,256,13) tensor**.

Data Flow:
```text
STAC Catalog
     ↓
Search
     ↓
STAC Item
     ↓
Asset Discovery
     ↓
Raw Asset Download
     ↓
data/raw/
     ↓
Tile Processor
```

## 6. Geospatial Raster Processing & Tiling
Transforms raw geospatial raster assets into spatially aligned, non-overlapping 256×256 tiles. This phase strictly maintains a 12-band L2A intermediate representation prior to the final 13-channel project specification.

Data Flow:
```text
Raw Sentinel-2 Assets
        ↓
Raster Metadata Inspection
        ↓
Reference Grid
        ↓
CRS Alignment
        ↓
Resolution Harmonization
        ↓
Band Ordering
        ↓
Windowed 256×256 Extraction
        ↓
Geospatial Tile Metadata
        ↓
12-Band L2A Intermediate
        ↓
Future Normalization / Signal Processing
```
*(This is distinct from the final Module 1 (256,256,13) float32 contract)*

## 7. CRS Handling
Verifying Coordinate Reference Systems and aligning/reprojecting data where required using bilinear resampling.

## 8. Tiling
Generating non-overlapping 256×256 spatial tiles from larger scenes with a discard edge-policy for incomplete windows.

## 9. Z-score Processing
Normalizing features per band to stabilize downstream learning.

## 10. 3D-DWT
Applying 3D Discrete Wavelet Transform for multi-resolution feature extraction.

## 11. Albedo Filtering
Suppressing solar glare, cloud shadows, and atmospheric scattering while retaining useful albedo features.

## 12. Validation
Ensuring all outputs meet shape, dtype, and finiteness requirements.

## 13. Error Handling
Robust handling for missing API responses, corrupt files, and invalid data blocks.

## 14. Memory Considerations
Tile-based block processing and avoiding redundant array copies to adhere to the 16 GB system memory constraint.

## 15. Testing
Comprehensive unit and integration testing of the STAC interface, preprocessing steps, and interface contracts.

## 16. Spatial-Spectral Data Contract
This contract exists to enforce strict interface boundaries between modules, ensuring Module 2 receives data in a guaranteed format.

- **Expected Shape**: (256, 256, 13)
- **Axis Order**: (height, width, spectral_bands)
- **Dtype**: `numpy.float32`
- **Finite-Value Requirement**: All output values must be finite.

## 9. Z-Score Normalization
Independent statistical normalization ($Z = (X - \mu) / \sigma$) applied to each spectral channel within a 256×256 tile.
- **Precision**: Statistics ($\mu$, $\sigma$) calculated in `float64`, applied and downcast to strictly `float32`.
- **Validity**: Utilizes explicit masks or `isfinite` checks to exclude nodata. (A value of `0.0` is treated as a valid physical measurement).
- **Constant Bands**: Zero-variance channels gracefully map to `0.0` to preserve the finite output contract.
- **Flexibility**: Uncoupled from band counts, supporting both the intermediate 12-band real data and final 13-band tensors.

**Validation**: 
Explicit validation occurs before handoff to Module 2 to prevent propagation of malformed tensors. Module 1 must ultimately produce a valid `(256, 256, 13), float32` tensor. The mapping of the 13th channel is a project-level specification decision and must not be fabricated by the ingestion pipeline.
> - **L1C / MSI**: 13 spectral bands including B10.
> - **L2A**: 12 surface-reflectance spectral bands because B10 is excluded (no bottom-of-atmosphere information).
> 
> The ingestion pipeline explicitly maps the 12 verified L2A bands. The 13th channel mapping is pending project-level specification clarification.

Explicit validation occurs before handoff to Module 2 to prevent propagation of malformed tensors. Module 1 must ultimately produce a valid `(256, 256, 13), float32` tensor.

```text
Raw GeoTIFF / satellite data
        ↓
Spatial processing
        ↓
256 × 256 tile
        ↓
13 spectral bands
        ↓
Normalization
        ↓
NumPy float32 tensor
        ↓
Validation
        ↓
Module 2
```

## 17. Synthetic Dataset Generation
To allow for concurrent downstream development, a deterministic synthetic dataset generator creates mock data strictly adhering to the spatial-spectral contract.

Workflow:
```text
Synthetic Generator
        ↓
100 × (256,256,13)
        ↓
Data Contract Validation
        ↓
data/synthetic/
        ↓
Module 2 / Module 3 development
```

## 18. Module 2 Integration
Data Flow:
```text
Raw Satellite Data
        ↓
STAC
        ↓
Raster
        ↓
CRS
        ↓
256×256 Tiling
        ↓
Normalization
        ↓
3D-DWT
        ↓
Albedo Filter
        ↓
Validation
        ↓
(256,256,13) Float32
        ↓
Module 2
```
