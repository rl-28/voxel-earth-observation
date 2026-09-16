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

## 10. 3D Discrete Wavelet Transform (3D-DWT)
Implements true 3D spatial-spectral wavelet decomposition via PyWavelets to compress spectral bands and capture edge/textural information without discarding spatial scale.
- **Wavelet Used**: `db1` (Haar) as a lightweight, deterministic baseline.
- **Decomposition**: `level=1` 3D DWT across Height, Width, and Spectral axes (`H x W x C`), producing approximation and detail coefficients.
- **Boundary Mode**: `symmetric` extension handles image borders smoothly.
- **Reconstruction & Spatial Preservation**: The API exposes `.reconstruct_approximation()` and `.reconstruct_full()` which invert the transform to restore the original `(256, 256, C)` spatial dimensions. Discarded frequencies are correctly zeroed prior to inversion.
- **Compression vs. Transform**: Currently implemented as a lossless, mathematically exact transform. The "compression" occurs downstream if subbands are discarded.
- **Agnostic Representation**: Generically processes both 12-band and 13-band datasets.
- **Validation**: Strict checks block NaN/Inf inputs and verify near-zero reconstruction error.

## 11. Albedo / Reflection Contamination Filtering
Provides physically motivated reflection and contamination suppression for surface-reflectance tensors. 
> **Important Scientific Scope**: Sentinel-2 Level-2A imagery is already atmospherically corrected by Copernicus. This module identifies residual anomalies; it is not a replacement for a full atmospheric radiative-transfer correction system, nor does it calculate absolute physical albedo.

- **Glare / Reflection Anomaly Mask**: Detects unusually high broadband reflectance by requiring simultaneous elevated responses in visible (B03), NIR (B08), and SWIR (B11) channels.
- **Atmospheric Anomaly Mask**: Detects residual atmospheric scattering/haze via isolated extreme responses in coastal aerosol (B01) and blue (B02) bands.
- **Cloud-Shadow Anomaly Mask**: Detects shadow-like contamination via abnormally low (but non-zero) visible (B04) and NIR (B08) responses.
- **Masking Strategy**: Combines individual anomaly masks into a master `contamination_mask`. Contaminated pixels are replaced with a predefined `FILTER_FILL_VALUE`, preserving the original spatial scale `(256, 256, C)`.
- **Valid-Mask Handling**: Pre-existing invalid pixels are safely excluded from thresholding and statistics to prevent false positives.
- **Thresholds**: Applies dynamically selected configurable thresholds based on whether the input is raw reflectance or Z-score normalized data.

**Validation**: 
Explicit validation occurs before handoff to Module 2 to prevent propagation of malformed tensors. Module 1 must ultimately produce a valid `(256, 256, 13), float32` tensor. The mapping of the 13th channel is a project-level specification decision and must not be fabricated by the ingestion pipeline.

## 12. End-to-End Pipeline Integration (Phase 8)
An orchestration layer (`module1_ingestion/pipeline.py`) unifies the ingestion, tiling, normalization, wavelet, and filtering modules into a single, cohesive processing sequence.
- **Pipeline Flow**: Ingestion -> CRS/Resolution Alignment -> 256x256 Tiling -> Valid Mask Extraction -> Z-Score Normalization -> 3D-DWT -> Albedo/Reflection Filtering -> Final Validation.
- **12/13-Band Interoperability**: Adapts fluidly to both 12-band real intermediate datasets and the 13-channel project specification, routing to appropriate validators.
- **Memory Management**: Yields processed tiles sequentially via a generator rather than materializing full processed scenes in RAM, strictly adhering to the 16 GB hardware guardrail.
- **Diagnostics & Result Structure**: Returns a `Module1Result` containing the filtered spatial tensor alongside comprehensive diagnostic timing, contamination statistics, masks, and metadata.
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
