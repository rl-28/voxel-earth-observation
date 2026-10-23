# VOXEL — Final Map Footprint and Spectral Output Verification Report

**Document Reference:** `docs/VOXEL_FOOTPRINT_INTEGRITY.md`  
**Date:** October 2026  
**System:** VOXEL Earth Observation Platform (Task-Adaptive Neural & Spectral Intelligence)  
**Status:** FULLY VERIFIED & COMPLIANT  

---

## 1. Executive Summary & Root Cause Analysis

### 1.1 The Observed Anomaly
During satellite analysis verification on the Ratnagiri AOI, the following anomaly was identified:
- **Selected Center:** `17.2615° N, 73.3715° E` (Ratnagiri coastal region, Maharashtra, India)
- **Selected BBOX:** `[73.3590, 17.2490, 73.3840, 17.2740]` ($\Delta\text{Lon} = 0.0250^\circ, \Delta\text{Lat} = 0.0250^\circ$)
- **Dashboard Reported Span:** `2.66 km × 2.78 km`
- **Footprint Safety Reported:** `Valid (1 Tile · 2.56 km)`
- **Standard Model Tile Size:** `256 × 256 pixels` at `10 m` Ground Sampling Distance (GSD) = `2560 m × 2560 m` (`2.56 km × 2.56 km`)

### 1.2 Geodesic Analysis & Physical Discrepancy
At latitude $\phi = 17.2615^\circ \text{ N}$ on the WGS84 reference ellipsoid:
- $1^\circ \text{ Latitude} \approx 110.66\text{ km} \implies \Delta\text{Lat} = 0.0250^\circ \approx 2.7665\text{ km} \approx 2.77\text{ km}$
- $1^\circ \text{ Longitude} \approx 111.32 \times \cos(17.2615^\circ) \approx 106.31\text{ km} \implies \Delta\text{Lon} = 0.0250^\circ \approx 2.6577\text{ km} \approx 2.66\text{ km}$

Because both dimensions ($2.66\text{ km}$ and $2.78\text{ km}$) strictly exceed the $2.56\text{ km}$ extent of a standard single tile, **a single 256×256 px tile physically cannot cover this selected bounding box**. To cover the requested area without omitting border territory, a $2 \times 2$ grid (**4 tiles**, covering $5.12\text{ km} \times 5.12\text{ km}$) is mathematically required.

### 1.3 Identified Root Causes
1. **Hardcoded UI HUD Status:** In `app.py`, the Footprint Safety HUD previously evaluated `if sel_type in ["marker_point", "map_click", "preset"]` and unconditionally hardcoded the string `"Valid (1 Tile · 2.56 km)"`, completely ignoring whether the computed span was actually $\le 2.56\text{ km}$.
2. **Symmetric Degree Box vs. Metric Footprint:** Point clicks and preset coordinates previously applied a fixed $\pm 0.0125^\circ$ angular offset. Due to ellipsoidal Earth geometry, $\pm 0.0125^\circ$ creates a $2.66\text{ km} \times 2.78\text{ km}$ rectangle, causing single point clicks to inadvertently create multi-tile bounding boxes.
3. **Top-Left Tile Truncation in Pipeline:** In `voxel_pipeline.py`, when a rectangular AOI was passed, Module 1 sliced only the top-left $256 \times 256\text{ px}$ tile (`tile_000_000`) and discarded the remaining portions of the requested rectangle, while the UI misleadingly implied the entire bounding box was classified.
4. **Boundary Floating-Point Precision Bloat:** Exact single tiles (e.g., $2560.0001\text{ m}$) divided by $2560.0\text{ m}$ produced $1.00000004$, causing `math.ceil()` to jump from 1 to 2 tiles without an engineering epsilon threshold ($\pm 50\text{ m}$).

---

## 2. Minimal Surgical Corrections Implemented

### 2.1 Stage 0 Geodesy & Canonical Tile Computation (`utils/llm_parser.py`)
- **`calculate_geographic_span(bbox)`**: Implemented WGS84 ellipsoidal distance calculations:
  $$\Delta y = (lat_{\max} - lat_{\min}) \times 111.32\text{ km}$$
  $$\Delta x = (lon_{\max} - lon_{\min}) \times 111.32 \times \cos(lat_{\text{mid}})\text{ km}$$
- **`compute_canonical_tile_bbox(lat, lon, tile_size_m=2560.0)`**: Added canonical helper that generates the exact decimal bounding box matching a $2560\text{ m} \times 2560\text{ m}$ single tile at any latitude in India ($8^\circ\text{ N}$ to $36^\circ\text{ N}$):
  $$\Delta\phi = \frac{2560.0}{111320.0} \approx 0.0230^\circ$$
  $$\Delta\lambda = \frac{2560.0}{111320.0 \times \cos(\phi_{\text{rad}})}$$
- Standardized point selection and preset clicks to use `compute_canonical_tile_bbox` so that 1-tile point selections remain precisely $2.56\text{ km} \times 2.56\text{ km}$.

### 2.2 Symmetrical Multi-Tile Ingestion (`module1_ingestion/stac_downloader.py`)
- Updated `download_bands()`:
  - Dynamically calculates required tile counts in projected coordinates:
    $$\text{tiles}_x = \max\left(1, \left\lceil \frac{gw - 50.0}{2560.0} \right\rceil\right), \quad \text{tiles}_y = \max\left(1, \left\lceil \frac{gh - 50.0}{2560.0} \right\rceil\right)$$
  - Reads COG windows sized to exact integer multiples of 256 px: $\text{width} = \text{tiles}_x \times 256$, $\text{height} = \text{tiles}_y \times 256$.
  - Centered symmetrically on $(cx, cy)$ of the requested AOI, guaranteeing **100% of the requested rectangle is covered** without cropping borders.
  - Generates seamless $256 \times 256\text{ px}$ sub-tile windows via `generate_tile_windows()`.

### 2.3 Pipeline Orchestration & Serialization Integrity (`voxel_pipeline.py`)
- **Explicit BBOX Prioritization:** Bounding box validation is executed at the very entrance of `VoxelPipeline.run()`, ensuring Stage 0 contract and downstream stages preserve the exact user-selected BBOX without silent modification.
- **Cache Isolation:** Added `bbox_slug` (`f"{selected_item.id}_{bbox_slug}"`) to STAC local cache directories, guaranteeing scenes for different geographic coordinates never cross-contaminate.
- **Multi-Tile Inference Execution:** When $\text{total\_tiles} > 1$, the pipeline executes neural inference on **every individual tile** across the grid, returning `diagnostics["multi_tile_predictions"]` with independent classification labels, confidence percentages, and WGS84 geographic bounding boxes.
- **Footprint Diagnostics:** Recorded `diagnostics["actual_processed_bbox"]` (the union bounding box of all processed tiles) and `diagnostics["actual_processed_tile_count"]`.
- **JSON Serialization Fix:** Replaced shallow dict filtering in `VoxelPipelineResult.to_dict()` with a recursive sanitizer `_sanitize()` that cleanly converts NumPy scalars and strips heavy arrays (`processed_tensor`), preventing `TypeError: Object of type ndarray is not JSON serializable`.

### 2.4 User Interface Transparency & Invalidation (`app.py`)
- **Dynamic HUD Tile Calculation:**
  $$\text{est}_w = \max\left(1, \left\lceil \frac{b_w - 0.05}{2.56} \right\rceil\right), \quad \text{est}_h = \max\left(1, \left\lceil \frac{b_h - 0.05}{2.56} \right\rceil\right), \quad \text{est}_{\text{tiles}} = \text{est}_w \times \text{est}_h$$
  - Displays `Valid (1 Tile · 2.56 km)` only when $\text{est}_{\text{tiles}} == 1$.
  - Displays `Valid ({est_tiles} Tiles · {est_w}×{est_h} grid)` for larger rectangles (e.g., `Valid (4 Tiles · 2×2 grid)` for the Ratnagiri anomaly).
- **Dual Map Layer Rendering:**
  - **Selected AOI (Blue Rectangle):** Displays the exact bounding box requested by the user.
  - **Processed Model Footprints (Green Rectangles):** Displays each $256 \times 256\text{ px}$ tile footprint actually evaluated by the CNN–ViT model, ensuring the user visually sees the physical spatial coverage.
- **Strict UI State Invalidation:**
  - Instantly clears `latest_pipeline_result = None` whenever the user clicks a preset, clicks the map, draws a rectangle, applies manual coordinates, or changes the task mode.
  - Added a staleness guard that checks if the active result's BBOX differs by $> 0.0005^\circ$ from `selected_bbox`, preventing stale inference results from persisting across AOI modifications.
- **Provenance & Metadata Audit Panel:** Displays the selected AOI, actual processed bounding box, coverage status, processed tile count, STAC Scene ID, and acquisition date.
- **Spectral Index Clarification:** Explicitly annotated NDVI captions to clarify that NDVI is a deterministic physical reflectance ratio, not a CNN–ViT model output.

---

## 3. Real Classification & Model Provenance

Every processed tile strictly routes through the validated VOXEL architecture:

| Property | Waterbody Task | Forest Cover Task |
| :--- | :--- | :--- |
| **Active Checkpoint** | `checkpoints/real_waterbody/best_fusion_net.pth` | `checkpoints/real_forest_cover/best_fusion_net.pth` |
| **Trained Task Mask** | `checkpoints/masks/waterbody_mask.json` | `checkpoints/masks/forest_cover_mask.json` |
| **Mask Band Indices** | `[0, 2, 3, 5, 8, 11]` (6 bands) | `[1, 3, 5, 11, 12]` (5 bands) |
| **Selected Sentinel-2 Bands**| `B01, B03, B04, B06, B8A, B11` | `B02, B04, B06, B11, B12` |
| **Output Classes** | `["Non-water", "Water"]` | `["Non-forest", "Forest"]` |
| **Data Source** | Real Sentinel-2 L2A STAC Acquisition | Real Sentinel-2 L2A STAC Acquisition |
| **Segmentation Mask** | **None fabricated** (Tile-level categorical classification) | **None fabricated** (Tile-level categorical classification) |

---

## 4. Radiometric Validation: RGB and NDVI

### 4.1 Surface Reflectance Sourcing
- Both RGB and NDVI are computed from the **exact same 13-band surface reflectance tensor** (`calibrated_tensor`, shape `(13, 256, 256)`) representing the identical spatial footprint.
- **RGB Preview Composite:**
  - Red Channel = Sentinel-2 Band 4 (Index 3, $665\text{ nm}$)
  - Green Channel = Sentinel-2 Band 3 (Index 2, $560\text{ nm}$)
  - Blue Channel = Sentinel-2 Band 2 (Index 1, $490\text{ nm}$)
  - Robust 2%–98% percentile radiometric contrast stretching scaled to $[0.0, 1.0]$.
- **Normalized Difference Vegetation Index (NDVI):**
  $$\text{NDVI} = \frac{\text{B08 (NIR)} - \text{B04 (Red)}}{\text{B08 (NIR)} + \text{B04 (Red)} + 10^{-8}}$$
  - Sourced from Band 8 (Index 7, $842\text{ nm}$) and Band 4 (Index 3, $665\text{ nm}$).

### 4.2 Radiometric Mathematics & Alignment Verification
In `tests/test_footprint_integrity.py::test_case_6_ndvi_rgb_radiometric_and_spatial_alignment`:
1. **Mathematical Consistency:** The computed NDVI array was compared against an independently derived NumPy reference matrix:
   $$\max |\text{NDVI}_{\text{calc}} - \text{NDVI}_{\text{ref}}| = 0.0 < 10^{-7}$$
2. **Physical Bounds:** Confirmed $-1.0 \le \text{NDVI} \le +1.0$ across all 65,536 pixels.
3. **Spatial Dimensions & Orientation:** Confirmed RGB and NDVI arrays have identical $(256, 256)$ spatial dimensions, identical 10m GSD, and matching coordinate orientation without transposition or flipping.
4. **Architectural Role:** NDVI is presented strictly as a radiometric physical index, not as a CNN–ViT segmentation mask.

---

## 5. Verification Test Suite: 6 Key Test Cases

All 6 required test scenarios have been implemented in `tests/test_footprint_integrity.py` and pass with 100% compliance:

```
tests/test_footprint_integrity.py::test_case_1_aoi_smaller_than_single_tile PASSED          [ 16%]
tests/test_footprint_integrity.py::test_case_2_aoi_extending_beyond_single_tile PASSED       [ 33%]
tests/test_footprint_integrity.py::test_case_3_rectangle_requiring_multiple_tiles_anomaly_resolution PASSED [ 50%]
tests/test_footprint_integrity.py::test_case_4_custom_indian_location_isolation PASSED     [ 66%]
tests/test_footprint_integrity.py::test_case_5_waterbody_and_forest_cover_on_same_location PASSED [ 83%]
tests/test_footprint_integrity.py::test_case_6_ndvi_rgb_radiometric_and_spatial_alignment PASSED [100%]
```

### Detailed Verification Evidence

#### Test Case 1: AOI Smaller Than One Tile
- **Input:** $1.0\text{ km} \times 1.0\text{ km}$ sub-tile bounding box centered at $17.2615^\circ\text{ N}, 73.3715^\circ\text{ E}$.
- **Result:** Span ($0.94\text{ km} \times 0.98\text{ km}$) safely clamped to $\text{tiles}_x = 1, \text{tiles}_y = 1$. The standard $256 \times 256\text{ px}$ tile footprint completely covers the requested sub-tile AOI.

#### Test Case 2: AOI Extending Beyond a Single Tile
- **Input:** Horizontally elongated AOI $[73.3500, 17.2550, 73.3830, 17.2710]$ ($3.51\text{ km} \times 1.77\text{ km}$).
- **Result:** Width exceeds $2.56\text{ km}$; height $\le 2.56\text{ km}$. Pipeline dynamically generated $\text{tiles}_x = 2, \text{tiles}_y = 1$ (2 tiles total). Both tiles (`tile_000_000` and `tile_000_001`) processed independently, covering the full requested rectangle.

#### Test Case 3: Rectangle Requiring Multiple Tiles (Ratnagiri Anomaly)
- **Input:** BBOX $[73.3590, 17.2490, 73.3840, 17.2740]$ (Span $2.66\text{ km} \times 2.78\text{ km}$).
- **Result:**
  - Evaluated $\text{tiles}_x = 2, \text{tiles}_y = 2$ (4 tiles total).
  - HUD correctly reports `Valid (4 Tiles · 2×2 grid)` rather than the false `Valid (1 Tile · 2.56 km)`.
  - Pipeline executes all 4 tiles (`tile_000_000`, `tile_000_001`, `tile_001_000`, `tile_001_001`) and records their individual bounds.
  - Selected BBOX is fully preserved without silent shrinking.

#### Test Case 4: Custom Indian Location (Sardar Sarovar Dam, Gujarat)
- **Input:** $[73.7400, 21.8200, 73.7650, 21.8450]$ at $21.83^\circ\text{ N}$.
- **Result:** Longitude scaling at higher latitude ($103.3\text{ km/deg}$) accurately computed. Unique cache directory slugging prevents collision with other locations.

#### Test Case 5: Waterbody and Forest Cover Selection on Same Location
- **Input:** Same Ratnagiri coordinates evaluated under both tasks.
- **Result:** Waterbody task strictly activates 6-band mask `[0, 2, 3, 5, 8, 11]` and outputs `["Non-water", "Water"]`. Forest cover task strictly activates 5-band mask `[1, 3, 5, 11, 12]` and outputs `["Non-forest", "Forest"]`. No shared state corruption.

#### Test Case 6: NDVI Alignment Against Corresponding RGB Imagery
- **Result:** RGB composite (B04, B03, B02) and NDVI (B08, B04) share identical $(256, 256)$ pixel grid. Zero discrepancy against independent mathematical reference ($< 10^{-7}$). Categorical model predictions distinguished from spectral index maps.

---

## 6. Regression Testing Summary

| Test Suite | Tests Run | Passed | Failed | Skipped | Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Existing Repository Test Suite** | 205 | 204 | 0 | 1 | 273.06s |
| **Interactive AOI Test Suite (`test_interactive_aoi.py`)** | 7 | 7 | 0 | 0 | 38.20s |
| **Footprint Integrity Test Suite (`test_footprint_integrity.py`)** | 6 | 6 | 0 | 0 | 47.95s |
| **Total Test Suite** | **218** | **217** | **0** | **1** | — |

*Zero regressions introduced. All constraints honored: no dashboard redesign, no model retraining, no alteration of BWOA optimization, and no modification of saved benchmark metrics.*
