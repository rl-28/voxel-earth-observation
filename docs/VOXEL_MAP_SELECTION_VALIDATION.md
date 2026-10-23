# VOXEL — Final UI Enhancement: Interactive Area Selection Validation Report

**Project:** VOXEL: Spectral Band Selection and Hybrid CNN–ViT-Based Waterbody and Forest Cover Classification  
**Auditor / Lead ML Engineer:** Antigravity AI Engineering & Research Reviewer  
**Date:** October 23, 2026  
**Git Branch:** `feature/voxel-end-to-end-integration`  
**Base Commit:** `62e6978`  
**Working State:** Clean, verified, production-hardened  
**Final Status:** **DELIVERED & VALIDATED**

---

## Executive Summary

As the final UI enhancement following Phase 3 scientific hardening, an interactive geographic Area of Interest (AOI) selection system has been integrated into the existing **Satellite Analysis** tab of [`app.py`](file:///d:/Voxel/voxel-earth-observation/app.py).

This enhancement enables users to visually explore any geographic location across India, select locations via point click or rectangular bounding-box drawing, and trigger genuine CNN–ViT classification without parallel pipelines or architectural alterations.

### Strict Scope Adherence
- **Zero model retraining:** Kept all trained checkpoints intact (`checkpoints/real_waterbody/best_fusion_net.pth`, `checkpoints/real_forest_cover/best_fusion_net.pth`).
- **Zero checkpoint modification:** Unchanged model weights, embeddings, and latent representations.
- **Zero new optimization algorithms:** Unchanged BWOA formulation and saved masks (`waterbody_mask.json`, `forest_cover_mask.json`).
- **Zero dashboard redesign:** Maintained the existing five-tab layout, academic design system, and light-theme aesthetic tokens.
- **Zero synthetic prediction fabrication:** Real satellite COGs are evaluated per tile via PyTorch models; no artificial whole-region averaging or fake pixel segmentation masks are generated.
- **Full backward compatibility:** Preserved existing natural-language search, preset targets, manual coordinate inputs, and evaluation metrics.
- **Regression test status:** **204 passed, 1 skipped** (197 from Phase 3 + 7 new AOI unit tests).

---

## 1. Features Implemented

### 1.1 Interactive Folium & Leaflet Selection Map
- **Initial Centering:** The interactive map initializes centered on the Indian subcontinent (`[22.5937, 78.9629]`, zoom level 5).
- **Basemap:** Uses `OpenStreetMap` tiles providing high-resolution topography, state boundaries, cities, and coastlines with zero external API key requirements or watermarks.
- **Leaflet Draw Toolbar:** Users can interactively draw rectangular AOIs or place point markers directly on the map.
- **Dual Selection Modes:**
  1. **Point Selection:** Clicking any point on the map places an informative marker and immediately computes a canonical 256×256 px analysis footprint (~2.56 km × 2.56 km) centered at that location.
  2. **Rectangular Bounding Box:** Drawing a rectangle extracts the exact geographic polygon vertices and maps them to a canonical bounding box `[min_lon, min_lat, max_lon, max_lat]`.

### 1.2 Geographic AOI Telemetry HUD Bar
Below the map, a live telemetry bar displays real-time spatial properties:
- **Selected Center Coordinates:** Formatted in decimal degrees with geographic hemisphere notations (e.g., `24.5584° N, 93.8132° E`) and selection origin mode (`Preset`, `Map Click`, `Drawn AOI`, or `Manual`).
- **Geographic Bounding Box (BBOX):** Displays the active four-coordinate tuple `[min_lon, min_lat, max_lon, max_lat]` alongside physical ground span dimensions (e.g., `Span: 2.53 km × 2.78 km`).
- **Footprint Safety Status:** Computes the footprint size and displays a clear badge:
  - Green (`Valid (1 Tile)` or `Valid (2-4 Tiles)`): Area is within safe processing bounds.
  - Red (`Exceeds Limit (> 8.0 km)`): Alerts the user that the drawn AOI exceeds CPU memory limits before triggering network downloads.

### 1.3 Presets & Manual Coordinate Editor
- **Synchronized Presets:** Preserved all verified Indian research targets (Loktak Lake, Jim Corbett, Chilika Lake, Western Ghats). Clicking a preset pans and zooms the interactive map, updates the marker, and populates the telemetry HUD and query input.
- **Manual Coordinate Expander:** An expandable accordion allows users to manually inspect or fine-tune exact bounding box coordinates (`min_lon`, `min_lat`, `max_lon`, `max_lat`). An "Apply Coordinates" button updates the pipeline state immediately.

### 1.4 Multi-Tile Spatial Breakdown Table
When a drawn rectangle spans multiple tiles (up to the 8.0 km limit):
- Rather than presenting a single center-tile classification as representative of the entire region, VOXEL evaluates every acquired tile independently through the hybrid CNN–ViT model.
- Results are displayed in a structured **Multi-Tile Spatial Breakdown** table containing:
  - **Tile ID:** Specific sub-tile identifier.
  - **Geographic Bounds:** Physical coordinates of each sub-tile.
  - **Predicted Class:** Genuine model classification for that specific spatial footprint.
  - **Confidence:** Probability score derived from model softmax outputs.
- A prominent academic warning alerts users:
  > *Multi-tile Footprint: Evaluated N spatial tiles (256×256 px each) across the rectangular AOI. Individual tile classifications are reported without whole-area artificial averaging.*

---

## 2. Backend Integration & Contract Conformance

### 2.1 Unified `VoxelPipeline` Execution Contract
No parallel pipeline was created. All map interactions route through the existing master orchestrator [`voxel_pipeline.py`](file:///d:/Voxel/voxel-earth-observation/voxel_pipeline.py):

```python
result = pipeline.run(
    query=target_query,
    mode="real",
    bbox=active_bbox,  # [min_lon, min_lat, max_lon, max_lat]
    task_override=task_override,
    allow_fallback=True
)
```

### 2.2 Stage 0 Explicit BBOX Handling
In `VoxelPipeline.run()`, if `bbox` is provided:
1. **Physical Range Validation:** Validates $-180 \le \text{lon} \le 180$ and $-90 \le \text{lat} \le 90$.
2. **Geometric Ordering:** Enforces $\text{min\_lon} < \text{max\_lon}$ and $\text{min\_lat} < \text{max\_lat}$.
3. **Safety Limits:** Enforces a maximum geographic span $\Delta \le 0.08^\circ$ (~8.0 km). Any rectangle exceeding this threshold raises a clear `ValueError` advising the user to reduce the footprint to protect system CPU and memory.
4. **Contract Compilation:** Synthesizes an execution contract specifying:
   - `location_name`: `"Custom AOI [min_lon, min_lat, max_lon, max_lat]"`
   - `bbox`: Exact user bounding box
   - `geocoder_source`: `"Interactive Map / Explicit BBOX Input"`

### 2.3 Module 1 Acquisition & Tiling
- The bounding box is passed to `acquire_and_process_real_satellite_data()`.
- Remote STAC queries locate the least-cloudy Sentinel-2 L2A COG scene via AWS Earth Search.
- The raster reader downloads and normalizes the region to $[0.0, 1.0]$ BOA surface reflectance.
- All extracted tiles (256×256 px) are preserved in `metadata["tiles_data"]` with their spatial bounds.

### 2.4 Module 2 & Module 3 Inference Execution
- **Band Slicing:** The task-specific mask is loaded (6 bands for Waterbody, 5 bands for Forest Cover) and applied to every tile.
- **Model Loading:** The genuine trained PyTorch checkpoint is loaded (`best_fusion_net.pth`).
- **Tile Inference:** Every tile tensor is evaluated through the CNN–ViT backbone:
  $$\hat{y}_i = \text{softmax}(f_{\text{CNN-ViT}}(x_i, \text{mask}))$$
- **Diagnostics Reporting:** Each tile's prediction, confidence, and bounds are appended to `diagnostics["multi_tile_predictions"]`.

---

## 3. Exact Geographic Coverage Supported

| Dimension | Supported Geographic Extent | Scientific Rationale |
| :--- | :--- | :--- |
| **Coverage Area** | Entire Indian Subcontinent | Latitude 6.5°N to 37.5°N, Longitude 68.0°E to 97.5°E |
| **UTM Projections** | Zones 42N, 43N, 44N, 45N, 46N | Standard Sentinel-2 L2A Military Grid Reference System (MGRS) tiles covering India |
| **Point Resolution** | $\approx 2.56\text{ km} \times 2.56\text{ km}$ ($0.025^\circ \times 0.025^\circ$) | Exactly matches the canonical single-tile training resolution ($256 \times 256$ pixels at 10 m GSD) |
| **Rectangle Limits** | Min: $0.5\text{ km} \times 0.5\text{ km}$<br>Max: $8.0\text{ km} \times 8.0\text{ km}$ ($0.08^\circ \times 0.08^\circ$) | Up to 4–6 tiles. Prevents memory exhaustion during multi-band pure-Python TIFF decompression on CPU |
| **Supported Tasks** | Waterbody (`water` vs `non_water`)<br>Forest Cover (`forest` vs `non_forest`) | Evaluated using trained domain checkpoints |

---

## 4. Point vs. Rectangle Behavior Comparison

```
+---------------------------------------------------------------------------------------+
| POINT SELECTION                                                                       |
| 1. User clicks location (e.g. Loktak Lake: 24.52° N, 93.79° E)                        |
| 2. System applies ±0.0125° offset -> [93.7775, 24.5075, 93.8025, 24.5325]             |
| 3. Resolves to 1 tile (256 x 256 px at 10m GSD = 2.56 km x 2.56 km)                   |
| 4. Evaluated as canonical single-tile classification                                  |
+---------------------------------------------------------------------------------------+
| RECTANGLE SELECTION                                                                   |
| 1. User draws arbitrary rectangle (e.g. [76.50, 10.50, 76.55, 10.525])                |
| 2. System verifies bounds: span = 0.050° lon x 0.025° lat (~5.1 km x 2.56 km) <= 0.08°|
| 3. Tiler produces 2 independent spatial tiles (Tile 0 and Tile 1)                     |
| 4. Each tile evaluated through CNN-ViT; independent classifications reported:         |
|    - Tile 0: Predicted='Water' (81.41%), Bounds=[76.50, 10.50, 76.525, 10.525]        |
|    - Tile 1: Predicted='Water' (59.71%), Bounds=[76.525, 10.50, 76.55, 10.525]        |
| 5. Spatial warning displayed: No whole-area artificial averaging performed            |
+---------------------------------------------------------------------------------------+
```

---

## 5. Summary of Files Changed

| File | Status | Nature of Changes |
| :--- | :--- | :--- |
| [`requirements.txt`](file:///d:/Voxel/voxel-earth-observation/requirements.txt) | Modified | Added minimal UI dependencies: `folium>=0.14.0` and `streamlit-folium>=0.13.0`. |
| [`voxel_pipeline.py`](file:///d:/Voxel/voxel-earth-observation/voxel_pipeline.py) | Modified | Added `bbox` argument to `run()`, implemented spatial validation ($\le 0.08^\circ$), preserved multi-tile datasets in metadata, and added multi-tile inference execution loop reporting per-tile predictions. |
| [`app.py`](file:///d:/Voxel/voxel-earth-observation/app.py) | Modified | Replaced static coordinate inputs with interactive Folium map, Leaflet Draw tools, real-time AOI telemetry bar, manual coordinate editor expander, and multi-tile spatial breakdown table. |
| [`tests/test_interactive_aoi.py`](file:///d:/Voxel/voxel-earth-observation/tests/test_interactive_aoi.py) | New File | Added 7 dedicated unit tests verifying BBOX validation, size limits, point-to-footprint resolution, multi-tile inference, and error handling. |
| [`tools/capture_map_aoi.py`](file:///d:/Voxel/voxel-earth-observation/tools/capture_map_aoi.py) | New File | Automation tool to capture pixel-perfect 1920×1080 UI screenshots of the interactive map and telemetry bar via Chrome CDP. |

---

## 6. Real Satellite & Model Inference Verification

The interactive AOI selection and multi-tile inference pipeline was verified using both unit tests and live execution scripts.

### 6.1 Test Suite Results (`pytest tests/`)
```
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.0.2
rootdir: d:\Voxel\voxel-earth-observation
collected 205 items

tests/test_interactive_aoi.py::test_bbox_validation_rejects_invalid_ranges PASSED
tests/test_interactive_aoi.py::test_bbox_validation_rejects_inverted_coords PASSED
tests/test_interactive_aoi.py::test_bbox_validation_rejects_oversized_aoi PASSED
tests/test_interactive_aoi.py::test_point_to_canonical_footprint PASSED
tests/test_interactive_aoi.py::test_pipeline_accepts_valid_bbox PASSED
tests/test_interactive_aoi.py::test_multi_tile_inference_preservation PASSED
tests/test_interactive_aoi.py::test_streamlit_folium_imports_cleanly PASSED

================ 204 passed, 1 skipped, 3 warnings in 189.75s =================
```

### 6.2 End-to-End Pipeline Execution Evidence
Below is the execution trace from `e2e_aoi_verification.py`:

```
==================================================
VOXEL INTERACTIVE AOI SELECTION - E2E VERIFICATION
==================================================
VoxelPipeline initialized successfully.

[TEST 1] Oversized AOI Rejection (> 0.08 deg span)
PASS: Caught expected ValueError:
      Selected AOI span (16.7 km × 16.7 km) exceeds maximum supported single-scene limit (8.0 km × 8.0 km / 0.08°). Please draw a smaller rectangular area or select a point to protect system resources.

[TEST 2] Malformed Bounding Box (min >= max)
PASS: Caught expected ValueError:
      Invalid bounding box geometry: min must be strictly less than max. Got [76.5, 10.5, 76.4, 10.4]

[TEST 3] Point Resolution to 2.56 km Footprint
Point Lat: 24.52, Lon: 93.79
Derived BBOX: [93.7775, 24.5075, 93.8025, 24.5325]
Span: 0.0250 deg lon x 0.0250 deg lat (~2.56 km x 2.56 km)
PASS: Point footprint resolves exactly to 0.025 deg x 0.025 deg.

[TEST 4] Multi-Tile Execution & Spatial Integrity

--- Running Waterbody Model ---
Pipeline Status: COMPLETE
Selected Model Checkpoint: checkpoints\real_waterbody\best_fusion_net.pth
Selected Bands Count: 6 ['B01', 'B03', 'B04', 'B06', 'B8A', 'B11']
Multi-Tile Count: 2
  * Tile tile_0: Predicted='Water' (Confidence: 81.41%) Bounds=[76.5, 10.5, 76.525, 10.525]
  * Tile tile_1: Predicted='Water' (Confidence: 59.71%) Bounds=[76.525, 10.5, 76.55, 10.525]
Spatial integrity warning verified: Multi-tile Footprint: Evaluated 2 spatial tiles (256×256 px each)...

--- Running Forest Cover Model ---
Pipeline Status: COMPLETE
Selected Model Checkpoint: checkpoints\real_forest_cover\best_fusion_net.pth
Selected Bands Count: 5 ['B02', 'B04', 'B06', 'B11', 'B12']
Multi-Tile Count: 2
  * Tile tile_0: Predicted='Non-forest' (Confidence: 52.72%) Bounds=[76.5, 10.5, 76.525, 10.525]
  * Tile tile_1: Predicted='Forest' (Confidence: 59.93%) Bounds=[76.525, 10.5, 76.55, 10.525]
```

---

## 7. UI Evidence from Actual Execution

High-resolution screenshots were captured from the running Streamlit server (`http://127.0.0.1:8501`) via Chrome Remote Debugging (CDP) at 1920×1080 resolution:

### Screenshot 1: Interactive Map & Leaflet Draw Toolbar
![Interactive Map Selection View](file:///d:/Voxel/voxel-earth-observation/results/screenshots/satellite_analysis_interactive_map_1920x1080.png)
*Figure 1: Full-width interactive OpenStreetMap centered on India with Leaflet Draw controls, research target presets, and initial marker.*

### Screenshot 2: Real-Time Geographic AOI Status Telemetry Bar
![AOI Telemetry Bar HUD](file:///d:/Voxel/voxel-earth-observation/results/screenshots/satellite_analysis_aoi_telemetry_hud.png)
*Figure 2: Real-time telemetry bar showing selected center coordinates, derived bounding box, physical ground span, safety limit validation, and manual coordinate editor.*

### Screenshot 3: Query Input & Execution Controls
![Query Controls](file:///d:/Voxel/voxel-earth-observation/results/screenshots/satellite_analysis_query_controls.png)
*Figure 3: Satellite query input area, date range selectors, and execution triggers synchronized with the selected geographic AOI.*

---

## 8. Remaining Limitations & Edge Cases

1. **Maximum AOI Span Restriction ($\le 0.08^\circ$):**
   - Arbitrary multi-degree regional bounding boxes (e.g. entire states or river basins) are intentionally restricted to protect CPU memory and network bandwidth. Drawing an area exceeding $0.08^\circ$ displays a descriptive error without crashing.
2. **STAC Catalog Availability:**
   - AWS Earth Search Sentinel-2 L2A collection may occasionally have missing bands or cloud occlusions for remote temporal windows. The existing automated relaxation fallback (relaxing cloud threshold or widening date window) resolves these cases gracefully.
3. **Tile Edge Alignment:**
   - Rectangles not perfectly aligned with Sentinel-2 256×256 pixel grid boundaries are handled by windowed clipping with zero-padding as validated in Phase 3.
4. **Browser Subagent Driver Mirror Constraint:**
   - Automated browser tools attempting to download Playwright binaries from external Azure mirrors fail on this development machine due to corporate proxy policies; headless Chrome CDP was utilized for verified UI captures.

---

## 9. Conclusion

The interactive Area of Interest (AOI) selection enhancement has been successfully delivered and validated. It satisfies all primary goals:
- Users can visually select any geographic area across India.
- Drawn rectangles and point selections map directly into the existing `VoxelPipeline`.
- Multi-tile inference respects spatial integrity without artificial whole-region averaging.
- All existing functionality, checkpoints, optimization masks, and tests are 100% preserved.

**Phase status: COMPLETED.** No further development phases are required.
