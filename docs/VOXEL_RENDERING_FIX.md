# VOXEL — STREAMLIT RENDERING BUG FIX REPORT

**Document ID:** `VOXEL-FIX-RENDERING-001`  
**Date:** 2026-10-23  
**Status:** VERIFIED & RESOLVED  
**Branch:** `feature/voxel-end-to-end-integration`  

---

## 1. Confirmed Root Cause

During live Sentinel-2 satellite inference on the query *"Track waterbody levels in Loktak Lake, Manipur"* (task mode: `Waterbody Classification (6 Bands)`), the backend completed execution successfully in ~25.6s, returning:
- **Predicted Class:** `Water`
- **Softmax Confidence:** `96.77%`
- **Active Bands:** 6 (`['B01', 'B03', 'B04', 'B06', 'B8A', 'B11']`)
- **Data Source:** Sentinel-2 L2A (`S2A_46REN_20241228_0_L2A`)
- **Model Checkpoint:** `checkpoints/real_waterbody/best_fusion_net.pth`

However, immediately upon entering `render_dashboard()` in [app.py](file:///d:/Voxel/voxel-earth-observation/app.py#L939), the Streamlit UI crashed with an uncaught runtime exception:

```text
2026-10-23 13:17:53.330 Uncaught app execution
Traceback (most recent call last):
  File "D:\Voxel\voxel-earth-observation\.venv\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 924, in code_to_exec
    exec(code, module.__dict__)
  File "D:\Voxel\voxel-earth-observation\app.py", line 1601, in <module>
    render_dashboard()
  File "D:\Voxel\voxel-earth-observation\app.py", line 939, in render_dashboard
    exec_ckpt_path = res.checkpoint_path or f"checkpoints/real_{res.task_type}/best_fusion_net.pth"
                     ^^^^^^^^^^^^^^^^^^^
AttributeError: 'VoxelPipelineResult' object has no attribute 'checkpoint_path'
```

### Analysis of the Result Contract
Inspection of [VoxelPipelineResult](file:///d:/Voxel/voxel-earth-observation/voxel_pipeline.py#L89-L125) revealed:
1. The dataclass specifies `model_checkpoint: Optional[str] = None` (Line 114) and `model_provenance: str = "unavailable"` (Line 115).
2. During inference in `voxel_pipeline.py` (Line 1337), the active checkpoint path is correctly assigned to `model_checkpoint=active_ckpt_path`.
3. `checkpoint_path` was never defined on `VoxelPipelineResult`.
4. Line 939 in `app.py` erroneously accessed `res.checkpoint_path`, aborting dashboard rendering before the Provenance Card, Interactive Map, True-Color RGB, NDWI Spectral Index, and Reflectance plots could render.

---

## 2. Exact Files and Lines Corrected

### 1. `app.py` — Line 939
**Replaced:**
```python
exec_ckpt_path = res.checkpoint_path or f"checkpoints/real_{res.task_type}/best_fusion_net.pth"
exec_ckpt_name = os.path.basename(exec_ckpt_path)
```
**With Verified Attribute & Safe Guard:**
```python
exec_ckpt_path = res.model_checkpoint or f"checkpoints/real_{res.task_type}/best_fusion_net.pth"
exec_ckpt_name = os.path.basename(exec_ckpt_path) if exec_ckpt_path else "None"
```

### 2. `voxel_pipeline.py` — Line 125 (`VoxelPipelineResult`)
Added a backward-compatible class property to prevent any future crashes from legacy tools or external scripts attempting to access `checkpoint_path`:
```python
    @property
    def checkpoint_path(self) -> Optional[str]:
        """Backward-compatible alias for model_checkpoint."""
        return self.model_checkpoint
```

### 3. AST Attribute Access Audit across `app.py`
An exhaustive Abstract Syntax Tree (AST) scan of all attribute accesses on `res` in [app.py](file:///d:/Voxel/voxel-earth-observation/app.py) was performed:
| Attribute | Location in `app.py` | Status on `VoxelPipelineResult` |
| :--- | :--- | :--- |
| `res.task_type` | `render_dashboard()` | **Verified Valid** (Field) |
| `res.resolved_location` | `render_dashboard()` | **Verified Valid** (Field) |
| `res.diagnostics` | `render_dashboard()` | **Verified Valid** (Field) |
| `res.prediction` | `render_dashboard()` | **Verified Valid** (Field) |
| `res.selected_bands` | `render_dashboard()` | **Verified Valid** (Field) |
| `res.candidate_bwoa_bands` | `render_dashboard()` | **Verified Valid** (Field) |
| `res.mask_provenance` | `render_dashboard()` | **Verified Valid** (Field) |
| `res.mask_compatibility_note` | `render_dashboard()` | **Verified Valid** (Field) |
| `res.band_reduction_pct` | `render_dashboard()` | **Verified Valid** (Field) |
| `res.timing_breakdown_ms` | `render_dashboard()` | **Verified Valid** (Field) |
| `res.data_source` | `render_dashboard()` | **Verified Valid** (Field) |
| `res.is_synthetic` | `render_dashboard()` | **Verified Valid** (Field) |
| `res.input_tensor_shape` | `render_dashboard()` | **Verified Valid** (Field) |
| `res.explainability` | `render_dashboard()` | **Verified Valid** (Field) |
| `res.similar_terrain` | `render_dashboard()` | **Verified Valid** (Field) |
| `res.to_dict` | `render_dashboard()` | **Verified Valid** (Method) |
| `res.model_checkpoint` | `render_dashboard()` (L939) | **Fixed & Verified** (Field) |
| `res.checkpoint_path` | `render_dashboard()` (L939) | **Fixed & Verified** (Property alias) |
| `res.tile_id`, `res.output_shape` | `run_module1_ingestion()` (L89, 93) | **Valid** (`res` is `Module1Result`) |

---

## 3. Verified Result Rendering & Visual Evidence

### Failing Workflow Test
- **Query:** *"Track waterbody levels in Loktak Lake, Manipur"*
- **Task Mode:** `Waterbody Classification (6 Bands)`
- **Execution Mode:** Real Satellite (STAC acquisition / cached tile `S2A_46REN_20241228_0_L2A`)
- **Model Checkpoint:** `checkpoints/real_waterbody/best_fusion_net.pth` (genuine trained weights)
- **Model Prediction:** `Water` (`96.77%` confidence)
- **Scientific Qualification:** Displayed research note confirming VOXEL operates on 256×256 px tile classification and does not quantify volumetric/depth changes.

### Complete Rendering Checklist
With the fix in place, all sections render without exception:
1. **Prediction Banner:** Displays `Task: Waterbody · Spectral Volume Reduction: 53.8%`, `Verified Sentinel-2 Model`, `Water`, and `96.77% Softmax Probability Density`.
2. **Key Telemetry Metrics:**
   - Active Bands: `6 / 13` (`-53.8% Volume`)
   - Total Latency: `24.5s` (`Measured Pipeline`)
   - Data Source: `Live Sentinel-2` (`Loktak Lake, Manipur`)
   - Surface Radiometry: `BOA [0.0, 1.0]` (`L2A Surface Reflectance`)
3. **Real Satellite Provenance & Geographic Locator:**
   - Selected AOI (BBOX): `[93.8007, 24.5459, 93.8257, 24.5709]`
   - Actual Processed BBOX: `[93.8007, 24.5459, 93.8257, 24.5709]`
   - Scene ID: `S2A_46REN_20241228_0_L2A`
   - Acquisition Date: `2024-12-28`
   - Coverage Status: `Full AOI Coverage (1 Tile)` / `Canonical 1-Tile Coverage (2.56 km)`
   - Executed Checkpoint: `best_fusion_net.pth` (from verified `checkpoints/real_waterbody/best_fusion_net.pth`)
   - Interactive Folium Locator Map: Rendered centered on Loktak Lake.
4. **Actual Satellite RGB & Real Spectral Index (NDWI):**
   - True-Color Composite (B04-Red, B03-Green, B02-Blue).
   - Real Spectral Index: `NDWI = (B03 - B08) / (B03 + B08)`.
5. **Measured Spectral Reflectance Signature & BWOA Band Selection Matrix:**
   - Reflectance curve across 12 valid L2A wavelengths with ±1σ spatial variance.
   - Cirrus Band 10 safely excluded.
   - Active task bands highlighted (`B01`, `B03`, `B04`, `B06`, `B8A`, `B11`).
6. **Class Probability Distribution:**
   - `Water`: `96.8%`
   - `Non-water`: `3.2%`
7. **Expanders & Ground Truth Notices:**
   - Detailed pipeline telemetry.
   - JSON Result Contract export.
   - Scientific Ground Truth and Scope Qualifications notice.

### Forest-Cover Workflow Test
- **Query:** *"Classify forest canopy cover in Jim Corbett, Uttarakhand"*
- **Task Mode:** `Forest Cover Classification (5 Bands)`
- **Model Checkpoint:** `checkpoints/real_forest_cover/best_fusion_net.pth`
- **Result:** `Forest` (`54.88%`), `NDVI` index, and complete UI rendering verified without exception.

---

## 4. Automated Regression Tests

A dedicated test suite was added in [tests/test_ui_rendering_contract.py](file:///d:/Voxel/voxel-earth-observation/tests/test_ui_rendering_contract.py):

1. `test_result_contract_contains_all_ui_rendered_attributes`: AST parsing of `app.py` verifying every attribute accessed on `res` exists on `VoxelPipelineResult`.
2. `test_checkpoint_path_property_backward_compatibility`: Validates that `res.checkpoint_path` mirrors `res.model_checkpoint`.
3. `test_ui_rendering_logic_single_and_multi_tile`: Executes all telemetry, label formatting, and tile breakdown logic on single and multi-tile results.
4. `test_streamlit_apptest_waterbody_and_forest_cover_rendering`: Uses Streamlit's official `AppTest` framework to simulate full dashboard execution on real waterbody and forest cover results, asserting 0 exceptions and complete element rendering.

### Pytest Execution Summary
```text
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
collected 19 items

tests\test_ui_rendering_contract.py ....                                 [ 21%]
tests\test_interactive_aoi.py .......                                    [ 57%]
tests\test_streamlit_integration.py .                                    [ 63%]
tests\test_footprint_integrity.py .......                                [100%]

================== 19 passed, 1 warning in 61.18s (0:01:01) ===================
```

---

## 5. Remaining Errors or Constraints

- **Remaining Errors:** Zero (`0`). The application renders all components end-to-end without any Python traceback.
- **Constraints Maintained:**
  - Zero model retraining.
  - Zero BWOA algorithm alterations.
  - Zero dataset modifications.
  - Zero synthetic predictions or hardcoded confidence values.
  - Preserved v1.0.2 sub-metre footprint boundary calculation.
  - Cleanest two-line code modification to fix the root cause with full backward compatibility.
