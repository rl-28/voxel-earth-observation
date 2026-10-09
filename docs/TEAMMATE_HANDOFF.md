# VOXEL Module 3: Teammate Handoff & Integration Guide

> **Module 3:** Deep Learning (CNN-ViT FusionNet) & Explainable Vector Retrieval  
> **Target Audience:** Teammate 1 (Module 1 - Data Ingestion) & Teammate 2 (Module 2 - Band Optimization)  
> **Branch:** `feature/module3-gpu-training`  
> **Status:** Pre-merge ready (Self-contained, GPU-verified, 18/18 passing tests)

---

## 1. Module 3 Specifications & Interface Contract

Module 3 expects data formatted according to the single source of truth defined in `config/constants.py`:

### A. Strict Spectral Band Order (13 Bands, 0-Indexed)
Tiles must strictly follow canonical Sentinel-2 L2A band ordering:

| Index | Band Key | Band Description | Central Wavelength |
| :---: | :---: | :--- | :---: |
| `0` | `B01` | Coastal Aerosol | 443 nm |
| `1` | `B02` | Blue | 490 nm |
| `2` | `B03` | Green | 560 nm |
| `3` | `B04` | Red | 665 nm |
| `4` | `B05` | Vegetation Red Edge 1 (VRE 1) | 705 nm |
| `5` | `B06` | Vegetation Red Edge 2 (VRE 2) | 740 nm |
| `6` | `B07` | Vegetation Red Edge 3 (VRE 3) | 783 nm |
| `7` | `B08` | Broad Near-Infrared (NIR) | 842 nm |
| `8` | `B8A` | Narrow Near-Infrared (Narrow NIR) | 865 nm |
| `9` | `B09` | Water Vapour | 945 nm |
| `10` | `B10` | Cirrus | 1375 nm |
| `11` | `B11` | Shortwave Infrared 1 (SWIR 1) | 1610 nm |
| `12` | `B12` | Shortwave Infrared 2 (SWIR 2) | 2190 nm |

### B. Spatial Dimensions & Tile Format
- **Shape:** `(13, 256, 256)` (Channel-first `(C=13, H=256, W=256)`).
  *(Note: Channel-last `(256, 256, 13)` is auto-detected and transposed automatically).*
- **Data Type:** `float32` or raw integer DN.
- **Supported File Formats:**
  - NumPy arrays (`.npy`): `(13, 256, 256)` or `(256, 256, 13)`.
  - Multi-band GeoTIFF (`.tif` / `.tiff`): 13 bands.

### C. Scaling Rule
- **Target Value Range:** Reflectance values in $[0.0, 1.0]$.
- **Scale Divisor:** Sentinel-2 L2A raw Digital Numbers (DN $\ge 1.0$) are divided by `DEFAULT_SCALE_DIVISOR = 10000.0` and clipped to $[0.0, 1.0]$.
- **Auto-Detection:** Module 3 inspects `max(tensor)`. If values are already $\le 1.0$, division is skipped.

### D. Canonical Land Cover Classes (EuroSAT-10)
All labels must map to integer IDs `0` through `9`:

| Class ID | Class Name |
| :---: | :--- |
| `0` | `Annual Crop` |
| `1` | `Forest` |
| `2` | `Herbaceous Vegetation` |
| `3` | `Highway / Infrastructure` |
| `4` | `Industrial Area` |
| `5` | `Pasture` |
| `6` | `Permanent Crop` |
| `7` | `Residential Area` |
| `8` | `River / Inland Water` |
| `9` | `Sea / Marine Water` |

*(Note: String labels matching these names or mapped via a `--label_map` JSON file are accepted).*

---

## 2. Ingestion Requirements (For Teammate 1 - Module 1)

Module 3 indexes data via a `manifest.csv`.

### Manifest Schema
| Column Name | Required | Data Type | Description |
| :--- | :---: | :---: | :--- |
| `tile_path` | **Yes** | `string` | Relative path from `--data_dir` or absolute path to `.npy` / `.tif`. |
| `label` | **Yes** | `int` or `str` | Integer ID `0-9` or class name string. |
| `split` | No | `str` | Pre-assigned split: `train`, `val`, or `test`. *(If omitted, Module 3 generates a 70/15/15 stratified split).* |
| `location_id` | No | `str` | Spatial cluster / granule identifier used for group splitting to prevent spatial autocorrelation leakage. |

### Example Manifest (`manifest.csv`)
```csv
tile_path,label,split,location_id
tiles/tile_0001.npy,1,train,LOC_REGION_04
tiles/tile_0002.npy,4,train,LOC_REGION_04
tiles/tile_0020.tif,Forest,val,LOC_REGION_12
tiles/tile_0021.tif,River / Inland Water,test,LOC_REGION_18
```

---

## 3. Band Optimization Requirements (For Teammate 2 - Module 2)

Module 3 can dynamically slice input tensors to any subset of bands optimized by Module 2 metaheuristics.

### A. Recommended JSON Format (`config/example_band_mask.json`)
```json
{
  "selected_indices": [1, 2, 3, 7, 11, 12],
  "selected_bands": ["B02", "B03", "B04", "B08", "B11", "B12"],
  "total_bands": 13,
  "method": "BWOA",
  "dataset": "Sentinel-2 L2A"
}
```
*Requirement:* `selected_indices` must be **0-indexed** integers ($0 \le i \le 12$) corresponding to the canonical 13-band table.

### B. Supported Alternative Formats
1. **NumPy Binary Array (`.npy`):** A 1D array of shape `(13,)` with values in $\{0, 1\}$ (e.g., `np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1])`).
2. **PyTorch Binary Tensor (`.pt`):** A 1D tensor of shape `(13,)` with values in $\{0, 1\}$.

---

## 4. Pre-Merge Integration Validation

Before opening a pull request or merging into `main`, please run the integration validation utility:

```bash
python tools/validate_integration.py --manifest <path_to_manifest.csv> --mask_path <path_to_mask.json>
```
*(Status: **coming soon** — this pre-merge validation CLI will be available once ingestion and optimization pipelines land. In the interim, ensure all specifications in this document are adhered to).*

---

## 5. Teammate Handoff Checklist

Before merging, please verify each of the following 6 requirements:

- [ ] **1. Band Order:** Does your tile array strictly follow Sentinel-2 13-band order `[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12]`? (`Yes` / `No`)
- [ ] **2. Reflectance Scaling:** Are tile pixel values normalized to $[0.0, 1.0]$ or raw Sentinel-2 DNs intended for divisor $10000.0$? (`Yes` / `No`)
- [ ] **3. Class IDs:** Do ground truth labels correspond to the 10 canonical classes (indices 0–9) specified in `config/constants.py`? (`Yes` / `No`)
- [ ] **4. Tile Spatial Size:** Are spatial dimensions formatted as $256 \times 256$ pixels ($H=256, W=256$)? (`Yes` / `No`)
- [ ] **5. Mask Schema:** Does your optimization mask output match the JSON schema (`selected_indices` 0-indexed, length 13) or a 13-element binary `.npy` / `.pt` vector? (`Yes` / `No`)
- [ ] **6. Location ID Availability:** Does `manifest.csv` include a `location_id` column to prevent spatial data leakage across train/val/test splits? (`Yes` / `No`)
