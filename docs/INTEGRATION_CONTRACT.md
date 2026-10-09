# VOXEL Module 3: Integration Contract & Interface Specification

This contract formally specifies the expected data structures, on-disk file formats, and handoff protocols between **Module 1 (Data Ingestion)**, **Module 2 (Band Optimization)**, and **Module 3 (Deep Learning & Vector Retrieval)**.

---

## 1. Spatial-Spectral Tile Specification (Module 1 Handshake)

| Attribute | Specification | Notes |
| :--- | :--- | :--- |
| **Tensor Dimensions** | `(13, 256, 256)` | Channel-first format `(C=13, H=256, W=256)`. Channel-last `(256, 256, 13)` is auto-detected. |
| **Data Type** | `float32` | Normalized surface reflectance. |
| **Value Range** | `[0.0, 1.0]` | Scaled via `DN / 10000.0`, clipped to $[0.0, 1.0]$. |
| **Band Count** | Exactly `13` | Files with $\ne 13$ bands will raise a `ValueError`. |
| **Band Order** | Strict Sentinel-2 L2A order | See canonical band table below. |

### Canonical Sentinel-2 Band Order
```
Index 00: B01 (Coastal Aerosol - 443 nm)
Index 01: B02 (Blue - 490 nm)
Index 02: B03 (Green - 560 nm)
Index 03: B04 (Red - 665 nm)
Index 04: B05 (Vegetation Red Edge 1 - 705 nm)
Index 05: B06 (Vegetation Red Edge 2 - 740 nm)
Index 06: B07 (Vegetation Red Edge 3 - 783 nm)
Index 07: B08 (Broad NIR - 842 nm)
Index 08: B8A (Narrow NIR - 865 nm)
Index 09: B09 (Water Vapour - 945 nm)
Index 10: B10 (Cirrus - 1375 nm)
Index 11: B11 (SWIR 1 - 1610 nm)
Index 12: B12 (SWIR 2 - 2190 nm)
```

---

## 2. Land Cover Target Classes

Labels must resolve to integer indices `0` through `9`, matching `LAND_COVER_CLASSES` in `config/constants.py`:

```python
0: "Annual Crop"
1: "Forest"
2: "Herbaceous Vegetation"
3: "Highway / Infrastructure"
4: "Industrial Area"
5: "Pasture"
6: "Permanent Crop"
7: "Residential Area"
8: "River / Inland Water"
9: "Sea / Marine Water"
```
*Note: If raw labels use strings or non-standard IDs, provide a JSON mapping via `--label_map` (e.g. `{"forest": 1, "river": 8}`).*

---

## 3. On-Disk Data Layouts & Manifest Schema

Module 3 supports two on-disk storage layouts indexed by a `manifest.csv`.

### Directory Tree Example
```
data/
├── manifest.csv
└── tiles/
    ├── tile_0001.npy        # Layout A: NumPy array (13, 256, 256)
    ├── tile_0002.npy
    ├── tile_0003.tif        # Layout B: 13-band GeoTIFF
    └── tile_0004.tif
```

### Manifest Schema (`manifest.csv`)
| Column | Required | Description | Example |
| :--- | :---: | :--- | :--- |
| `tile_path` | **Yes** | Relative to data dir or absolute path to `.npy` or `.tif` | `tiles/tile_0001.npy` |
| `label` | **Yes** | Integer `0-9` or string mapped via `--label_map` | `1` or `"Forest"` |
| `split` | No | Pre-assigned split: `train`, `val`, or `test` | `train` |
| `location_id` | No | Spatial group ID to prevent spatial data leakage across splits | `LOC_REGION_04` |

---

## 4. Band Selection Mask Specification (Module 2 Handshake)

Module 3 accepts band masks in **three formats**:
1. **JSON Format** (Recommended, see `config/example_band_mask.json`):
   ```json
   {
     "selected_indices": [1, 2, 3, 7, 11, 12],
     "selected_bands": ["B02", "B03", "B04", "B08", "B11", "B12"],
     "total_bands": 13,
     "method": "BWOA",
     "dataset": "Sentinel-2 L2A"
   }
   ```
2. **NumPy Array (`.npy`)**: 1D binary array of length 13 (`np.float32` or `np.uint8`).
3. **PyTorch Tensor (`.pt`)**: 1D binary tensor of shape `(13,)`.

*Rules:* Mask length must equal 13, values must be strictly binary $\{0, 1\}$, and at least 1 band must be active.

---

## 5. Teammate Handoff Checklists

### What Ingestion (Teammate 1) Must Deliver:
- [ ] 13-band tiles saved as either `.npy` or `.tif` in `(13, 256, 256)` format.
- [ ] Pixel values scaled to $[0.0, 1.0]$ or raw DN with scale divisor documented.
- [ ] `manifest.csv` referencing all valid tiles and ground truth labels.
- [ ] If spatial grouping is used, populated `location_id` column.

### What Optimization (Teammate 2) Must Deliver:
- [ ] Output mask file saved as JSON, `.npy`, or `.pt`.
- [ ] 0-indexed band indices referencing canonical 13-band Sentinel-2 order.
- [ ] Metadata documenting optimization algorithm (e.g. `BWOA`, `Jaya`, `RF-DE`).

---

## 6. Dependency & Environment Conflict Notes

To ensure smooth integration across modules, be aware of known package sensitivities:
- **`GDAL` / `rasterio`**: On Windows, install pre-compiled binary wheels or use `pip install rasterio`. Avoid conda-gdal mixing.
- **`mealpy`**: Optimization algorithms must output standard JSON or NumPy arrays to hand off to Module 3 without requiring `mealpy` in inference environments.
- **`shap` / `numba`**: Pinned to `shap==0.49.1` and `numba==0.68.0` on Python 3.10.
