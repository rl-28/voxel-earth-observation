"""
tools/build_india_dataset.py

End-to-end resumable dataset builder for real Sentinel-2 L2A Indian imagery
and ESA WorldCover 2021 v200 ground truth labels.

Tasks:
1. forest_cover: Tree cover (ESA WorldCover class 10) >= 60% (positive) vs <= 5% (negative)
2. waterbody: Permanent water (ESA WorldCover class 80) >= 10% (positive) vs <= 0.5% (negative)

Features:
- Resumable: skips any AOI whose tiles and .done marker already exist
- Timestamped progress logging flushed to stdout and logs/build_india.log
- Network requests use 120s timeout and up to 3 retries with exponential backoff
- Skips failing AOIs gracefully, records reason, and continues
- Generates data/india/tiles_index.csv, manifest_forest_cover.csv, manifest_waterbody.csv
- Generates results/label_check.png (8 sample tiles per task with RGB and WorldCover map)
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
from pystac_client import Client
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module1_ingestion.config import EARTH_SEARCH_STAC_URL, EARTH_SEARCH_BAND_TO_ASSET, SENTINEL_2_L2A_SPECTRAL_BANDS
from module1_ingestion.tile_processor import process_raster_to_tiles
from module1_ingestion.pipeline import Module1Pipeline
from config.constants import TASKS

# -----------------------------------------------------------------------------
# LOGGING UTILITIES
# -----------------------------------------------------------------------------
_LOG_FILE = None

def init_logger(log_path: str):
    global _LOG_FILE
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        _LOG_FILE = open(log_path, "a", encoding="utf-8")
    except Exception:
        _LOG_FILE = None

def log(msg: str, stage: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{stage:<12}] {msg}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        try:
            _LOG_FILE.write(line + "\n")
            _LOG_FILE.flush()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# AOI DEFINITIONS
# -----------------------------------------------------------------------------
AOIS = {
    "western_ghats": {
        "name": "Western Ghats forest",
        "center": (13.35, 75.30),
        "bbox": [75.15, 13.20, 75.45, 13.50],
        "notes": "Dense evergreen rainforest (Kudremukh range)",
    },
    "kanha": {
        "name": "Kanha, central India",
        "center": (22.30, 80.60),
        "bbox": [80.45, 22.15, 80.75, 22.45],
        "notes": "Central Indian moist deciduous sal forest",
    },
    "kaziranga": {
        "name": "Kaziranga, Assam",
        "center": (26.60, 93.40),
        "bbox": [93.25, 26.45, 93.55, 26.75],
        "notes": "Assam Brahmaputra riverine floodplain & wetland",
    },
    "hirakud": {
        "name": "Hirakud reservoir, Odisha",
        "center": (21.50, 83.80),
        "bbox": [83.65, 21.35, 83.95, 21.65],
        "notes": "Major reservoir on the Mahanadi River",
    },
    "chilika": {
        "name": "Chilika lake",
        "center": (19.70, 85.30),
        "bbox": [85.15, 19.55, 85.45, 19.85],
        "notes": "Large coastal brackish water lagoon",
    },
    "sardar_sarovar": {
        "name": "Sardar Sarovar, Gujarat",
        "center": (21.80, 73.75),
        "bbox": [73.60, 21.65, 73.90, 21.95],
        "notes": "Large reservoir on Narmada River",
    },
    "punjab": {
        "name": "Punjab cropland",
        "center": (30.90, 75.80),
        "bbox": [75.65, 30.75, 75.95, 31.05],
        "notes": "Intensive agricultural wheat cropland",
    },
    "thar": {
        "name": "Thar desert near Jaisalmer",
        "center": (26.90, 70.90),
        "bbox": [70.75, 26.75, 71.05, 27.05],
        "notes": "Arid sand dunes and bare desert soil",
    },
    "mumbai": {
        "name": "Mumbai sample",
        "center": (19.00, 72.85),
        "bbox": [72.80, 18.90, 73.00, 19.20],
        "notes": "Coastal urban land & Arabian sea coastline",
    },
}

WORLDCOVER_CLASSES = {
    0: ("No Data", "#000000"),
    10: ("Tree cover", "#006400"),
    20: ("Shrubland", "#ffbb22"),
    30: ("Grassland", "#ffff4c"),
    40: ("Cropland", "#f096ff"),
    50: ("Built-up", "#fa0000"),
    60: ("Bare / sparse", "#b4b4b4"),
    70: ("Snow and ice", "#f0f0f0"),
    80: ("Permanent water", "#0064c8"),
    90: ("Herbaceous wetland", "#0096a0"),
    95: ("Mangroves", "#00cf75"),
    100: ("Moss and lichen", "#fae6a0"),
}


def get_worldcover_tile_name(lat: float, lon: float) -> str:
    lat_idx = int(lat // 3 * 3)
    lon_idx = int(lon // 3 * 3)
    return f"N{lat_idx:02d}E{lon_idx:03d}"


def get_worldcover_url(tile_name: str) -> str:
    return f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile_name}_Map.tif"


# -----------------------------------------------------------------------------
# RETRY & NETWORK HELPERS
# -----------------------------------------------------------------------------
def retry_call(fn, max_retries: int = 3, retry_delay: float = 3.0, op_name: str = "network operation"):
    """Executes a function with up to max_retries retries and exponential backoff."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            log(f"Attempt {attempt}/{max_retries} failed for {op_name}: {e}", stage="RETRY")
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
    raise last_exc


# -----------------------------------------------------------------------------
# 1. SCENE DISCOVERY
# -----------------------------------------------------------------------------
def search_scene_for_aoi(
    aoi_key: str,
    aoi_info: Dict[str, Any],
    client: Client,
    date_start: str,
    date_end: str,
    cloud_limit: float,
) -> Dict[str, Any]:
    """Finds lowest cloud scene fully covering the AOI."""
    log(f"Searching scenes for {aoi_info['name']} (bbox: {aoi_info['bbox']})", stage="SCENE_SEARCH")
    bbox = aoi_info["bbox"]

    def _query():
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{date_start}/{date_end}",
            query={"eo:cloud_cover": {"lte": cloud_limit}},
            max_items=50,
        )
        return list(search.items())

    items = retry_call(_query, max_retries=3, op_name=f"STAC search for {aoi_key}")

    candidates = []
    for it in items:
        ib = it.bbox
        full = ib[0] <= bbox[0] and ib[1] <= bbox[1] and ib[2] >= bbox[2] and ib[3] >= bbox[3]
        cc = it.properties.get("eo:cloud_cover", 100.0)
        dt = it.properties.get("datetime", "")[:10]
        if full:
            candidates.append((cc, dt, it))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_cc, best_dt, best_item = candidates[0]
    elif items:
        items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100.0))
        best_item = items[0]
        best_cc = best_item.properties.get("eo:cloud_cover", 100.0)
        best_dt = best_item.properties.get("datetime", "")[:10]
    else:
        raise RuntimeError(f"No Sentinel-2 scene found for {aoi_key} within cloud limit {cloud_limit}%")

    log(f"Selected scene {best_item.id} (date: {best_dt}, cloud: {best_cc:.2f}%)", stage="SCENE_FOUND")
    return {
        "aoi_key": aoi_key,
        "name": aoi_info["name"],
        "scene_id": best_item.id,
        "date": best_dt,
        "cloud_cover": best_cc,
        "local_raw_dir": os.path.join("data", "raw", "india_raw", aoi_key),
        "stac_item": best_item,
        "bbox": bbox,
    }


# -----------------------------------------------------------------------------
# 2. BAND DOWNLOADING (WINDOWED COG CROP)
# -----------------------------------------------------------------------------
def download_cropped_bands(scene_info: Dict[str, Any], overwrite: bool = False) -> Dict[str, str]:
    """Downloads windowed cropped GeoTIFFs for the 12 Sentinel-2 bands covering the AOI bbox."""
    local_dir = scene_info["local_raw_dir"]
    os.makedirs(local_dir, exist_ok=True)
    item = scene_info["stac_item"]
    bbox = scene_info["bbox"]
    aoi_key = scene_info["aoi_key"]

    band_paths = {}

    # Check cached directory (e.g. Mumbai)
    if item is None:
        for band_key in SENTINEL_2_L2A_SPECTRAL_BANDS:
            p = os.path.join(local_dir, f"{band_key}.tif")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing cached band: {p}")
            band_paths[band_key] = p
        log(f"Using {len(band_paths)} pre-existing bands in {local_dir}", stage="BANDS_CACHED")
        return band_paths

    for band_key in SENTINEL_2_L2A_SPECTRAL_BANDS:
        dest_path = os.path.join(local_dir, f"{band_key}.tif")
        band_paths[band_key] = dest_path

        if os.path.exists(dest_path) and not overwrite and os.path.getsize(dest_path) > 1000:
            log(f"Band {band_key} already exists ({os.path.getsize(dest_path)} bytes), skipping.", stage="BAND_SKIP")
            continue

        asset_name = EARTH_SEARCH_BAND_TO_ASSET[band_key]
        if asset_name not in item.assets:
            raise KeyError(f"Asset '{asset_name}' not found in STAC item '{item.id}'")

        asset_url = item.assets[asset_name].href
        log(f"Downloading window for {band_key} ({asset_name}) from {item.id}...", stage="BAND_DOWNLOAD")

        def _download_band():
            # GDAL / rasterio window read with 120s timeout
            with rasterio.Env(GDAL_HTTP_TIMEOUT=120, GDAL_HTTP_MAX_RETRY=3):
                with rasterio.open(asset_url) as src:
                    tb = transform_bounds("EPSG:4326", src.crs, *bbox)
                    win = from_bounds(*tb, transform=src.transform)
                    win = win.round_offsets().round_lengths()

                    col_off = max(0, min(src.width - 1, int(win.col_off)))
                    row_off = max(0, min(src.height - 1, int(win.row_off)))
                    width = max(1, min(src.width - col_off, int(win.width)))
                    height = max(1, min(src.height - row_off, int(win.height)))
                    clamped_win = Window(col_off, row_off, width, height)

                    arr = src.read(1, window=clamped_win)
                    win_transform = rasterio.windows.transform(clamped_win, src.transform)

                    meta = src.meta.copy()
                    meta.update({
                        "height": arr.shape[0],
                        "width": arr.shape[1],
                        "transform": win_transform,
                        "compress": "deflate",
                    })

                    with rasterio.open(dest_path, "w", **meta) as dst:
                        dst.write(arr, 1)

        retry_call(_download_band, max_retries=3, op_name=f"Download {aoi_key} {band_key}")
        log(f"Saved {band_key} ({os.path.getsize(dest_path)} bytes)", stage="BAND_SAVED")

    return band_paths


# -----------------------------------------------------------------------------
# 3. TILING, MODULE 1 & WORLDCOVER LABELS
# -----------------------------------------------------------------------------
def process_aoi(
    scene_info: Dict[str, Any],
    band_paths: Dict[str, str],
    max_tiles: int = 120,
) -> List[Dict[str, Any]]:
    """
    Slices non-overlapping 256x256 tiles, processes through Module1Pipeline.for_module3(),
    resamples ESA WorldCover 2021 v200, and returns per-tile label records.
    """
    aoi_key = scene_info["aoi_key"]
    scene_id = scene_info["scene_id"]
    center_lat, center_lon = AOIS[aoi_key]["center"]

    out_dir = os.path.join("data", "india", "tiles", aoi_key)
    os.makedirs(out_dir, exist_ok=True)
    done_marker = os.path.join(out_dir, ".done")
    records_file = os.path.join(out_dir, "records.json")

    # 1. WorldCover AOI slice into memory (1 request per AOI)
    wc_tile_id = get_worldcover_tile_name(center_lat, center_lon)
    wc_url = get_worldcover_url(wc_tile_id)
    log(f"Reading ESA WorldCover 2021 v200 tile {wc_tile_id}...", stage="WORLDCOVER_READ")

    def _read_wc_slice():
        with rasterio.Env(GDAL_HTTP_TIMEOUT=120, GDAL_HTTP_MAX_RETRY=3):
            with rasterio.open(wc_url) as wc_src:
                buf_bbox = [
                    scene_info["bbox"][0] - 0.05,
                    scene_info["bbox"][1] - 0.05,
                    scene_info["bbox"][2] + 0.05,
                    scene_info["bbox"][3] + 0.05,
                ]
                wc_win = from_bounds(*buf_bbox, transform=wc_src.transform).round_offsets().round_lengths()
                c_off = max(0, min(wc_src.width - 1, int(wc_win.col_off)))
                r_off = max(0, min(wc_src.height - 1, int(wc_win.row_off)))
                w_len = max(1, min(wc_src.width - c_off, int(wc_win.width)))
                h_len = max(1, min(wc_src.height - r_off, int(wc_win.height)))
                clamped_win = Window(c_off, r_off, w_len, h_len)

                wc_arr = wc_src.read(1, window=clamped_win)
                wc_transform = rasterio.windows.transform(clamped_win, wc_src.transform)
                wc_crs = wc_src.crs
                return wc_arr, wc_transform, wc_crs

    wc_aoi_arr, wc_aoi_transform, wc_src_crs = retry_call(
        _read_wc_slice, max_retries=3, op_name=f"WorldCover read for {aoi_key}"
    )
    log(f"Cached WorldCover array in-memory: shape {wc_aoi_arr.shape}", stage="WORLDCOVER_READY")

    # 2. Pipeline for Module 3
    pipeline = Module1Pipeline.for_module3()
    tile_records = []
    saved_count = 0

    log(f"Extracting up to {max_tiles} valid 256x256 tiles with Module 1...", stage="TILING_START")

    for raw_tile_dict in process_raster_to_tiles(band_paths, reference_band_key="B04"):
        if saved_count >= max_tiles:
            break

        row_idx = raw_tile_dict["row_idx"]
        col_idx = raw_tile_dict["col_idx"]
        tile_name = f"tile_{row_idx:03d}_{col_idx:03d}.npy"
        tile_path = os.path.join(out_dir, tile_name)
        tile_rel_path = os.path.join("data", "india", "tiles", aoi_key, tile_name).replace("\\", "/")

        # If tile already on disk from previous run, load it for validation
        if os.path.exists(tile_path) and os.path.getsize(tile_path) > 3000000:
            processed_tensor = np.load(tile_path)
            valid_ratio = 1.0
        else:
            try:
                m1_res = pipeline.process_tile(raw_tile_dict)
            except Exception:
                continue

            processed_tensor = m1_res.processed_tensor
            valid_mask = m1_res.valid_mask
            valid_ratio = float(np.mean(valid_mask))
            if valid_ratio < 0.95:  # skip tiles with > 5% nodata
                continue

            np.save(tile_path, processed_tensor.astype(np.float32))

        # 3. Resample WorldCover to tile grid (NEAREST)
        tile_transform = raw_tile_dict["transform"]
        tile_crs = raw_tile_dict["crs"]
        tile_bounds = raw_tile_dict["bounds"]

        wc_tile_arr = np.zeros((256, 256), dtype=np.uint8)
        reproject(
            source=wc_aoi_arr,
            destination=wc_tile_arr,
            src_transform=wc_aoi_transform,
            src_crs=wc_src_crs,
            dst_transform=tile_transform,
            dst_crs=tile_crs,
            resampling=Resampling.nearest,
        )

        total_pixels = 256 * 256
        class_counts = {c: int(np.sum(wc_tile_arr == c)) for c in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]}
        class_fracs = {c: class_counts[c] / total_pixels for c in class_counts}
        geo_bounds = transform_bounds(tile_crs, "EPSG:4326", *tile_bounds)

        block_id = f"b{row_idx // 4}_{col_idx // 4}"
        location_id = f"{aoi_key}_{scene_id}_{block_id}"

        rec = {
            "tile_path": tile_rel_path,
            "aoi": aoi_key,
            "scene_id": scene_id,
            "row_idx": row_idx,
            "col_idx": col_idx,
            "location_id": location_id,
            "min_lon": geo_bounds[0],
            "min_lat": geo_bounds[1],
            "max_lon": geo_bounds[2],
            "max_lat": geo_bounds[3],
            "valid_ratio": round(valid_ratio, 4),
            "frac_10_tree": round(class_fracs[10], 4),
            "frac_20_shrub": round(class_fracs[20], 4),
            "frac_30_grass": round(class_fracs[30], 4),
            "frac_40_crop": round(class_fracs[40], 4),
            "frac_50_built": round(class_fracs[50], 4),
            "frac_60_bare": round(class_fracs[60], 4),
            "frac_70_snow": round(class_fracs[70], 4),
            "frac_80_water": round(class_fracs[80], 4),
            "frac_90_wetland": round(class_fracs[90], 4),
            "frac_95_mangrove": round(class_fracs[95], 4),
            "frac_100_moss": round(class_fracs[100], 4),
            "dominant_class": max(class_counts, key=class_counts.get),
        }

        tile_records.append(rec)
        saved_count += 1

    # Save per-AOI records and write done marker
    with open(records_file, "w", encoding="utf-8") as f:
        json.dump(tile_records, f)

    with open(done_marker, "w", encoding="utf-8") as f:
        json.dump({
            "aoi": aoi_key,
            "scene_id": scene_id,
            "tiles_count": saved_count,
            "completed_at": datetime.now().isoformat(),
        }, f, indent=2)

    log(f"Completed AOI {aoi_key}: {saved_count} tiles generated, .done marker written.", stage="AOI_DONE")
    return tile_records


# -----------------------------------------------------------------------------
# 4. BINARY LABELS, SPLITS & MANIFESTS
# -----------------------------------------------------------------------------
def derive_binary_labels_and_splits(
    df: pd.DataFrame,
    forest_pos_thresh: float = 0.60,
    forest_neg_thresh: float = 0.05,
    water_pos_thresh: float = 0.10,
    water_neg_thresh: float = 0.005,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Derives binary labels and assigns spatial group splits by AOI:
    - Test AOIs (holdout 2 AOIs): 'kanha' (Forest-rich) and 'hirakud' (Water-rich)
    - Val AOI (holdout 1 AOI): 'sardar_sarovar' (Water-rich)
    - Train AOIs: 'western_ghats', 'kaziranga', 'chilika', 'punjab', 'thar', 'mumbai'
    """
    test_aois = ["kanha", "hirakud"]
    val_aois = ["sardar_sarovar"]

    def get_split(aoi):
        if aoi in test_aois:
            return "test"
        elif aoi in val_aois:
            return "val"
        else:
            return "train"

    df["split"] = df["aoi"].apply(get_split)

    # 1. Forest Cover Task: class 10 >= 60% pos, <= 5% neg, drop in between
    forest_rows = []
    for _, row in df.iterrows():
        f10 = row["frac_10_tree"]
        if f10 >= forest_pos_thresh:
            forest_rows.append({
                "tile_path": row["tile_path"],
                "label": "Forest",
                "label_id": 1,
                "split": row["split"],
                "location_id": row["location_id"],
                "aoi": row["aoi"],
                "tree_fraction": f10,
            })
        elif f10 <= forest_neg_thresh:
            forest_rows.append({
                "tile_path": row["tile_path"],
                "label": "Non-forest",
                "label_id": 0,
                "split": row["split"],
                "location_id": row["location_id"],
                "aoi": row["aoi"],
                "tree_fraction": f10,
            })

    df_forest = pd.DataFrame(forest_rows)

    # 2. Waterbody Task: class 80 >= 10% pos, <= 0.5% neg, drop in between
    water_rows = []
    for _, row in df.iterrows():
        f80 = row["frac_80_water"]
        if f80 >= water_pos_thresh:
            water_rows.append({
                "tile_path": row["tile_path"],
                "label": "Water",
                "label_id": 1,
                "split": row["split"],
                "location_id": row["location_id"],
                "aoi": row["aoi"],
                "water_fraction": f80,
            })
        elif f80 <= water_neg_thresh:
            water_rows.append({
                "tile_path": row["tile_path"],
                "label": "Non-water",
                "label_id": 0,
                "split": row["split"],
                "location_id": row["location_id"],
                "aoi": row["aoi"],
                "water_fraction": f80,
            })

    df_water = pd.DataFrame(water_rows)
    return df_forest, df_water


# -----------------------------------------------------------------------------
# 5. VISUAL INSPECTION SHEET (results/label_check.png)
# -----------------------------------------------------------------------------
def generate_label_check_figure(
    df_forest: pd.DataFrame,
    df_water: pd.DataFrame,
    df_index: pd.DataFrame,
    output_png: str = "results/label_check.png",
):
    """Saves 8 tiles per task (RGB composite alongside WorldCover map)."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig, axes = plt.subplots(4, 8, figsize=(24, 12))
    plt.subplots_adjust(wspace=0.15, hspace=0.3)

    tasks_data = [
        ("Task: forest_cover", df_forest, "Forest", "Non-forest", 0, 1),
        ("Task: waterbody", df_water, "Water", "Non-water", 2, 3),
    ]

    cmap_colors = [WORLDCOVER_CLASSES[c][1] for c in sorted(WORLDCOVER_CLASSES.keys())]
    cmap = mcolors.ListedColormap(cmap_colors)
    bounds = list(sorted(WORLDCOVER_CLASSES.keys())) + [110]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for task_name, df_task, pos_cls, neg_cls, row_rgb, row_wc in tasks_data:
        pos_df = df_task[df_task["label"] == pos_cls]
        neg_df = df_task[df_task["label"] == neg_cls]
        pos_n = min(4, len(pos_df))
        neg_n = min(4, len(neg_df))

        pos_samples = pos_df.sample(n=pos_n, random_state=42) if pos_n > 0 else pos_df
        neg_samples = neg_df.sample(n=neg_n, random_state=42) if neg_n > 0 else neg_df
        samples = pd.concat([pos_samples, neg_samples]).reset_index(drop=True)

        for col_idx in range(min(8, len(samples))):
            s_row = samples.iloc[col_idx]
            tile_path = s_row["tile_path"]
            label = s_row["label"]
            aoi = s_row["aoi"]

            arr = np.load(tile_path)
            # RGB: B04 (idx 3), B03 (idx 2), B02 (idx 1)
            r = arr[3]
            g = arr[2]
            b = arr[1]
            rgb = np.stack([r, g, b], axis=-1)

            # Percentile stretch
            p2, p98 = np.percentile(rgb, (2, 98))
            rgb_stretched = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)

            ax_rgb = axes[row_rgb, col_idx]
            ax_rgb.imshow(rgb_stretched)
            ax_rgb.set_title(f"{pos_cls if label == pos_cls else neg_cls}\n[{aoi}]", fontsize=9, fontweight="bold")
            ax_rgb.axis("off")

            # WorldCover slice
            center_lat, center_lon = AOIS[aoi]["center"]
            wc_tile_id = get_worldcover_tile_name(center_lat, center_lon)
            idx_match = df_index[df_index["tile_path"] == tile_path].iloc[0]

            with rasterio.open(get_worldcover_url(wc_tile_id)) as wc_src:
                tb = (idx_match["min_lon"], idx_match["min_lat"], idx_match["max_lon"], idx_match["max_lat"])
                win = from_bounds(*tb, transform=wc_src.transform)
                wc_arr = wc_src.read(1, window=win, out_shape=(256, 256), resampling=Resampling.nearest)

            ax_wc = axes[row_wc, col_idx]
            ax_wc.imshow(wc_arr, cmap=cmap, norm=norm)
            ax_wc.set_title(f"WorldCover\n({label})", fontsize=8)
            ax_wc.axis("off")

    fig.suptitle("VOXEL Real Indian Satellite Dataset: Visual Label Inspection (Sentinel-2 L2A RGB vs ESA WorldCover 2021 v200)", fontsize=14, fontweight="bold")
    plt.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close()
    log(f"Visual label check sheet saved to: {output_png}", stage="VISUAL_CHECK")


# -----------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VOXEL Real Indian Dataset Builder")
    parser.add_argument("--aois", type=str, default=None, help="Comma-separated AOI keys to process")
    parser.add_argument("--max_tiles_per_aoi", type=int, default=120, help="Max valid tiles per AOI (default: 120)")
    parser.add_argument("--cloud_limit", type=float, default=10.0, help="STAC cloud limit percentage (default: 10.0)")
    parser.add_argument("--date_start", type=str, default="2020-11-01", help="Start date (default: 2020-11-01)")
    parser.add_argument("--date_end", type=str, default="2021-03-31", help="End date (default: 2021-03-31)")
    parser.add_argument("--forest_pos_thresh", type=float, default=0.60, help="Forest positive threshold (default: 0.60)")
    parser.add_argument("--forest_neg_thresh", type=float, default=0.05, help="Forest negative threshold (default: 0.05)")
    parser.add_argument("--water_pos_thresh", type=float, default=0.10, help="Water positive threshold (default: 0.10)")
    parser.add_argument("--water_neg_thresh", type=float, default=0.005, help="Water negative threshold (default: 0.005)")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip already completed AOIs (default: True)")
    parser.add_argument("--overwrite_crops", action="store_true", help="Overwrite local band crops")
    parser.add_argument("--log_file", type=str, default="logs/build_india.log", help="Log file path")
    args = parser.parse_args()

    init_logger(args.log_file)
    log("=" * 80, stage="START")
    log("VOXEL REAL INDIAN DATASET BUILDER INITIALIZED", stage="START")
    log(f"Log file: {args.log_file}", stage="START")
    log(f"Parameters: max_tiles={args.max_tiles_per_aoi}, cloud_limit={args.cloud_limit}%, dates={args.date_start} to {args.date_end}", stage="START")
    log(f"Thresholds: forest=[<={args.forest_neg_thresh*100}%, >={args.forest_pos_thresh*100}%], water=[<={args.water_neg_thresh*100}%, >={args.water_pos_thresh*100}%]", stage="START")
    log("=" * 80, stage="START")

    target_aoi_keys = [k.strip() for k in args.aois.split(",")] if args.aois else list(AOIS.keys())
    log(f"Target AOIs to process ({len(target_aoi_keys)}): {target_aoi_keys}", stage="AOI_LIST")

    stac_client = Client.open(EARTH_SEARCH_STAC_URL)
    all_tile_records = []
    completed_aois = []
    failed_aois = []

    for aoi_key in target_aoi_keys:
        if aoi_key not in AOIS:
            log(f"Unknown AOI key: {aoi_key}, skipping.", stage="WARN")
            continue

        aoi_info = AOIS[aoi_key]
        aoi_tiles_dir = os.path.join("data", "india", "tiles", aoi_key)
        done_marker = os.path.join(aoi_tiles_dir, ".done")
        records_file = os.path.join(aoi_tiles_dir, "records.json")

        # Check if already completed and resume is active
        if args.resume and os.path.exists(done_marker) and os.path.exists(records_file):
            try:
                with open(records_file, "r", encoding="utf-8") as rf:
                    recs = json.load(rf)
                log(f"AOI {aoi_key} already completed (.done marker present). Loaded {len(recs)} cached records.", stage="RESUME_SKIP")
                all_tile_records.extend(recs)
                completed_aois.append(aoi_key)
                continue
            except Exception as e:
                log(f"Failed to read cached records for {aoi_key}: {e}. Reprocessing.", stage="WARN")

        # Execute processing for this AOI
        log(f">>> STARTING AOI: {aoi_key} ({aoi_info['name']})", stage="AOI_START")
        try:
            # 1. Scene discovery
            if aoi_key == "mumbai" and os.path.exists("data/raw/real_scene_mumbai/B04.tif"):
                scene_info = {
                    "aoi_key": aoi_key,
                    "name": aoi_info["name"],
                    "scene_id": "S2B_42QZF_20210328_0_L2A",
                    "date": "2021-03-28",
                    "cloud_cover": 0.07,
                    "local_raw_dir": "data/raw/real_scene_mumbai",
                    "stac_item": None,
                    "bbox": aoi_info["bbox"],
                }
            else:
                scene_info = search_scene_for_aoi(
                    aoi_key=aoi_key,
                    aoi_info=aoi_info,
                    client=stac_client,
                    date_start=args.date_start,
                    date_end=args.date_end,
                    cloud_limit=args.cloud_limit,
                )

            # 2. Download cropped bands
            band_paths = download_cropped_bands(scene_info, overwrite=args.overwrite_crops)

            # 3. Tiling, Module 1 & WorldCover labels
            recs = process_aoi(scene_info, band_paths, max_tiles=args.max_tiles_per_aoi)
            all_tile_records.extend(recs)
            completed_aois.append(aoi_key)
            log(f">>> FINISHED AOI: {aoi_key} ({len(recs)} valid tiles)", stage="AOI_SUCCESS")

        except Exception as e:
            log(f">>> FAILED AOI {aoi_key}: {e}. Skipping and continuing.", stage="AOI_FAILED")
            failed_aois.append((aoi_key, str(e)))

    log("=" * 80, stage="SUMMARY")
    log(f"All AOIs processed. Completed: {len(completed_aois)} ({completed_aois}), Failed: {len(failed_aois)}", stage="SUMMARY")
    if failed_aois:
        for fa, reason in failed_aois:
            log(f"  • Failed AOI: {fa} -> {reason}", stage="FAIL_REASON")

    if not all_tile_records:
        log("No tile records generated across all AOIs! Exiting.", stage="ERROR")
        sys.exit(1)

    # 4. Save master tiles_index.csv
    df_index = pd.DataFrame(all_tile_records)
    index_csv = os.path.join("data", "india", "tiles_index.csv")
    df_index.to_csv(index_csv, index=False)
    log(f"Saved master tiles index ({len(df_index)} tiles) to {index_csv}", stage="TILES_INDEX")

    # 5. Derive Binary Labels and Manifests
    df_forest, df_water = derive_binary_labels_and_splits(
        df_index,
        forest_pos_thresh=args.forest_pos_thresh,
        forest_neg_thresh=args.forest_neg_thresh,
        water_pos_thresh=args.water_pos_thresh,
        water_neg_thresh=args.water_neg_thresh,
    )

    forest_manifest = os.path.join("data", "india", "manifest_forest_cover.csv")
    water_manifest = os.path.join("data", "india", "manifest_waterbody.csv")

    df_forest[["tile_path", "label", "split", "location_id"]].to_csv(forest_manifest, index=False)
    df_water[["tile_path", "label", "split", "location_id"]].to_csv(water_manifest, index=False)

    log(f"Saved manifest_forest_cover.csv: {len(df_forest)} tiles", stage="MANIFEST")
    log(f"Saved manifest_waterbody.csv: {len(df_water)} tiles", stage="MANIFEST")

    # 6. Report Task Statistics and Class Imbalance
    log("=" * 80, stage="REPORT")
    log("DATASET CLASS BALANCE AND SPLIT REPORT", stage="REPORT")
    for task_name, df_t, pos_lbl, neg_lbl in [
        ("forest_cover", df_forest, "Forest", "Non-forest"),
        ("waterbody", df_water, "Water", "Non-water"),
    ]:
        log(f"--- TASK: {task_name.upper()} ---", stage="TASK_STATS")
        n_pos = int(np.sum(df_t["label"] == pos_lbl))
        n_neg = int(np.sum(df_t["label"] == neg_lbl))
        tot = n_pos + n_neg
        pos_pct = (n_pos / tot * 100.0) if tot > 0 else 0.0
        neg_pct = (n_neg / tot * 100.0) if tot > 0 else 0.0
        minority_pct = min(pos_pct, neg_pct)

        log(f"Total: {tot} tiles | {pos_lbl}: {n_pos} ({pos_pct:.1f}%) | {neg_lbl}: {n_neg} ({neg_pct:.1f}%)", stage="TASK_STATS")
        if minority_pct < 15.0:
            log(f"[IMBALANCE WARNING] Minority class is {minority_pct:.1f}% (< 15.0% threshold). Task is imbalanced.", stage="WARN")
        else:
            log(f"[BALANCE OK] Minority class is {minority_pct:.1f}% (>= 15.0% threshold). Task is well-balanced.", stage="TASK_STATS")

        # Split breakdown
        for sp in ["train", "val", "test"]:
            sub = df_t[df_t["split"] == sp]
            p_cnt = int(np.sum(sub["label"] == pos_lbl))
            n_cnt = int(np.sum(sub["label"] == neg_lbl))
            log(f"  Split '{sp}': {len(sub)} tiles ({pos_lbl}={p_cnt}, {neg_lbl}={n_cnt})", stage="SPLIT_BREAKDOWN")

    # 7. Visual Inspection Sheet
    log("Generating results/label_check.png visual sheet...", stage="VISUAL_CHECK")
    try:
        generate_label_check_figure(df_forest, df_water, df_index, "results/label_check.png")
    except Exception as e:
        log(f"Failed to generate label_check.png: {e}", stage="WARN")

    log("=" * 80, stage="DONE")
    log("DATASET BUILD COMPLETE", stage="DONE")


if __name__ == "__main__":
    main()
