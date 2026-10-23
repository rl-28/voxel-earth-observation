"""
utils/geospatial_adapter.py

VOXEL-Controlled Geospatial Compatibility Adapter.
Provides guaranteed, reproducible rasterio compatibility across diverse execution
environments (including Windows hosts subject to WDAC code integrity restrictions).

Prioritizes official C-extension rasterio when available and unblocked, seamlessly
falling back to pure-Python geospatial_raster when binary extensions cannot execute.
"""

import sys
from utils.geospatial_raster import ensure_rasterio_shim

# Ensure shim is registered if official rasterio cannot load
ensure_rasterio_shim()

import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine, from_origin, guard_transform
from rasterio.crs import CRS
from rasterio.warp import transform_bounds, calculate_default_transform, reproject, Resampling

__all__ = [
    "rasterio",
    "Window",
    "Affine",
    "from_origin",
    "guard_transform",
    "CRS",
    "transform_bounds",
    "calculate_default_transform",
    "reproject",
    "Resampling",
    "ensure_rasterio_shim",
]
