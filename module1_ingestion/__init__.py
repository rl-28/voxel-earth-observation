"""
Module 1: Ingestion & Signal Processing
"""

try:
    from utils.geospatial_raster import ensure_rasterio_shim
    ensure_rasterio_shim()
except ImportError:
    pass
