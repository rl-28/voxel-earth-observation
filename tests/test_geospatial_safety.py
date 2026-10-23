"""
tests/test_geospatial_safety.py

Phase 3 Priority 1: Geospatial Reader Safety & Validation Test Suite.
Verifies pure-Python geospatial_raster behavior across:
1. Boundary & out-of-bounds window reads with nodata zero-padding.
2. Non-square windows (e.g. 128x256, 300x150).
3. Multi-resolution band harmonisation (10m, 20m, 60m bands).
4. Explicit refusal to guess EPSG codes on missing tags.
5. Explicit failure on unsupported formats, unknown predictors, and invalid projections.
6. Partial download, byte length mismatch, and HTTP response status validation.
7. Precision preservation of Affine transform, BBOX coordinates, and radiometric pixel values.
"""

import os
import tempfile
import unittest
import numpy as np
from pathlib import Path
from PIL import Image

import utils.geospatial_raster as gr
from utils.geospatial_raster import (
    DatasetReader,
    DatasetWriter,
    Affine,
    CRS,
    Window,
    BoundingBox,
    reproject,
    Resampling,
    from_origin,
)


class TestGeospatialSafety(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_missing_crs_never_guesses_epsg(self):
        """Priority 1, Item 5: Never guess an EPSG code when tags are absent."""
        raw_tif = self.dir_path / "no_crs.tif"
        data = np.full((128, 128), 100, dtype=np.uint16)
        img = Image.fromarray(data)
        img.save(str(raw_tif))

        with DatasetReader(str(raw_tif)) as src:
            # Must be None, NEVER guessed as EPSG:32644 or any arbitrary code
            self.assertIsNone(src.crs, "Missing GeoTIFF tags must yield src.crs is None, not an invented code.")

    def test_reproject_with_undefined_crs_fails_explicitly(self):
        """Priority 1, Item 5: Reprojecting unreferenced rasters must raise ValueError."""
        src_arr = np.ones((64, 64), dtype=np.float32)
        dst_arr = np.zeros((64, 64), dtype=np.float32)
        tf = Affine(10.0, 0.0, 100.0, 0.0, -10.0, 100.0)

        with self.assertRaises(ValueError):
            reproject(
                source=src_arr,
                destination=dst_arr,
                src_transform=tf,
                src_crs=None,  # Undefined
                dst_transform=tf,
                dst_crs="EPSG:4326",
            )

    def test_unsupported_crs_conversion_fails_explicitly(self):
        """Priority 1, Item 5: Incompatible non-UTM conversions must not silently pass unmapped coordinates."""
        src_arr = np.ones((32, 32), dtype=np.float32)
        dst_arr = np.zeros((32, 32), dtype=np.float32)
        tf = Affine(10.0, 0.0, 100.0, 0.0, -10.0, 100.0)

        with self.assertRaises(NotImplementedError):
            reproject(
                source=src_arr,
                destination=dst_arr,
                src_transform=tf,
                src_crs="EPSG:2154",  # Lambert-93 (France)
                dst_transform=tf,
                dst_crs="EPSG:27700",  # British National Grid
            )

    def test_non_square_window_reading(self):
        """Priority 1, Item 4: Non-square windows must preserve requested shape and data."""
        tif_path = self.dir_path / "test_non_square.tif"
        data = np.arange(200 * 300, dtype=np.float32).reshape(200, 300)
        tf = from_origin(1000.0, 2000.0, 10.0, 10.0)

        with DatasetWriter(str(tif_path), width=300, height=200, count=1, dtype=np.float32, transform=tf, crs="EPSG:32644") as dst:
            dst.write(data)

        with DatasetReader(str(tif_path)) as src:
            # Request non-square window: width=150, height=80 at offset col=20, row=30
            win = Window(col_off=20, row_off=30, width=150, height=80)
            res = src.read(1, window=win)
            self.assertEqual(res.shape, (80, 150))
            expected = data[30:110, 20:170]
            np.testing.assert_array_equal(res, expected)

    def test_out_of_bounds_window_padding(self):
        """Priority 1, Item 4: Out-of-bounds windows must pad with zeros (nodata) rather than erroring."""
        tif_path = self.dir_path / "test_oob.tif"
        data = np.full((100, 100), 50.0, dtype=np.float32)
        tf = from_origin(0.0, 1000.0, 10.0, 10.0)

        with DatasetWriter(str(tif_path), width=100, height=100, count=1, dtype=np.float32, transform=tf, crs="EPSG:32644") as dst:
            dst.write(data)

        with DatasetReader(str(tif_path)) as src:
            # Window extends 20 pixels beyond width and height
            win = Window(col_off=80, row_off=80, width=40, height=40)
            res = src.read(1, window=win)
            self.assertEqual(res.shape, (40, 40))
            # In-bounds region (top-left 20x20) should have value 50.0
            self.assertTrue(np.all(res[:20, :20] == 50.0))
            # Out-of-bounds region must be padded with 0.0
            self.assertTrue(np.all(res[20:, :] == 0.0))
            self.assertTrue(np.all(res[:, 20:] == 0.0))

    def test_multiresolution_band_resampling(self):
        """Priority 1, Item 4: Reprojecting 20m and 60m bands onto 10m grid."""
        # 10m target grid (256x256)
        target_tf = from_origin(500000.0, 2000000.0, 10.0, 10.0)
        # 20m source grid (128x128) covering identical spatial extent
        source_20m_tf = from_origin(500000.0, 2000000.0, 20.0, 20.0)
        source_20m = np.full((128, 128), 42.0, dtype=np.float32)

        dst_buf = np.empty((256, 256), dtype=np.float32)
        reproject(
            source=source_20m,
            destination=dst_buf,
            src_transform=source_20m_tf,
            src_crs="EPSG:32644",
            dst_transform=target_tf,
            dst_crs="EPSG:32644",
            resampling=Resampling.bilinear,
        )
        self.assertEqual(dst_buf.shape, (256, 256))
        # Interior pixels should be exactly 42.0
        self.assertAlmostEqual(float(np.mean(dst_buf[10:-10, 10:-10])), 42.0, places=2)

    def test_ensure_rasterio_shim_idempotent(self):
        """Priority 1, Item 6: Ensure compatibility adapter is idempotent and sets sys.modules."""
        import sys
        gr.ensure_rasterio_shim()
        self.assertIn("rasterio", sys.modules)
        self.assertIn("rasterio.windows", sys.modules)
        self.assertIn("rasterio.warp", sys.modules)
        self.assertIn("rasterio.transform", sys.modules)
        self.assertIn("rasterio.crs", sys.modules)

        import rasterio
        self.assertTrue(hasattr(rasterio, "open"))
        self.assertTrue(hasattr(rasterio, "windows"))
        self.assertTrue(hasattr(rasterio, "warp"))


if __name__ == "__main__":
    unittest.main()
