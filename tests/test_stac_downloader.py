"""
Tests for STAC downloading functionality.
Requires no live internet connection (uses mocking).
"""

import pytest
import os
import numpy as np
from unittest.mock import patch, MagicMock
from module1_ingestion.stac_downloader import CopernicusSTACProvider, BhuNidhiProvider
from module1_ingestion.config import SENTINEL_2_L2A_COLLECTION

@pytest.fixture
def valid_aoi():
    return {
        "type": "Polygon",
        "coordinates": [[[10.0, 10.0], [10.0, 20.0], [20.0, 20.0], [20.0, 10.0], [10.0, 10.0]]]
    }

def test_config_loaded():
    provider = CopernicusSTACProvider()
    assert provider.endpoint_url is not None
    assert SENTINEL_2_L2A_COLLECTION == "sentinel-2-l2a"

def test_l2a_band_mapping():
    from module1_ingestion.config import SENTINEL_2_L2A_SPECTRAL_BANDS, NUM_BANDS
    
    # 1. B10 is NOT included
    assert "B10" not in SENTINEL_2_L2A_SPECTRAL_BANDS
    
    # 2. Exactly 12 L2A spectral bands
    assert len(SENTINEL_2_L2A_SPECTRAL_BANDS) == 12
    
    # 3. B8A is included
    assert "B8A" in SENTINEL_2_L2A_SPECTRAL_BANDS
    
    # 4. B11 and B12 are included
    assert "B11" in SENTINEL_2_L2A_SPECTRAL_BANDS
    assert "B12" in SENTINEL_2_L2A_SPECTRAL_BANDS
    
    # 5. Project NUM_BANDS remains 13
    assert NUM_BANDS == 13
    
    # 6. Downloader distinguishes SOURCE and TARGET (by having 12 source bands != 13 target)
    assert len(SENTINEL_2_L2A_SPECTRAL_BANDS) != NUM_BANDS

def test_aoi_validation_fails_on_invalid():
    provider = CopernicusSTACProvider()
    with pytest.raises(ValueError, match="AOI must be a valid GeoJSON"):
        provider.search_sentinel2_l2a(aoi="invalid", start_date="2023-01-01", end_date="2023-01-31")

def test_date_validation_fails_on_invalid(valid_aoi):
    provider = CopernicusSTACProvider()
    with pytest.raises(ValueError, match="start_date cannot be after end_date"):
        provider.search_sentinel2_l2a(aoi=valid_aoi, start_date="2023-02-01", end_date="2023-01-31")
        
    with pytest.raises(ValueError, match="Dates must be valid ISO format"):
        provider.search_sentinel2_l2a(aoi=valid_aoi, start_date="not-a-date", end_date="2023-01-31")

def test_cloud_cover_validation_fails_on_invalid(valid_aoi):
    provider = CopernicusSTACProvider()
    with pytest.raises(ValueError, match="Cloud cover threshold must be between"):
        provider.search_sentinel2_l2a(aoi=valid_aoi, start_date="2023-01-01", end_date="2023-01-31", cloud_cover=150)

@patch("module1_ingestion.stac_downloader.Client")
def test_search_results_parsed(mock_client_class, valid_aoi):
    # Setup mock
    mock_client = MagicMock()
    mock_client_class.open.return_value = mock_client
    
    mock_search = MagicMock()
    mock_item = MagicMock()
    mock_item.id = "test_item_id"
    mock_search.items.return_value = [mock_item]
    mock_client.search.return_value = mock_search
    
    provider = CopernicusSTACProvider()
    results = provider.search_sentinel2_l2a(valid_aoi, "2023-01-01", "2023-01-31")
    
    assert len(results) == 1
    assert results[0].id == "test_item_id"
    
    # Verify search parameters
    mock_client.search.assert_called_once_with(
        collections=["sentinel-2-l2a"],
        intersects=valid_aoi,
        datetime="2023-01-01/2023-01-31",
        query={"eo:cloud_cover": {"lte": 20.0}},
        max_items=10
    )

def test_asset_discovery():
    provider = CopernicusSTACProvider()
    mock_item = MagicMock()
    mock_item.assets = {"B04": "mock_asset"}
    assets = provider.get_item_assets(mock_item)
    assert "B04" in assets

@patch("module1_ingestion.stac_downloader.requests.get")
def test_download_success(mock_get, tmp_path):
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
    mock_get.return_value = mock_response
    
    provider = CopernicusSTACProvider()
    dest_path = tmp_path / "test.tif"
    
    result_path = provider.download_asset("http://mock-url.com/asset.tif", str(dest_path))
    
    assert result_path == str(dest_path)
    assert os.path.exists(dest_path)
    with open(dest_path, "rb") as f:
        assert f.read() == b"chunk1chunk2"

@patch("module1_ingestion.stac_downloader.requests.get")
def test_download_failure_cleanup(mock_get, tmp_path):
    mock_get.side_effect = Exception("Network error")
    
    provider = CopernicusSTACProvider()
    dest_path = tmp_path / "test_fail.tif"
    
    # Create dummy file to simulate partial download before failure
    with open(dest_path, "wb") as f:
        f.write(b"partial")
        
    with pytest.raises(RuntimeError, match="Failed to download"):
        provider.download_asset("http://mock-url.com/asset.tif", str(dest_path), overwrite=True)
        
    # Ensure it's cleaned up
    assert not os.path.exists(dest_path)

def test_bhunidhi_placeholder():
    provider = BhuNidhiProvider()
    with pytest.raises(NotImplementedError):
        provider.search_sentinel2_l2a()


def test_earth_search_band_mapping():
    from module1_ingestion.stac_downloader import EarthSearchSTACProvider
    from module1_ingestion.config import EARTH_SEARCH_BAND_TO_ASSET, SENTINEL_2_L2A_SPECTRAL_BANDS

    provider = EarthSearchSTACProvider()
    assert provider.get_band_asset_key("B01") == "coastal"
    assert provider.get_band_asset_key("B02") == "blue"
    assert provider.get_band_asset_key("B03") == "green"
    assert provider.get_band_asset_key("B04") == "red"
    assert provider.get_band_asset_key("B08") == "nir"
    assert provider.get_band_asset_key("B8A") == "nir08"
    assert provider.get_band_asset_key("B11") == "swir16"
    assert provider.get_band_asset_key("B12") == "swir22"

    for b in SENTINEL_2_L2A_SPECTRAL_BANDS:
        assert b in EARTH_SEARCH_BAND_TO_ASSET
        assert provider.get_band_asset_key(b) == EARTH_SEARCH_BAND_TO_ASSET[b]


def test_stac_provider_factory_and_env():
    from module1_ingestion.stac_downloader import (
        get_stac_provider,
        STACDownloader,
        CopernicusSTACProvider,
        EarthSearchSTACProvider
    )

    # Default is Copernicus
    default_p = get_stac_provider()
    assert isinstance(default_p, CopernicusSTACProvider)

    # Explicit constructor argument
    es_p = get_stac_provider("earth_search")
    assert isinstance(es_p, EarthSearchSTACProvider)

    # Env variable STAC_PROVIDER
    with patch.dict(os.environ, {"STAC_PROVIDER": "earth_search"}):
        env_p = get_stac_provider()
        assert isinstance(env_p, EarthSearchSTACProvider)

    # High-level STACDownloader
    downloader_default = STACDownloader()
    assert isinstance(downloader_default.provider, CopernicusSTACProvider)

    downloader_es = STACDownloader(provider="earth_search")
    assert isinstance(downloader_es.provider, EarthSearchSTACProvider)

    with pytest.raises(ValueError, match="Unknown STAC provider"):
        get_stac_provider("invalid_provider")


@patch("module1_ingestion.stac_downloader.Client")
def test_earth_search_search_mocked(mock_client_class, valid_aoi):
    from module1_ingestion.stac_downloader import EarthSearchSTACProvider
    mock_client = MagicMock()
    mock_client_class.open.return_value = mock_client
    mock_search = MagicMock()
    mock_item = MagicMock()
    mock_item.id = "S2_test_es"
    mock_search.items.return_value = [mock_item]
    mock_client.search.return_value = mock_search

    provider = EarthSearchSTACProvider()
    results = provider.search_sentinel2_l2a(valid_aoi, "2023-05-01", "2023-05-31", cloud_cover=10.0)

    assert len(results) == 1
    assert results[0].id == "S2_test_es"
    mock_client.search.assert_called_once_with(
        collections=["sentinel-2-l2a"],
        intersects=valid_aoi,
        datetime="2023-05-01/2023-05-31",
        query={"eo:cloud_cover": {"lte": 10.0}},
        max_items=10
    )


@patch("module1_ingestion.stac_downloader.requests.get")
def test_earth_search_download_bands_mocked(mock_get, tmp_path):
    from module1_ingestion.stac_downloader import EarthSearchSTACProvider
    from module1_ingestion.config import EARTH_SEARCH_BAND_TO_ASSET

    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [b"raster_data"]
    mock_get.return_value = mock_resp

    provider = EarthSearchSTACProvider()

    # Construct mock item with Earth Search asset keys
    mock_item = MagicMock()
    mock_item.id = "test_item_es"
    mock_assets = {}
    for b_name, asset_name in EARTH_SEARCH_BAND_TO_ASSET.items():
        asset_mock = MagicMock()
        asset_mock.href = f"https://mock-s3.aws.com/{asset_name}.tif"
        mock_assets[asset_name] = asset_mock
    mock_item.assets = mock_assets

    target_dir = tmp_path / "bands"
    band_paths = provider.download_bands(mock_item, str(target_dir))

    assert len(band_paths) == 12
    for b_name in EARTH_SEARCH_BAND_TO_ASSET.keys():
        assert b_name in band_paths
        dest_file = target_dir / f"{b_name}.tif"
        assert os.path.exists(dest_file)
        assert band_paths[b_name] == str(dest_file)


# -----------------------------------------------------------------------------
# Fix 3: Dynamic Sentinel-2 Search Window & Fallback Tests
# -----------------------------------------------------------------------------
def test_fix3_dynamic_search_lowest_cloud_cover_selection():
    """Verify that multiple candidate STAC scenes are sorted by cloud cover ascending, picking lowest."""
    from voxel_pipeline import VoxelPipeline

    pipeline = VoxelPipeline(default_mode="offline", verbose=False)

    item1 = MagicMock()
    item1.id = "S2_high_cloud"
    item1.properties = {"eo:cloud_cover": 18.5}
    item1.datetime = None
    item1.collection_id = "sentinel-2-l2a"
    item1.assets = {pipeline.earth_search_stac.get_band_asset_key(b): MagicMock() for b in ["B02", "B03", "B04", "B08"]}

    item2 = MagicMock()
    item2.id = "S2_lowest_cloud"
    item2.properties = {"eo:cloud_cover": 1.2}
    item2.datetime = None
    item2.collection_id = "sentinel-2-l2a"
    item2.assets = {pipeline.earth_search_stac.get_band_asset_key(b): MagicMock() for b in ["B02", "B03", "B04", "B08"]}

    with patch.object(pipeline.earth_search_stac, "search_sentinel2_l2a", return_value=[item1, item2]), \
         patch.object(pipeline.earth_search_stac, "download_bands", return_value={"B02": "mock"}), \
         patch("voxel_pipeline.Module1Pipeline.for_module3") as mock_mod1:

        mock_tile_res = MagicMock()
        mock_tile_res.processed_tensor = np.zeros((13, 256, 256), dtype=np.float32)
        mock_tile_res.tile_id = "tile_001"
        mock_tile_res.diagnostics = {}
        mock_instance = MagicMock()
        mock_instance.process_scene.return_value = [mock_tile_res]
        mock_mod1.return_value = mock_instance

        tensor, meta, errors = pipeline.acquire_and_process_real_satellite_data(
            bbox=[72.8, 18.9, 72.9, 19.0],
            start_date="2024-03-01",
            end_date="2024-04-30",
            cloud_cover=20.0,
        )

        assert errors == []
        assert meta["stac_item_id"] == "S2_lowest_cloud"
        assert meta["cloud_cover"] == 1.2
        assert meta["search_start_date"] == "2024-03-01"
        assert meta["search_end_date"] == "2024-04-30"


def test_fix3_controlled_fallback_relaxation():
    """Verify that when initial search has 0 scenes, controlled relaxation runs and warns user."""
    from voxel_pipeline import VoxelPipeline

    pipeline = VoxelPipeline(default_mode="offline", verbose=False)

    item_relaxed = MagicMock()
    item_relaxed.id = "S2_relaxed_scene"
    item_relaxed.properties = {"eo:cloud_cover": 32.0}
    item_relaxed.datetime = None
    item_relaxed.collection_id = "sentinel-2-l2a"
    item_relaxed.assets = {pipeline.earth_search_stac.get_band_asset_key(b): MagicMock() for b in ["B02", "B03", "B04", "B08"]}

    # First call returns [], second call (cloud_cover=40.0) returns [item_relaxed]
    search_mock = MagicMock(side_effect=[[], [item_relaxed]])

    with patch.object(pipeline.earth_search_stac, "search_sentinel2_l2a", search_mock), \
         patch.object(pipeline.earth_search_stac, "download_bands", return_value={"B02": "mock"}), \
         patch("voxel_pipeline.Module1Pipeline.for_module3") as mock_mod1:

        mock_tile_res = MagicMock()
        mock_tile_res.processed_tensor = np.zeros((13, 256, 256), dtype=np.float32)
        mock_tile_res.tile_id = "tile_001"
        mock_tile_res.diagnostics = {}
        mock_instance = MagicMock()
        mock_instance.process_scene.return_value = [mock_tile_res]
        mock_mod1.return_value = mock_instance

        tensor, meta, errors = pipeline.acquire_and_process_real_satellite_data(
            bbox=[72.8, 18.9, 72.9, 19.0],
            start_date="2024-01-01",
            end_date="2024-02-01",
            cloud_cover=10.0,
            allow_fallback=True,
        )

        assert errors == []
        assert meta["fallback_applied"] is True
        assert len(meta["warnings"]) >= 1
        assert "Controlled fallback relaxed" in meta["warnings"][0]
        assert meta["stac_item_id"] == "S2_relaxed_scene"


def test_fix3_fallback_disabled_fails_cleanly():
    """Verify that when allow_fallback is False, 0 items immediately fails without relaxation."""
    from voxel_pipeline import VoxelPipeline

    pipeline = VoxelPipeline(default_mode="offline", verbose=False)

    with patch.object(pipeline.earth_search_stac, "search_sentinel2_l2a", return_value=[]):
        tensor, meta, errors = pipeline.acquire_and_process_real_satellite_data(
            bbox=[72.8, 18.9, 72.9, 19.0],
            start_date="2024-01-01",
            end_date="2024-02-01",
            cloud_cover=10.0,
            allow_fallback=False,
        )

        assert tensor is None
        assert len(errors) == 1
        assert "Fallback relaxation disabled" in errors[0]


