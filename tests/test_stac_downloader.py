"""
Tests for STAC downloading functionality.
Requires no live internet connection (uses mocking).
"""

import pytest
import os
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
