import pytest
import numpy as np
from module1_ingestion.albedo_filter import filter_albedo_contamination
from module1_ingestion.config import (
    GLARE_Z_THRESHOLD, SHADOW_Z_THRESHOLD, ATMOSPHERIC_Z_THRESHOLD,
    SENTINEL_2_L2A_SPECTRAL_BANDS, FILTER_FILL_VALUE
)

def test_12_band_input():
    tensor = np.zeros((16, 16, 12), dtype=np.float32)
    res = filter_albedo_contamination(tensor)
    assert res.filtered.shape == (16, 16, 12)
    assert res.filtered.dtype == np.float32

def test_13_band_input():
    tensor = np.zeros((32, 32, 13), dtype=np.float32)
    res = filter_albedo_contamination(tensor)
    assert res.filtered.shape == (32, 32, 13)

def test_shape_preservation():
    tensor = np.zeros((256, 256, 13), dtype=np.float32)
    res = filter_albedo_contamination(tensor)
    assert res.filtered.shape == (256, 256, 13)

def test_finite_output():
    tensor = np.zeros((16, 16, 12), dtype=np.float32)
    tensor[0,0,:] = GLARE_Z_THRESHOLD + 1.0 # Will be filtered
    res = filter_albedo_contamination(tensor)
    assert np.isfinite(res.filtered).all()

def test_reflection_anomaly():
    tensor = np.zeros((16, 16, 12), dtype=np.float32)
    # Synthesize glare
    tensor[5:10, 5:10, :] = GLARE_Z_THRESHOLD + 2.0
    res = filter_albedo_contamination(tensor)
    assert res.statistics["glare_pixels"] == 25
    assert res.statistics["total_contaminated_pixels"] >= 25
    assert np.allclose(res.filtered[5:10, 5:10, :], FILTER_FILL_VALUE)
    # The rest should be 0.0
    assert res.filtered[0,0,0] == 0.0

def test_shadow_anomaly():
    tensor = np.zeros((16, 16, 12), dtype=np.float32)
    tensor[5:10, 5:10, :] = SHADOW_Z_THRESHOLD - 1.0
    res = filter_albedo_contamination(tensor)
    assert res.statistics["shadow_pixels"] == 25
    assert res.statistics["total_contaminated_pixels"] >= 25

def test_atmospheric_anomaly():
    tensor = np.zeros((16, 16, 12), dtype=np.float32)
    b01_idx = SENTINEL_2_L2A_SPECTRAL_BANDS.index("B01")
    b02_idx = SENTINEL_2_L2A_SPECTRAL_BANDS.index("B02")
    tensor[2:4, 2:4, b01_idx] = ATMOSPHERIC_Z_THRESHOLD + 1.0
    tensor[2:4, 2:4, b02_idx] = ATMOSPHERIC_Z_THRESHOLD + 1.0
    res = filter_albedo_contamination(tensor)
    assert res.statistics["atmospheric_pixels"] == 4

def test_explicit_valid_mask():
    tensor = np.zeros((16, 16, 12), dtype=np.float32)
    tensor[:,:,:] = GLARE_Z_THRESHOLD + 1.0 # Normally all glare
    
    valid_mask = np.zeros((16, 16), dtype=bool)
    valid_mask[0:5, 0:5] = True # Only this part is valid
    
    res = filter_albedo_contamination(tensor, valid_mask=valid_mask)
    assert res.statistics["total_valid_pixels"] == 25
    assert res.statistics["glare_pixels"] == 25 # Because invalid pixels shouldn't count
    assert res.statistics["total_contaminated_pixels"] == 25

def test_nan_inf_rejection():
    tensor = np.zeros((16, 16, 12), dtype=np.float32)
    tensor[0,0,0] = np.nan
    with pytest.raises(ValueError):
        filter_albedo_contamination(tensor)

def test_zero_valued_valid_pixels():
    tensor = np.zeros((16, 16, 12), dtype=np.float32)
    res = filter_albedo_contamination(tensor)
    # Zeros should not trigger any mask by themselves if thresholds don't align
    # (Since SHADOW_Z_THRESHOLD is < 0, 0 is not a shadow)
    assert res.statistics["total_contaminated_pixels"] == 0
    assert res.statistics["total_valid_pixels"] == 256

def test_combined_contamination_no_double_counting():
    tensor = np.zeros((16, 16, 12), dtype=np.float32)
    # Pixel (0,0) is both glare and atmospheric (high B01, B02, B03, B08, B11)
    tensor[0, 0, :] = GLARE_Z_THRESHOLD + 1.0
    res = filter_albedo_contamination(tensor)
    assert res.statistics["glare_pixels"] == 1
    assert res.statistics["atmospheric_pixels"] == 1
    # Should only count as 1 contaminated pixel
    assert res.statistics["total_contaminated_pixels"] == 1

def test_false_positive_safety():
    # Vegetated pixel: high NIR, low Red
    tensor = np.zeros((16, 16, 12), dtype=np.float32)
    b04_idx = SENTINEL_2_L2A_SPECTRAL_BANDS.index("B04")
    b08_idx = SENTINEL_2_L2A_SPECTRAL_BANDS.index("B08")
    tensor[:,:,b08_idx] = GLARE_Z_THRESHOLD + 1.0 # High NIR
    tensor[:,:,b04_idx] = SHADOW_Z_THRESHOLD - 1.0 # Low Red
    res = filter_albedo_contamination(tensor)
    
    # Should not be glare (requires B03, B08, B11 all high)
    # Should not be shadow (requires B04 and B08 both low)
    # Should not be atmospheric (requires B01, B02 high)
    assert res.statistics["total_contaminated_pixels"] == 0

def test_deterministic_output():
    tensor = np.random.randn(32, 32, 12).astype(np.float32)
    res1 = filter_albedo_contamination(tensor)
    res2 = filter_albedo_contamination(tensor)
    assert np.allclose(res1.filtered, res2.filtered)
    assert np.array_equal(res1.contamination_mask, res2.contamination_mask)

def test_input_scale_handling():
    # Reflectance input where values are 5000 (above GLARE_REFLECTANCE_THRESHOLD = 4000)
    tensor = np.full((16, 16, 12), 5000.0, dtype=np.float32)
    res = filter_albedo_contamination(tensor, input_scale="reflectance")
    assert res.statistics["glare_pixels"] == 256
    
    # Same tensor, but interpreted as Z-scores (5000 is way above 3.0)
    res2 = filter_albedo_contamination(tensor, input_scale="normalized")
    assert res2.statistics["glare_pixels"] == 256
