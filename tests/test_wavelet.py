import pytest
import numpy as np
import time
from module1_ingestion.wavelet_3d import apply_3d_dwt

def test_input_validation():
    # Not a numpy array
    with pytest.raises(ValueError):
        apply_3d_dwt([1, 2, 3])
        
    # Invalid dimensions (2D instead of 3D)
    with pytest.raises(ValueError):
        apply_3d_dwt(np.zeros((256, 256)))
        
    # Invalid dimensions (4D)
    with pytest.raises(ValueError):
        apply_3d_dwt(np.zeros((256, 256, 13, 2)))
        
    # Invalid shape (0 dimension)
    with pytest.raises(ValueError):
        apply_3d_dwt(np.zeros((256, 0, 13)))
        
    # Non-numeric dtype
    with pytest.raises(ValueError):
        apply_3d_dwt(np.full((256, 256, 13), 'a'))
        
    # NaN check
    tensor = np.zeros((256, 256, 13), dtype=np.float32)
    tensor[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        apply_3d_dwt(tensor)
        
    # Inf check
    tensor[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        apply_3d_dwt(tensor)

def test_12_band_input():
    tensor = np.random.randn(256, 256, 12).astype(np.float32)
    res = apply_3d_dwt(tensor)
    
    assert res.input_shape == (256, 256, 12)
    assert res.reconstruct_full().shape == (256, 256, 12)

def test_13_band_input():
    tensor = np.random.randn(256, 256, 13).astype(np.float32)
    res = apply_3d_dwt(tensor)
    
    assert res.input_shape == (256, 256, 13)
    assert res.reconstruct_full().shape == (256, 256, 13)

def test_constant_input():
    tensor = np.full((256, 256, 13), 5.0, dtype=np.float32)
    res = apply_3d_dwt(tensor, wavelet='db1', level=1)
    
    # Detail coefficients should be near zero
    for detail_dict in res.coefficients[1:]:
        for k, v in detail_dict.items():
            assert np.allclose(v, 0, atol=1e-5)
            
    # Reconstruction should exactly match the constant tensor
    recon = res.reconstruct_full()
    assert np.allclose(recon, tensor, atol=1e-5)

def test_zero_input():
    tensor = np.zeros((256, 256, 13), dtype=np.float32)
    res = apply_3d_dwt(tensor, wavelet='db1', level=1)
    
    # All coefficients should be finite and approximately zero
    assert np.isfinite(res.approximation).all()
    assert np.allclose(res.approximation, 0)
    
    for detail_dict in res.coefficients[1:]:
        for k, v in detail_dict.items():
            assert np.isfinite(v).all()
            assert np.allclose(v, 0)
            
    recon = res.reconstruct_full()
    assert np.allclose(recon, 0)

def test_structured_input():
    tensor = np.zeros((256, 256, 13), dtype=np.float32)
    # Create an edge horizontally at an odd index so Haar pair (0,1) isn't uniform
    tensor[:, 127:, :] = 1.0
    
    res = apply_3d_dwt(tensor, wavelet='db1', level=1)
    
    # Check that detail coefficients are not universally zero
    details_zero = True
    for detail_dict in res.coefficients[1:]:
        for k, v in detail_dict.items():
            if not np.allclose(v, 0):
                details_zero = False
                break
    assert not details_zero

def test_finite_coefficients():
    tensor = np.random.randn(256, 256, 13).astype(np.float32)
    res = apply_3d_dwt(tensor)
    
    assert np.isfinite(res.approximation).all()
    for detail_dict in res.coefficients[1:]:
        for k, v in detail_dict.items():
            assert np.isfinite(v).all()

def test_reconstruction_error():
    tensor = np.random.randn(256, 256, 13).astype(np.float32)
    res = apply_3d_dwt(tensor, wavelet='db1', level=1)
    
    recon = res.reconstruct_full()
    
    mae = np.mean(np.abs(tensor - recon))
    mse = np.mean((tensor - recon)**2)
    max_err = np.max(np.abs(tensor - recon))
    
    # Errors should be close to float32 numerical precision
    assert mae < 1e-5
    assert mse < 1e-10
    assert max_err < 1e-4

def test_spatial_resolution_preservation():
    tensor = np.random.randn(256, 256, 13).astype(np.float32)
    res = apply_3d_dwt(tensor)
    
    # Both full and approx reconstructions must preserve exactly (256, 256)
    approx_recon = res.reconstruct_approximation()
    assert approx_recon.shape == (256, 256, 13)
    
    full_recon = res.reconstruct_full()
    assert full_recon.shape == (256, 256, 13)

def test_deterministic_output():
    tensor = np.random.randn(256, 256, 13).astype(np.float32)
    
    res1 = apply_3d_dwt(tensor, wavelet='db1', level=1, mode='symmetric')
    res2 = apply_3d_dwt(tensor, wavelet='db1', level=1, mode='symmetric')
    
    assert np.allclose(res1.approximation, res2.approximation)
    for i, detail_dict in enumerate(res1.coefficients[1:], 1):
        for k in detail_dict:
            assert np.allclose(detail_dict[k], res2.coefficients[i][k])

def test_configurable_wavelet_and_mode():
    tensor = np.random.randn(256, 256, 13).astype(np.float32)
    
    res = apply_3d_dwt(tensor, wavelet='haar', level=1, mode='reflect')
    assert res.wavelet == 'haar'
    assert res.mode == 'reflect'
    
    recon = res.reconstruct_full()
    assert recon.shape == (256, 256, 13)
    assert np.allclose(tensor, recon, atol=1e-5)

def test_level_validation():
    tensor = np.random.randn(256, 256, 13).astype(np.float32)
    
    # Level 2 should work
    res = apply_3d_dwt(tensor, level=2)
    assert res.level == 2
    assert len(res.coefficients) == 3 # [cAn, details_2, details_1]

def test_performance():
    # This is a soft performance test just to ensure it doesn't hang.
    # The benchmark fixture is used if pytest-benchmark is installed, 
    # but we will just measure time normally.
    tensor = np.random.randn(256, 256, 13).astype(np.float32)
    
    start_time = time.time()
    res = apply_3d_dwt(tensor)
    exec_time = time.time() - start_time
    
    assert exec_time < 5.0 # Should be very fast, well under 5 seconds.
