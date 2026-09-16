import numpy as np
import pywt
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

from .config import WAVELET_NAME, WAVELET_LEVEL, WAVELET_MODE

@dataclass
class Wavelet3DResult:
    """
    Container for the 3D Discrete Wavelet Transform result.
    
    Attributes:
        input_shape: Shape of the input tensor (H, W, C).
        wavelet: Name of the wavelet used.
        level: Decomposition level.
        mode: Boundary extension mode.
        coefficients: List of coefficients from pywt.wavedecn.
                      Format: [cAn, {details_level_n}, ..., {details_level_1}]
                      Where cAn is the approximation array.
    """
    input_shape: Tuple[int, int, int]
    wavelet: str
    level: int
    mode: str
    coefficients: list

    @property
    def approximation(self) -> np.ndarray:
        """Returns the low-frequency approximation coefficients."""
        return self.coefficients[0]

    def reconstruct_approximation(self) -> np.ndarray:
        """
        Reconstructs ONLY the approximation back to the original
        spatial-spectral resolution (input_shape), preserving spatial scale.
        All detail coefficients are zeroed out before inverse transform.
        """
        reconstructed_coeffs = [self.coefficients[0]]
        for detail_dict in self.coefficients[1:]:
            zeroed_dict = {k: np.zeros_like(v) for k, v in detail_dict.items()}
            reconstructed_coeffs.append(zeroed_dict)
            
        reconstructed = pywt.waverecn(
            reconstructed_coeffs, 
            wavelet=self.wavelet, 
            mode=self.mode,
            axes=(0, 1, 2)
        )
        
        # Crop in case of slightly larger inverse due to odd dimensions
        h, w, c = self.input_shape
        return reconstructed[:h, :w, :c]

    def reconstruct_full(self) -> np.ndarray:
        """
        Reconstructs the full signal using all approximation and detail coefficients.
        Provides a mathematically verifiable spatial-preservation baseline.
        """
        reconstructed = pywt.waverecn(
            self.coefficients, 
            wavelet=self.wavelet, 
            mode=self.mode,
            axes=(0, 1, 2)
        )
        
        h, w, c = self.input_shape
        return reconstructed[:h, :w, :c]


def apply_3d_dwt(
    tensor: np.ndarray,
    wavelet: str = WAVELET_NAME,
    level: int = WAVELET_LEVEL,
    mode: str = WAVELET_MODE
) -> Wavelet3DResult:
    """
    Applies a 3D Discrete Wavelet Transform over the (Height, Width, Spectral) axes.
    
    Args:
        tensor: A 3D NumPy array of shape (H, W, C).
        wavelet: PyWavelets wavelet name (e.g., 'db1').
        level: Decomposition level.
        mode: Boundary handling mode (default 'symmetric').
        
    Returns:
        Wavelet3DResult containing coefficients and reconstruction methods.
        
    Raises:
        ValueError: If input is not 3D, has non-positive dimensions, or contains invalid data.
    """
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a NumPy ndarray.")
        
    if tensor.ndim != 3:
        raise ValueError(f"Input must be exactly 3-dimensional (H, W, C). Got shape {tensor.shape}.")
        
    h, w, c = tensor.shape
    if h <= 0 or w <= 0 or c <= 0:
        raise ValueError(f"Input dimensions must be positive. Got shape {tensor.shape}.")
        
    if not np.issubdtype(tensor.dtype, np.number):
        raise ValueError(f"Input must be a numeric dtype. Got {tensor.dtype}.")
        
    if not np.isfinite(tensor).all():
        raise ValueError("Input contains non-finite values (NaN or Inf). Cannot perform DWT.")

    # pywt handles precision based on input; we ensure float32 or float64.
    # Apply 3D DWT across all three axes (0: Height, 1: Width, 2: Spectral)
    coeffs = pywt.wavedecn(
        tensor,
        wavelet=wavelet,
        mode=mode,
        level=level,
        axes=(0, 1, 2)
    )

    return Wavelet3DResult(
        input_shape=(h, w, c),
        wavelet=wavelet,
        level=level,
        mode=mode,
        coefficients=coeffs
    )
