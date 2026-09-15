"""
Configuration settings for Module 1.

Centralizes paths, parameters, and constants.

TODO:
- load paths from environment
- define processing constants
"""

import numpy as np

# Important processing constants conceptually defined
TILE_SIZE = 256
NUM_BANDS = 13
OUTPUT_DTYPE = np.float32

# Paths will be relative or configured dynamically, not hardcoded.
