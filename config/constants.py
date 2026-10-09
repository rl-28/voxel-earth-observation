"""
VOXEL Project Shared Constants & Specifications.
Single source of truth for spectral bands, land cover classes, and tensor dimensions.
"""

# Spectral Band Definitions (Sentinel-2 L2A standard ordering)
SENTINEL2_BAND_KEYS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]

SENTINEL2_BANDS = [
    "B01 (Coastal)",
    "B02 (Blue)",
    "B03 (Green)",
    "B04 (Red)",
    "B05 (VRE 1)",
    "B06 (VRE 2)",
    "B07 (VRE 3)",
    "B08 (NIR)",
    "B8A (Narrow NIR)",
    "B09 (Water Vapour)",
    "B10 (Cirrus)",
    "B11 (SWIR 1)",
    "B12 (SWIR 2)",
]

# Canonical Land Cover Classes (EuroSAT-derived 10 classes)
LAND_COVER_CLASSES = [
    "Annual Crop",
    "Forest",
    "Herbaceous Vegetation",
    "Highway / Infrastructure",
    "Industrial Area",
    "Pasture",
    "Permanent Crop",
    "Residential Area",
    "River / Inland Water",
    "Sea / Marine Water",
]

# Tensor & Spatial Dimensions
NUM_BANDS: int = 13
NUM_CLASSES: int = len(LAND_COVER_CLASSES)  # 10
TILE_SIZE: int = 256  # 256x256 spatial dimension (H=256, W=256)
DEFAULT_SCALE_DIVISOR: float = 10000.0  # Sentinel-2 L2A DN scaling factor

# Representation Vector Dimensions
CNN_FEATURE_DIM: int = 1024
VIT_FEATURE_DIM: int = 192
EMBEDDING_DIM: int = CNN_FEATURE_DIM + VIT_FEATURE_DIM  # 1216

# Default Fallback Band Mask (B02, B03, B04, B08, B11, B12)
DEFAULT_SELECTED_BAND_INDICES = [1, 2, 3, 7, 11, 12]
