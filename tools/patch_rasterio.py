import os
import shutil

backup_path = '.venv/Lib/site-packages/rasterio/__init__.py.bak'
target_path = '.venv/Lib/site-packages/rasterio/__init__.py'

if not os.path.exists(backup_path):
    shutil.copyfile(target_path, backup_path)

with open(backup_path, 'r', encoding='utf-8') as f:
    orig = f.read()

patch_code = '''"""Rasterio"""

# start delvewheel patch
def _delvewheel_patch_1_11_2():
    import os
    if os.path.isdir(libs_dir := os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'rasterio.libs'))):
        os.add_dll_directory(libs_dir)

_delvewheel_patch_1_11_2()
del _delvewheel_patch_1_11_2
# end delvewheel patch

import os
import platform
import sys

_USE_FALLBACK = False
try:
    from rasterio._base import DatasetBase
except (ImportError, OSError):
    _USE_FALLBACK = True

if _USE_FALLBACK:
    try:
        from utils import geospatial_raster as _gr
    except ImportError:
        ws_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        if ws_dir not in sys.path:
            sys.path.insert(0, ws_dir)
        from utils import geospatial_raster as _gr

    CRS = _gr.CRS
    Affine = _gr.Affine
    from_origin = _gr.from_origin
    guard_transform = _gr.guard_transform
    Window = _gr.Window
    windows = _gr.windows
    warp = _gr.warp
    transform = _gr.transform
    reproject = _gr.reproject
    Resampling = _gr.Resampling
    transform_bounds = _gr.transform_bounds
    calculate_default_transform = _gr.calculate_default_transform
    Env = _gr.Env
    band = _gr.band
    open = _gr.open
    DatasetReader = _gr.DatasetReader
    DatasetWriter = _gr.DatasetWriter

    float32 = _gr.float32
    float64 = _gr.float64
    uint16 = _gr.uint16
    uint8 = _gr.uint8
    int16 = _gr.int16
    int32 = _gr.int32
    uint32 = _gr.uint32
    bool_ = _gr.bool_
    check_dtype = _gr.check_dtype

    import types
    mod_windows = types.ModuleType('rasterio.windows')
    mod_windows.Window = _gr.Window
    mod_windows.transform = _gr.windows.transform
    mod_windows.bounds = _gr.windows.bounds
    mod_windows.from_bounds = _gr.windows.from_bounds
    sys.modules['rasterio.windows'] = mod_windows

    mod_warp = types.ModuleType('rasterio.warp')
    mod_warp.Resampling = _gr.Resampling
    mod_warp.transform_bounds = _gr.transform_bounds
    mod_warp.calculate_default_transform = _gr.calculate_default_transform
    mod_warp.reproject = _gr.reproject
    sys.modules['rasterio.warp'] = mod_warp

    mod_transform = types.ModuleType('rasterio.transform')
    mod_transform.Affine = _gr.Affine
    mod_transform.from_origin = _gr.from_origin
    mod_transform.guard_transform = _gr.guard_transform
    sys.modules['rasterio.transform'] = mod_transform

    mod_crs = types.ModuleType('rasterio.crs')
    mod_crs.CRS = _gr.CRS
    sys.modules['rasterio.crs'] = mod_crs

    mod_dtypes = types.ModuleType('rasterio.dtypes')
    mod_dtypes.float32 = _gr.float32
    mod_dtypes.float64 = _gr.float64
    mod_dtypes.uint16 = _gr.uint16
    mod_dtypes.uint8 = _gr.uint8
    mod_dtypes.int16 = _gr.int16
    mod_dtypes.int32 = _gr.int32
    mod_dtypes.uint32 = _gr.uint32
    mod_dtypes.bool_ = _gr.bool_
    mod_dtypes.check_dtype = _gr.check_dtype
    sys.modules['rasterio.dtypes'] = mod_dtypes

    mod_env = types.ModuleType('rasterio.env')
    mod_env.Env = _gr.Env
    sys.modules['rasterio.env'] = mod_env

    mod_io = types.ModuleType('rasterio.io')
    mod_io.DatasetReader = _gr.DatasetReader
    mod_io.DatasetWriter = _gr.DatasetWriter
    sys.modules['rasterio.io'] = mod_io

    __all__ = ['band', 'open', 'pad', 'Band', 'Env', 'CRS', 'Affine', 'from_origin', 'Window', 'windows', 'warp', 'transform', 'reproject', 'Resampling']
    __version__ = '1.4.4'
else:
'''

orig_lines = orig.splitlines()
body_lines = orig_lines[14:]
indented_body = '\n'.join('    ' + l for l in body_lines)
full_content = patch_code + indented_body + '\n'

with open(target_path, 'w', encoding='utf-8') as f:
    f.write(full_content)

print('Successfully patched rasterio/__init__.py!')
