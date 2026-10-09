"""
VOXEL Environment and GPU Acceleration Verification Script
Verifies all dependencies, CUDA availability, GPU device info, and VRAM memory.
"""
import sys
import os
import platform

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def check_package(pkg_name, import_alias=None):
    alias = import_alias if import_alias else pkg_name
    try:
        mod = __import__(alias)
        version = getattr(mod, "__version__", "Installed")
        return True, version, "OK"
    except ImportError as e:
        return False, "Not Installed", str(e)
    except Exception as e:
        return False, "Error", str(e)

def main():
    print("=" * 70)
    print(" [VOXEL] MODULE 3 ENVIRONMENT & GPU ACCELERATION VERIFICATION")
    print("=" * 70)
    print(f"OS: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print("-" * 70)

    # Core required libraries to check
    packages_to_check = [
        ("PyTorch", "torch"),
        ("Torchvision", "torchvision"),
        ("Torchaudio", "torchaudio"),
        ("TIMM", "timm"),
        ("FAISS", "faiss"),
        ("SHAP", "shap"),
        ("PyWavelets", "pywt"),
        ("Rasterio", "rasterio"),
        ("Scikit-Learn", "sklearn"),
        ("GeoPandas", "geopandas"),
        ("OpenCV", "cv2"),
        ("Mealpy", "mealpy"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
    ]

    print(f"{'Package':<18} | {'Status':<10} | {'Version':<18} | {'Notes'}")
    print("-" * 70)

    all_passed = True
    for name, module_name in packages_to_check:
        ok, ver, note = check_package(name, module_name)
        status_str = "[PASS]" if ok else "[FAIL]"
        if not ok:
            all_passed = False
        print(f"{name:<18} | {status_str:<10} | {str(ver):<18} | {note}")

    print("=" * 70)
    print(" GPU & CUDA ACCELERATION STATUS")
    print("=" * 70)

    torch_ok, _, _ = check_package("torch", "torch")
    if torch_ok:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available:           {'[YES]' if cuda_avail else '[NO]'}")
        if cuda_avail:
            gpu_count = torch.cuda.device_count()
            current_dev = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_dev)
            vram_bytes = torch.cuda.get_device_properties(current_dev).total_memory
            vram_gb = vram_bytes / (1024 ** 3)
            cuda_ver = torch.version.cuda
            cudnn_ver = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"

            print(f"GPU Device Count:         {gpu_count}")
            print(f"Active GPU Name:          {gpu_name}")
            print(f"Total Dedicated VRAM:     {vram_gb:.2f} GB ({vram_bytes / (1024**2):.0f} MiB)")
            print(f"PyTorch CUDA Version:     {cuda_ver}")
            print(f"cuDNN Version:            {cudnn_ver}")
            print(f"Device Capability:        {torch.cuda.get_device_capability(current_dev)}")
            
            # Simple tensor allocation test on GPU
            try:
                test_tensor = torch.randn((1000, 1000), device='cuda')
                _ = torch.matmul(test_tensor, test_tensor)
                print("GPU Tensor Ops Test:      [PASS] (CUDA Matrix Multiplication OK)")
            except Exception as e:
                print(f"GPU Tensor Ops Test:      [FAIL] ({e})")
                all_passed = False
        else:
            print("[WARN] PyTorch is installed without CUDA support or GPU is unavailable.")
            all_passed = False
    else:
        print("[FAIL] Cannot verify GPU acceleration: 'torch' is not installed.")
        all_passed = False

    print("=" * 70)
    if all_passed:
        print(" [SUMMARY] All Module 3 dependencies and GPU readiness VERIFIED!")
    else:
        print(" [SUMMARY] Some packages require installation.")
    print("=" * 70)

if __name__ == "__main__":
    main()
