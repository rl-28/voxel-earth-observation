# VOXEL Module 3: GPU Memory, Speed & Scaling Profile

**Execution Date:** 2026-10-10 10:20:00<br>
**Compute Device:** `NVIDIA GeForce RTX 3050 4GB Laptop GPU` (4.00 GB Total VRAM)<br>
**Software Stack:** PyTorch `2.5.1+cu121` | CUDA `12.1`<br>
**Model Architecture:** CNN-ViT FusionNet (1216-dim representation, 10 classes)

---

## 1. Hardware & CUDA Environment Confirmation

- **CUDA Availability:** `True`
- **Active GPU:** `NVIDIA GeForce RTX 3050 4GB Laptop GPU`
- **Total Physical VRAM:** `4.00 GB` (4,096 MB)
- **PyTorch Version:** `2.5.1+cu121`
- **CUDA Driver / Runtime:** `CUDA 12.1` / Driver `566.07`

---

## 2. Batch Size Scaling & Memory Headroom (AMP Enabled)

Evaluated across 20 training steps per configuration using synthetic Sentinel-2 tiles:

| Batch Size | Peak VRAM Allocated (GB) | Peak VRAM Reserved (GB) | VRAM Utilization (% of 4GB) | Average Step Time (s) | Throughput (samples/s) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2** | 0.97 GB | 1.12 GB | 24.3% | 0.1914s | 10.5 | ✅ Stable |
| **4** | 1.69 GB | 1.85 GB | 42.3% | 0.3051s | 13.1 | ✅ Stable |
| **8** | **3.10 GB** | **3.57 GB** | **77.5%** | **0.5647s** | **14.2** | ✅ **Optimal (Safe ~22.5% headroom)** |
| **16** | 5.95 GB | 6.98 GB | >100% | 5.6101s | 2.9 | ⚠️ **WDDM Host RAM Paged (Throughput collapsed ~80%)** |
| **32** | **OOM** | **OOM** | >100% | N/A | N/A | ❌ **OOM (`torch.cuda.OutOfMemoryError`)** |

> [!NOTE]
> **Windows WDDM Memory Paging Behavior:**
> At batch size 16, activation requirements exceed the physical 4.00 GB VRAM limit. Under Windows WDDM, PyTorch pages excess allocations into shared system RAM over PCIe instead of immediately throwing an error. This causes severe memory thrashing: step time jumps by nearly 10x (from 0.56s to 5.61s) and throughput plummets from 14.2 to 2.9 samples/s. At batch size 32, allocation fails completely with `torch.cuda.OutOfMemoryError`.

---

## 3. Full 13-Band (No Mask) Configuration at Batch Size 8

To establish whether the model can run with all 13 Sentinel-2 spectral bands without channel slicing:

| Configuration | Input Tensor Shape | Peak VRAM Allocated (GB) | Peak VRAM Reserved (GB) | VRAM Headroom Remaining | Result |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Default 6-Band Mask** | `(8, 6, 256, 256)` | 3.10 GB (77.5%) | 3.57 GB (89.2%) | 0.43 GB (10.8%) | ✅ Stable |
| **Full 13 Bands (All Bands)** | `(8, 13, 256, 256)` | **3.16 GB (79.0%)** | **3.57 GB (89.2%)** | **0.43 GB (10.8%)** | ✅ **Stable (Fits within 4GB)** |

**Conclusion:** The full 13-band spectrum fits safely within the 4.00 GB hardware envelope at batch size 8 with AMP ON, requiring only 60 MB additional activation memory over the 6-band mask and maintaining ~10.8% VRAM headroom.

---

## 4. Gradient Accumulation Scaling (Effective Batch Size Alternatives)

Gradient accumulation simulates larger batch sizes without increasing peak activation memory:

| Micro Batch Size | Grad Accum Steps | Effective Batch Size | Peak VRAM Allocated (GB) | Step Time (s) | Throughput (samples/s) | Stability / Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **8** | 2 | **16** | **3.11 GB** | 0.5296s | **15.1** | ✅ **Ideal for Eff BS = 16** |
| **8** | 4 | **32** | **3.11 GB** | 0.4922s | **16.3** | ✅ **Ideal for Eff BS = 32** |
| **16** | 2 | 32 | 5.95 GB | 5.4701s | 2.9 | ⚠️ Host RAM Thrashing |
| **16** | 4 | 64 | 5.95 GB | 5.5139s | 2.9 | ⚠️ Host RAM Thrashing |

> [!NOTE]
> **Optimization Dynamics Nuance:**
> Gradient accumulation provides near-equivalent parameter gradient dynamics to large batch sizes. However, any normalization layers (such as BatchNorm) continue to compute running mean and variance statistics on the individual micro-batches (size 8), rather than across the accumulated effective batch. Using micro-batch size 8 with `--grad_accum_steps 2` or `4` delivers **5.5x higher throughput** (16.3 samples/s vs 2.9 samples/s) while staying safely inside physical VRAM.

---

## 5. DataLoader Multiprocessing Scaling (num_workers)

Evaluated with Batch Size `8` on Windows 11:

| Workers (`--num_workers`) | Average Step Time (s) | Throughput (samples/s) | Speedup vs 0 | Observations & Stability |
| :---: | :---: | :---: | :---: | :--- |
| **0** | 0.6131s | 13.0 | 1.00x | Synchronous single-process. Zero worker IPC overhead; fallback baseline. |
| **2** | 0.4309s | 18.6 | 1.43x | Dual worker pool prefetching tiles asynchronously. |
| **4** | **0.2972s** | **26.9** | **2.07x** | Quad worker pool. Default setting in `train_gpu.py`. Fully saturates GPU compute pipeline. |

---

## 6. Mixed Precision Comparison (AMP ON vs AMP OFF)

Evaluated with Batch Size `8`:

| Configuration | Peak VRAM Allocated (GB) | Average Step Time (s) | Throughput (samples/s) | VRAM Savings | Measured Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AMP ON (FP16)** | **3.10 GB** | **0.4799s** | **16.7 samples/s** | **20.9% lower** | **~3.5x** |
| **AMP OFF (FP32)** | 3.92 GB | 1.6938s | 4.7 samples/s | Baseline (Near OOM) | Baseline |

> [!NOTE]
> **Context on Speedup Measurement:**
> In the FP32 baseline run, peak allocated VRAM reached **3.92 GB** (98% of physical capacity). At this extreme threshold, background Windows WDDM paging mechanisms were likely active, introducing PCIe memory latency. As a result, the measured 3.53x speedup is likely an overestimate compared to pure compute acceleration on larger VRAM GPUs. Nevertheless, the operational takeaway is unambiguous: AMP is essential on 4GB hardware to avoid paging stalls and ensure memory safety.

---

## 7. Training Time Projections for Real-World Datasets (40 Epochs)

### A. Conservative Configuration (`--batch_size 8`, `--num_workers 0`, 14.2 samples/s)
| Dataset Size | Epoch Duration | 40-Epoch Total Duration | Practical Deployment Profile |
| :---: | :---: | :---: | :--- |
| **5,000 tiles** | ~5.9 min (352s) | **~3.92 hours** | Excellent for day-time iteration and baseline verification. |
| **20,000 tiles** | ~23.5 min (1,408s) | **~15.69 hours** | Feasible for overnight runs on RTX 3050. |

### B. High-Throughput Configuration (`--batch_size 8`, `--num_workers 4`, 26.9 samples/s)
| Dataset Size | Epoch Duration | 40-Epoch Total Duration | Practical Deployment Profile |
| :---: | :---: | :---: | :--- |
| **5,000 tiles** | ~3.1 min (186s) | **~2.07 hours** | Rapid training; finishes in ~2 hours. |
| **20,000 tiles** | ~12.4 min (743s) | **~8.25 hours** | Complete 40-epoch training in a single 8-hour workday. |

---

## 8. Recommendations & Safe Production Configuration

1. **Largest Safe Batch Size:**
   - **Recommended:** `--batch_size 8` (Peak Allocated VRAM: `3.10 GB` for 6 bands, `3.16 GB` for 13 bands).
   - Leaves **~10.8% - 22.5% VRAM headroom** on 4.00 GB hardware. Prevents WDDM host RAM paging and provides a safe buffer against fragmentation.

2. **Effective Batch Size Scaling:**
   - For Batch Size 16: `--batch_size 8 --grad_accum_steps 2`
   - For Batch Size 32: `--batch_size 8 --grad_accum_steps 4`
   - **Never run `--batch_size 16` directly** on 4GB VRAM.

3. **DataLoader Workers (`--num_workers`):**
   - **Default & Recommended:** `--num_workers 4`.
   - `train_gpu.py` probes worker startup at initialization and automatically falls back to `num_workers=0` if worker spawning fails.

4. **Mixed Precision:**
   - Always run with `--device cuda` which activates AMP (`torch.amp.autocast('cuda')`).

---

## 9. Performance Bottlenecks & Optimization Roadmap

### Bottlenecks Identified:
1. **On-Disk I/O Latency:**
   - Reading individual uncompressed `.npy` or GeoTIFF `.tif` tiles from disk per batch causes step latency to increase when worker prefetching is disabled.
2. **Memory Footprint of Tile Caching:**
   - A single $13 \times 256 \times 256$ `float32` tile occupies $\approx 3.41\text{ MB}$.
   - A 5,000-tile dataset occupies $\approx 17.0\text{ GB}$, and 20,000 tiles occupies $\approx 68.2\text{ GB}$. Uncompressed float32 caching in system RAM would exhaust host memory on typical 16GB laptops.

### Optimizations Implemented & Proposed:
1. **Memory-Mapped Tile Loading (`mmap_mode='r'`) [Implemented]:**
   - In `data/datasets.py`, `.npy` files are read with `np.load(full_path, mmap_mode='r')`. This allows the operating system page cache to keep frequently accessed tiles hot in memory with zero user-space copying overhead while ensuring contiguous `float32` tensors are returned.
2. **Optional RAM Budget-Capped Cache (`--cache_budget_gb`) [Proposed for Future Implementation]:**
   - Rather than uncompressed float32 caching, an optional cache storing tiles in `float16` or `uint16` (raw DNs) capped by a user-defined RAM budget (e.g. `--cache_budget_gb 4`, off by default). Float16 reduces the footprint to $\approx 1.7\text{ MB}$ per tile, allowing 2,000+ hot tiles to fit safely in 3.5 GB of system RAM without risk of host OOM.
