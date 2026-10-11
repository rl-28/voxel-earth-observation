"""
tools/benchmark_cpu_training.py - CPU Training Feasibility & Performance Benchmark

Measures on the local host CPU (16 GB RAM):
- Batch processing latency (forward pass, backward pass, optimizer step)
- Process RAM consumption (RSS MB)
- Training throughput (samples/sec)
- Projected epoch duration across dataset sizes (50, 200, 1000 samples)
"""

import time
import os
import sys
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

import torch
import torch.nn as nn
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.fusion_net import FusionNet
from utils.band_mask import load_band_mask
from config.constants import EMBEDDING_DIM, NUM_BANDS


def measure_cpu_benchmark(
    batch_sizes=[2, 4, 8],
    num_batches=5,
    selected_channels=6,
    num_classes=2,
):
    print("=" * 70)
    print("  VOXEL FUSIONNET CPU TRAINING BENCHMARK & FEASIBILITY ASSESSMENT")
    print(f"  CPU Cores Available: {os.cpu_count()}")
    if HAS_PSUTIL:
        print(f"  System RAM Total   : {psutil.virtual_memory().total / (1024**3):.2f} GB")
        print(f"  System RAM Available: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    else:
        print("  System RAM: 16 GB (Host specification)")
    print("=" * 70)

    process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
    results = {}

    for bs in batch_sizes:
        print(f"\n[BENCHMARK] Testing Batch Size: {bs} (Selected Channels: {selected_channels}, Classes: {num_classes})")
        
        # Instantiate model
        model = FusionNet(
            num_classes=num_classes,
            total_bands=NUM_BANDS,
            selected_channels=selected_channels,
            cnn_feature_dim=1024,
            vit_feature_dim=192,
        )
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        mask = torch.zeros(NUM_BANDS)
        mask[[0, 2, 3, 5, 8, 11][:selected_channels]] = 1.0

        # Warmup
        x_warmup = torch.rand(bs, NUM_BANDS, 256, 256)
        y_warmup = torch.randint(0, num_classes, (bs,))
        out = model(x_warmup, mask=mask)
        loss = criterion(out, y_warmup)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        forward_times = []
        backward_times = []
        total_batch_times = []

        ram_before = (process.memory_info().rss / (1024**2)) if process else 0.0

        for b in range(num_batches):
            x = torch.rand(bs, NUM_BANDS, 256, 256)
            y = torch.randint(0, num_classes, (bs,))

            t0 = time.perf_counter()
            logits = model(x, mask=mask)
            t1 = time.perf_counter()
            loss = criterion(logits, y)
            loss.backward()
            t2 = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad()
            t3 = time.perf_counter()

            forward_times.append(t1 - t0)
            backward_times.append(t2 - t1)
            total_batch_times.append(t3 - t0)

        ram_after = (process.memory_info().rss / (1024**2)) if process else 0.0
        avg_fwd = np.mean(forward_times) * 1000.0
        avg_bwd = np.mean(backward_times) * 1000.0
        avg_batch = np.mean(total_batch_times) * 1000.0
        throughput = bs / (np.mean(total_batch_times))

        print(f"  * Forward Pass Latency  : {avg_fwd:.2f} ms")
        print(f"  * Backward Pass Latency : {avg_bwd:.2f} ms")
        print(f"  * Total Step Latency    : {avg_batch:.2f} ms per batch")
        print(f"  * Training Throughput   : {throughput:.2f} samples/sec")
        if process:
            print(f"  * Process RAM Usage     : {ram_after:.1f} MB (Delta: +{ram_after - ram_before:.1f} MB)")

        # Projected durations
        for sample_count in [40, 100, 500]:
            batches_per_epoch = int(np.ceil(sample_count / bs))
            epoch_sec = (batches_per_epoch * avg_batch) / 1000.0
            print(f"    - Projected {sample_count:3d} samples epoch: {epoch_sec:.1f} s ({epoch_sec/60:.2f} min)")

        results[bs] = {
            "forward_ms": avg_fwd,
            "backward_ms": avg_bwd,
            "batch_total_ms": avg_batch,
            "throughput_samples_per_sec": throughput,
            "ram_mb": ram_after,
        }

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)
    return results


if __name__ == "__main__":
    measure_cpu_benchmark()
