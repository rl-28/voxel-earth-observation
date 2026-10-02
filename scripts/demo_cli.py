"""
scripts/demo_cli.py - Interactive Terminal Testing Tool for Voxel Platform.

Tests Stage 0 (Query Compiler & Geocoding), APIManager Health, and Module 2 (Band Selection).
"""

import sys
import os
from pathlib import Path
import numpy as np

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.api_manager import get_api_manager
from utils.llm_parser import VoxelStageZeroParser
from module2_optimization import apply_module2_band_selection, BAND_NAMES


def print_banner():
    print("=" * 75)
    print("  VOXEL SATELLITE EARTH OBSERVATION PLATFORM - TERMINAL TEST BENCH  ")
    print("=" * 75)


def run_diagnostics():
    print("\n[1/3] CHECKING API & SERVICE HEALTH STATUS...")
    mgr = get_api_manager()
    status = mgr.health_check()
    print(f"  * Environment           : {status['environment']}")
    print(f"  * Active LLM Provider   : {status['active_llm_provider'].upper()}")
    print(f"  * Gemini Configured     : {status['stage_0_llm']['gemini_available']} ({status['stage_0_llm']['gemini_key_masked']})")
    print(f"  * CDSE S3 Configured    : {status['stage_1_ingestion']['copernicus_cdse_configured']}")
    print(f"  * Mapbox Configured     : {status['visualization']['mapbox_available']}")
    print(f"  * Offline Fallbacks     : {status['stage_0_llm']['offline_fallback_ready']}")


def run_pipeline_test(query: str):
    print("\n" + "-" * 75)
    print(f"[2/3] RUNNING STAGE 0: NATURAL LANGUAGE QUERY COMPILATION")
    print(f"User Query: \"{query}\"")
    print("-" * 75)

    parser = VoxelStageZeroParser()
    contract = parser.parse(query)

    print(f"  [+] Location Extracted   : {contract.location_name}")
    print(f"  [+] Spatial BBOX         : {contract.bbox} (min_lon, min_lat, max_lon, max_lat)")
    print(f"  [+] Task Profile         : {contract.task_type}")
    print(f"  [+] Target Band Budget   : {contract.target_band_count} bands")
    print(f"  [+] Priority Spectrum    : {', '.join(contract.priority_spectrum_domains)}")
    print(f"  [+] Scorecard Weights    : {contract.scorecard_weights}")
    print(f"  [+] Sensor Summary       : {contract.user_summary}")

    print("\n" + "-" * 75)
    print(f"[3/3] RUNNING MODULE 2: BINARY WHALE OPTIMIZATION BAND SELECTION")
    print("-" * 75)
    print("  * Generating synthetic Sentinel-2 Level-2A cube (256, 256, 13)...")
    np.random.seed(42)
    sample_tensor = np.random.randn(256, 256, 13).astype(np.float32)
    sample_tensor[128:, :, 1] += 2.5
    sample_tensor[:, 128:, 7] += 3.5

    reduced_tensor, selected_indices, metrics = apply_module2_band_selection(
        tensor_13_bands=sample_tensor,
        contract=contract,
        pop_size=10,
        max_iter=15,
        seed=42
    )

    print(f"  [+] Input Spatial-Spectral Shape   : {sample_tensor.shape}")
    print(f"  [+] Reduced Output Tensor Shape    : {reduced_tensor.shape}")
    print(f"  [+] Selected Spectral Channels     : {metrics['selected_bands']} (Indices: {selected_indices})")
    print(f"  [+] Data Volume Reduction          : {metrics['data_reduction_pct']:.1f}%")
    print(f"  [+] Early-Stage FLOP Savings       : {metrics['flop_savings_pct']:.1f}%")
    print(f"  [+] Surrogate Accuracy Score       : {metrics['accuracy']:.4f}")
    print(f"  [+] Jeffries-Matusita Separability : {metrics['jm_distance']:.4f} (Normalized: {metrics['jm_normalized']:.4f})")
    print(f"  [+] Green-AI Multi-Objective Score : {metrics['fitness']:.4f}")
    print(f"  [+] Domain Distribution            : {metrics['domain_distribution']}")

    print("\n" + "=" * 75)
    print("  STAGE 0 -> MODULE 1 -> MODULE 2 HANDOFF COMPLETED SUCCESSFULLY!  ")
    print("=" * 75)


if __name__ == "__main__":
    print_banner()
    run_diagnostics()

    default_query = "Predict where highest rainfall accumulation and flood risk will occur in South Mumbai"
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = default_query

    run_pipeline_test(user_query)
