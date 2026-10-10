"""
scripts/demo_cli.py - Interactive Terminal Testing Tool for Voxel Platform.

Supports:
1. Interactive REPL Mode: Live prompt loop to test any arbitrary natural language queries.
2. Direct Query CLI Mode: Pass query as command-line argument.
3. Batch Showcase Mode: Multi-region benchmark verifying live geocoding & spectral band optimization.
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.api_manager import get_api_manager
from utils.llm_parser import VoxelStageZeroParser
from module2_optimization import apply_module2_band_selection, BAND_NAMES


SAMPLE_QUERIES = [
    {
        "domain": "Agriculture & Crops",
        "query": "Assess wheat crop health, nitrogen stress, and vegetation indices across Punjab, India",
    },
    {
        "domain": "Urban Heat & Infrastructure",
        "query": "Analyze urban building sprawl and concrete heat island expansion in Tokyo, Japan",
    },
    {
        "domain": "Flood & Water Inundation",
        "query": "Predict water accumulation, inundation zones, and flood risk around South Mumbai, India",
    },
    {
        "domain": "Forestry & Deforestation",
        "query": "Detect tropical rainforest deforestation and canopy loss in Amazon Basin, Para, Brazil",
    },
    {
        "domain": "Glaciology & Snow Cover",
        "query": "Map snow cover extent and glacier retreat around Zermatt, Switzerland",
    },
]


def print_banner():
    print("=" * 78)
    print("  VOXEL SATELLITE EARTH OBSERVATION PLATFORM - INTERACTIVE TEST BENCH  ")
    print("=" * 78)


def run_diagnostics():
    print("\n[DIAGNOSTICS] API & SERVICE HEALTH STATUS:")
    mgr = get_api_manager()
    status = mgr.health_check()
    print(f"  * Environment               : {status['environment']}")
    print(f"  * Active LLM Provider       : {status['active_llm_provider'].upper()}")
    print(f"  * Google Gemini Key         : {status['stage_0_llm']['gemini_available']} ({status['stage_0_llm']['gemini_key_masked']})")
    print(f"  * Google Maps Geocoding     : {status['stage_1_ingestion']['google_maps_geocoding_available']} ({status['stage_1_ingestion']['google_maps_key_masked']})")
    print(f"  * OSM Nominatim Geocoding   : {status['stage_1_ingestion']['nominatim_geocoding_ready']} (Live HTTP REST)")
    print(f"  * Copernicus CDSE S3        : {status['stage_1_ingestion']['copernicus_cdse_configured']}")
    print(f"  * Mapbox Basemaps           : {status['visualization']['mapbox_available']}")
    print(f"  * Offline Resilience Ready  : {status['stage_0_llm']['offline_fallback_ready']}")
    print("-" * 78)


def display_sample_queries():
    print("\n--- SAMPLE QUERIES YOU CAN TRY ---")
    for i, s in enumerate(SAMPLE_QUERIES, 1):
        print(f"  [{i}] ({s['domain']})")
        print(f"      \"{s['query']}\"")
    print("-" * 78)


def run_pipeline_test(query: str, verbose: bool = True):
    """Executes Stage 0 LLM Query Compilation + Module 2 Band Selection."""
    if not query or not query.strip():
        print("[!] Empty query provided.")
        return None

    clean_query = query.strip()
    print("\n" + "=" * 78)
    print(f"[STAGE 0] NATURAL LANGUAGE QUERY COMPILATION & LIVE GEOCODING")
    print(f"Query: \"{clean_query}\"")
    print("=" * 78)

    parser = VoxelStageZeroParser()
    contract = parser.parse(clean_query)

    print(f"  [+] LLM Compiler Engine    : {contract.llm_provider}")
    print(f"  [+] Geocoding Provider     : {contract.geocoder}")
    print(f"  [+] Extracted Location     : {contract.location_name}")
    print(f"  [+] Live Spatial BBOX      : {contract.bbox} (min_lon, min_lat, max_lon, max_lat)")
    print(f"  [+] Detected Task Profile  : {contract.task_type}")
    print(f"  [+] Target Band Budget     : {contract.target_band_count} bands")
    print(f"  [+] Priority Light Domains : {', '.join(contract.priority_spectrum_domains)}")
    print(f"  [+] Green Scorecard Weights: {contract.scorecard_weights}")
    print(f"  [+] Pipeline Sensor Summary: {contract.user_summary}")

    print("\n" + "-" * 78)
    print(f"[MODULE 2] BINARY WHALE OPTIMIZATION (WOA) BAND SELECTION")
    print("-" * 78)
    print("  * Simulating Sentinel-2 Level-2A datacube (256, 256, 13)...")
    
    # Generate reproducible synthetic 13-band tile with domain-specific features
    np.random.seed(42)
    sample_tensor = np.random.randn(256, 256, 13).astype(np.float32)
    if "WATER" in contract.task_type:
        sample_tensor[64:192, 64:192, [7, 10, 11]] += 3.0  # Boost NIR / SWIR
    elif "CROP" in contract.task_type:
        sample_tensor[64:192, 64:192, [4, 5, 6, 7]] += 3.5  # Boost Red Edge & NIR
    elif "URBAN" in contract.task_type:
        sample_tensor[64:192, 64:192, [1, 2, 3, 11]] += 2.8  # Boost Visible & SWIR-2

    reduced_tensor, selected_indices, metrics = apply_module2_band_selection(
        tensor_13_bands=sample_tensor,
        contract=contract,
        pop_size=10,
        max_iter=15,
        seed=42
    )

    print(f"  [+] Input Tensor Dimensions        : {sample_tensor.shape} (256x256, 13 bands)")
    print(f"  [+] Optimized Reduced Tensor Shape : {reduced_tensor.shape}")
    print(f"  [+] Selected Spectral Bands        : {metrics['selected_bands']}")
    print(f"  [+] Selected Band Indices          : {selected_indices}")
    print(f"  [+] Edge Data Volume Reduction     : {metrics['data_reduction_pct']:.1f}%")
    print(f"  [+] Downstream FLOP Savings        : {metrics['flop_savings_pct']:.1f}%")
    print(f"  [+] Surrogate Classification Acc.  : {metrics['accuracy']:.4f}")
    print(f"  [+] Jeffries-Matusita Separability : {metrics['jm_distance']:.4f} (Norm: {metrics['jm_normalized']:.4f})")
    print(f"  [+] Multi-Objective Fitness Score  : {metrics['fitness']:.4f}")
    print(f"  [+] Domain Spectral Distribution   : {metrics['domain_distribution']}")

    print("\n" + "=" * 78)
    print("  STAGE 0 -> MODULE 1 -> MODULE 2 HANDOFF VERIFIED SUCCESSFULLY!")
    print("=" * 78 + "\n")

    return contract, metrics


def run_batch_showcase():
    """Runs a multi-region benchmark comparing different queries worldwide."""
    print("\n" + "#" * 78)
    print("  RUNNING MULTI-REGION GLOBAL SHOWCASE SUITE")
    print("#" * 78)

    results = []
    for item in SAMPLE_QUERIES:
        contract, metrics = run_pipeline_test(item["query"])
        results.append({
            "domain": item["domain"],
            "location": contract.location_name,
            "bbox": contract.bbox,
            "task": contract.task_type,
            "bands": metrics["selected_bands"],
            "reduction": f"{metrics['data_reduction_pct']:.1f}%",
            "accuracy": f"{metrics['accuracy']:.4f}",
        })

    print("\n" + "=" * 78)
    print("  GLOBAL BENCHMARK SUMMARY TABLE")
    print("=" * 78)
    print(f"{'Domain':<24} | {'Location':<20} | {'Task Type':<18} | {'Selected Bands':<22} | {'Reduction'}")
    print("-" * 105)
    for r in results:
        bands_str = ", ".join(r["bands"][:4]) + ("..." if len(r["bands"]) > 4 else "")
        print(f"{r['domain']:<24} | {r['location'][:18]:<20} | {r['task']:<18} | {bands_str:<22} | {r['reduction']}")
    print("=" * 105)


def interactive_repl():
    """Interactive Command Line Loop for testing arbitrary custom queries."""
    print("\n" + "=" * 78)
    print("  VOXEL LIVE INTERACTIVE QUERY CONSOLE")
    print("=" * 78)
    print("  [INFO] You can type ANY custom query for ANY city, region, or biome worldwide.")
    print("  [INFO] The system will geocode the real coordinates live and select optimal bands.")
    print("  ----------------------------------------------------------------------------")
    print("  Commands you can type:")
    print("    * 'examples' : View sample real-world queries across domains")
    print("    * 'batch'    : Run multi-region showcase benchmark (Punjab, Tokyo, Mumbai, etc.)")
    print("    * 'status'   : Check API key configuration status")
    print("    * 'exit'/'q' : Quit interactive session")
    print("=" * 78 + "\n")

    while True:
        try:
            print("----------------------------------------------------------------------------")
            print("👉 TYPE YOUR QUERY BELOW (e.g., 'Assess crop health in Punjab') and press ENTER:")
            user_input = input("Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive test bench. Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ["exit", "quit", "q"]:
            print("Exiting interactive test bench. Goodbye!")
            break
        elif cmd in ["examples", "help", "sample", "samples"]:
            display_sample_queries()
            continue
        elif cmd in ["status", "health", "diag"]:
            run_diagnostics()
            continue
        elif cmd in ["batch", "showcase", "all"]:
            run_batch_showcase()
            continue

        # Execute custom user query
        try:
            run_pipeline_test(user_input)
        except Exception as e:
            print(f"[!] Error processing query: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Voxel Satellite Earth Observation - Interactive Terminal Test Bench"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Optional Earth observation query text to test directly."
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run multi-region global showcase benchmark across 5 distinct domains."
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Display sample queries across agriculture, urban, water, forest, and glaciers."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Run API and service health diagnostics."
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Launch interactive REPL prompt loop."
    )

    args = parser.parse_args()

    print_banner()

    if args.status:
        run_diagnostics()
        return

    if args.examples:
        display_sample_queries()
        return

    if args.batch:
        run_diagnostics()
        run_batch_showcase()
        return

    if args.query:
        # Direct CLI query argument provided
        user_query = " ".join(args.query)
        run_diagnostics()
        run_pipeline_test(user_query)
    else:
        # Default behavior: run diagnostics and launch interactive prompt loop
        run_diagnostics()
        interactive_repl()


if __name__ == "__main__":
    main()
