"""
Test Live Real Satellite Inference across two independent geographic locations in India:
Location 1: Loktak Lake, Manipur (Waterbody Task)
Location 2: Jim Corbett National Park, Uttarakhand (Forest Cover Task)
"""
import sys
import os
import json
import time
sys.path.insert(0, os.path.abspath("."))

from voxel_pipeline import VoxelPipeline, VoxelPipelineResult

def run_live_tests():
    print("=" * 80)
    print("VOXEL: LIVE REAL SATELLITE INFERENCE VERIFICATION")
    print("=" * 80)

    pipeline = VoxelPipeline(default_mode="real", verbose=True)

    locations = [
        {
            "name": "Loktak Lake, Manipur",
            "query": "Track waterbody levels in Loktak Lake, Manipur",
            "task": "waterbody",
        },
        {
            "name": "Jim Corbett National Park, Uttarakhand",
            "query": "Assess forest canopy in Jim Corbett National Park, Uttarakhand",
            "task": "forest_cover",
        }
    ]

    results = []

    for loc in locations:
        print("\n" + "#" * 80)
        print(f"TEST LOCATION: {loc['name']}")
        print(f"QUERY:         \"{loc['query']}\"")
        print(f"EXPECTED TASK: {loc['task']}")
        print("#" * 80)

        t0 = time.perf_counter()
        res: VoxelPipelineResult = pipeline.run(
            query=loc["query"],
            mode="real",
            task_override=loc["task"],
            run_shap=False,
            run_faiss=True,
        )
        elapsed = time.perf_counter() - t0

        print("\n--- INFERENCE RESULTS AUDIT ---")
        print(f"Status:            {res.status}")
        print(f"Execution Mode:    {res.execution_mode}")
        print(f"Data Source:       {res.data_source}")
        print(f"Is Synthetic:      {res.is_synthetic}")
        print(f"Resolved Location: {res.resolved_location}")
        print(f"Resolved Task:     {res.task_type}")
        print(f"Input Shape:       {res.input_tensor_shape}")
        print(f"Active Bands:      {res.selected_bands} (Reduction: {res.band_reduction_pct:.1f}%)")
        print(f"Model Provenance:  {res.model_provenance}")
        print(f"Prediction:        {res.prediction}")
        print(f"Timing Breakdown:  {res.timing_breakdown_ms}")
        print(f"Errors:            {res.errors}")
        print(f"Warnings:          {res.warnings}")

        # Assertions
        assert res.is_synthetic is False, "Violation: Synthetic flag is True in Real Satellite Mode!"
        assert res.status in ["COMPLETE", "PARTIALLY_COMPLETE"], f"Unexpected status: {res.status}"
        assert len(res.selected_bands) > 0, "No spectral bands selected!"
        if res.prediction:
            assert res.prediction.get("provenance") == "real_inference", "Violation: Prediction not labeled real_inference!"
            assert 0.0 <= res.prediction.get("confidence_pct", 0) <= 100.0, "Invalid confidence value!"

        results.append(res)

    print("\n" + "=" * 80)
    print("LIVE REAL-SATELLITE INFERENCE VERIFICATION SUCCESSFUL")
    print("=" * 80)
    return results

if __name__ == "__main__":
    run_live_tests()
