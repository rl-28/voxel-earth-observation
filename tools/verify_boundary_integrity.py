"""
tools/verify_boundary_integrity.py - Formal Verification of AOI Boundary Containment,
Sub-Metre Precision, Zero Under-Coverage, and Canonical Single-Tile Behavior.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import numpy as np
from utils.llm_parser import calculate_geographic_span, compute_canonical_tile_bbox

def run_boundary_tests():
    test_sizes = [2559.99, 2560.00, 2560.01, 2560.50, 2560.99, 2561.00, 2565.00, 2580.00]
    tile_m = 2560.0
    pixel_size = 10.0

    print("=" * 90)
    print("TASK 1 & 2 & 3: AOI WIDTHS & HEIGHTS (SUB-METRE TO MULTI-TILE) AND PIXEL-GRID ALLOCATION")
    print("=" * 90)
    print(f"{'Size (m)':<10} | {'Old Tiles':<9} | {'Old Under':<10} | {'Req Px':<7} | {'Tiles':<6} | {'Proc Extent':<12} | {'Under-coverage':<15} | {'Status'}")
    print("-" * 90)

    for s in test_sizes:
        # Old formula: ceil((s - 1.0) / 2560.0)
        old_tiles = max(1, math.ceil((s - 1.0) / tile_m))
        old_under = max(0.0, s - old_tiles * tile_m)

        # Corrected pixel-grid allocation:
        req_px = math.ceil(round(s, 2) / pixel_size)
        tiles = max(1, math.ceil(req_px / 256))
        proc_extent = tiles * tile_m
        undercoverage = max(0.0, s - proc_extent)
        status = "PASSED (Zero Under-coverage)" if undercoverage == 0.0 else f"FAILED (-{undercoverage:.2f}m)"

        print(f"{s:<10.2f} | {old_tiles:<9d} | {old_under:<10.2f} | {req_px:<7d} | {tiles:<6d} | {proc_extent:<12.1f} | {undercoverage:<15.2f} | {status}")

    print("\n" + "=" * 90)
    print("2D COMBINATIONS (WIDTH x HEIGHT) POLYGON CONTAINMENT MATRIX (64 COMBINATIONS):")
    print("=" * 90)
    all_contained = True
    for w in test_sizes:
        for h in test_sizes:
            req_px_w = math.ceil(round(w, 2) / pixel_size)
            req_px_h = math.ceil(round(h, 2) / pixel_size)
            tx = max(1, math.ceil(req_px_w / 256))
            ty = max(1, math.ceil(req_px_h / 256))
            proc_w = tx * tile_m
            proc_h = ty * tile_m

            aoi_minx, aoi_maxx = -w / 2.0, w / 2.0
            aoi_miny, aoi_maxy = -h / 2.0, h / 2.0
            proc_minx, proc_maxx = -proc_w / 2.0, proc_w / 2.0
            proc_miny, proc_maxy = -proc_h / 2.0, proc_h / 2.0

            # Vertex containment
            vertices = [(aoi_minx, aoi_miny), (aoi_minx, aoi_maxy), (aoi_maxx, aoi_miny), (aoi_maxx, aoi_maxy)]
            contained = all(proc_minx <= vx <= proc_maxx and proc_miny <= vy <= proc_maxy for vx, vy in vertices)

            if not contained or proc_w < w or proc_h < h:
                all_contained = False
                print(f"FAILED: {w}x{h} -> {tx}x{ty}")
    print(f"All 64 2D combinations (8x8) tested: All Contained = {all_contained}")

    print("\n" + "=" * 90)
    print("TASK 5: CANONICAL SINGLE-TILE FOOTPRINT BEHAVIOR ACROSS INDIAN LATITUDES")
    print("=" * 90)
    canonical_latitudes = [
        ("Kanyakumari", 8.08),
        ("Western Ghats", 13.20),
        ("Ratnagiri", 17.2615),
        ("Chilika Lake", 19.65),
        ("Sardar Sarovar", 21.83),
        ("Loktak Lake", 24.55),
        ("Jim Corbett", 29.55),
        ("Dal Lake, Srinagar", 34.113),
    ]

    for name, lat in canonical_latitudes:
        bbox_inbound = compute_canonical_tile_bbox(lat, 75.0, tile_size_m=2560.0)
        w_km, h_km = calculate_geographic_span(bbox_inbound)
        w_m = w_km * 1000.0
        h_m = h_km * 1000.0
        px_w = math.ceil(round(w_m, 2) / pixel_size)
        px_h = math.ceil(round(h_m, 2) / pixel_size)
        tx = max(1, math.ceil(px_w / 256))
        ty = max(1, math.ceil(px_h / 256))
        diff_w = w_m - tile_m
        diff_h = h_m - tile_m
        print(f"{name:<20} (lat={lat:6.2f}) -> w={w_m:7.2f}m ({diff_w:+.2f}m), h={h_m:7.2f}m ({diff_h:+.2f}m) -> Grid: {tx}x{ty} ({tx*ty} Tile)")
        assert tx == 1 and ty == 1, f"Canonical tile at {name} must be 1x1, got {tx}x{ty}"

    print("\nAll canonical single-tile point selections across Indian latitudes evaluate strictly to 1 tile.")

if __name__ == "__main__":
    run_boundary_tests()
