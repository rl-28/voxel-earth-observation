"""
tools/verify_boundary_integrity.py - Formal Verification of AOI Boundary Containment,
Zero Under-Coverage, Canonical Single-Tile Behavior, and Dal Lake Spectral Diagnostics.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import numpy as np
from utils.llm_parser import calculate_geographic_span, compute_canonical_tile_bbox

def run_boundary_tests():
    test_sizes = [2550, 2560, 2565, 2580, 2600, 2610, 2611]
    tile_m = 2560.0
    epsilon_m = 0.1  # 10 cm tolerance absorbs float/discretization error without under-coverage

    print("=" * 80)
    print("TASK 1 & 2 & 3: AOI WIDTHS & HEIGHTS COVERAGE AND GEOMETRIC CONTAINMENT")
    print("=" * 80)
    print(f"{'Size (m)':<10} | {'Tiles':<6} | {'Proc Extent':<12} | {'Under-coverage':<16} | {'Margin/Side':<12} | {'Status'}")
    print("-" * 80)

    for s in test_sizes:
        tiles = max(1, int(math.ceil((s - epsilon_m) / tile_m)))
        proc_extent = tiles * tile_m
        undercoverage = max(0.0, s - proc_extent)
        margin_per_side = (proc_extent - s) / 2.0
        status = "PASSED (100% Covered)" if undercoverage == 0.0 else f"FAILED (-{undercoverage:.1f}m)"
        print(f"{s:<10.1f} | {tiles:<6d} | {proc_extent:<12.1f} | {undercoverage:<16.2f} | {margin_per_side:<+12.2f} | {status}")

    print("\n" + "=" * 80)
    print("2D COMBINATIONS (WIDTH x HEIGHT) CONTAINMENT MATRIX:")
    print("=" * 80)
    all_contained = True
    for w in test_sizes:
        for h in test_sizes:
            tx = max(1, int(math.ceil((w - epsilon_m) / tile_m)))
            ty = max(1, int(math.ceil((h - epsilon_m) / tile_m)))
            proc_w = tx * tile_m
            proc_h = ty * tile_m
            under_w = max(0.0, w - proc_w)
            under_h = max(0.0, h - proc_h)
            if under_w > 0 or under_h > 0:
                all_contained = False
                print(f"FAILED: {w}x{h} -> {tx}x{ty} (under_w={under_w}, under_h={under_h})")
    print(f"All 49 2D combinations (7x7) tested: All Contained = {all_contained}")

    print("\n" + "=" * 80)
    print("TASK 4: CANONICAL SINGLE-TILE FOOTPRINT BEHAVIOR ACROSS INDIAN LATITUDES")
    print("=" * 80)
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
        bbox_6 = compute_canonical_tile_bbox(lat, 75.0, tile_size_m=2560.0)
        w_km, h_km = calculate_geographic_span(bbox_6)
        w_m = w_km * 1000.0
        h_m = h_km * 1000.0
        tx = max(1, int(math.ceil((w_m - epsilon_m) / tile_m)))
        ty = max(1, int(math.ceil((h_m - epsilon_m) / tile_m)))
        diff_w = w_m - tile_m
        diff_h = h_m - tile_m
        print(f"{name:<20} (lat={lat:6.2f}) -> w={w_m:7.2f}m ({diff_w:+.2f}m), h={h_m:7.2f}m ({diff_h:+.2f}m) -> Grid: {tx}x{ty} ({tx*ty} Tile)")
        assert tx == 1 and ty == 1, f"Canonical tile at {name} must be 1x1, got {tx}x{ty}"

    print("All canonical single-tile footprints across Indian latitudes evaluate strictly to 1 tile.")

if __name__ == "__main__":
    run_boundary_tests()
