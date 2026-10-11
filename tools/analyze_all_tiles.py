"""
Analyze all 140 tiles for Forest Cover classification.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("data/india/tiles_index.csv")

print("=== ALL 140 TILES IN INDIA DATASET ===")
print(f"Total: {len(df)}")
for aoi, g in df.groupby('aoi'):
    print(f"\n--- AOI: {aoi} (N={len(g)}) ---")
    for idx, row in g.iterrows():
        tree = row['frac_10_tree']
        crop = row['frac_40_crop']
        water = row['frac_80_water']
        shrub = row['frac_20_shrub']
        grass = row['frac_30_grass']
        bare = row['frac_60_bare']
        built = row['frac_50_built']
        print(f"  {row['tile_path'][-16:]} | Tree: {tree:5.1%} | Crop: {crop:5.1%} | Water: {water:5.1%} | Shrub: {shrub:5.1%} | Grass: {grass:5.1%} | Bare: {bare:5.1%} | Dom: {row['dominant_class']}")
