"""
Analyze ESA WorldCover distributions across the 140 genuine Sentinel-2 tiles.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("data/india/tiles_index.csv")
print("Total tiles in index:", len(df))
print("\nTiles per AOI:")
print(df['aoi'].value_counts())

print("\n--- Tree Cover (frac_10_tree) statistics by AOI ---")
for aoi, group in df.groupby('aoi'):
    trees = group['frac_10_tree'].values
    print(f"AOI: {aoi:15s} | Count: {len(group):2d} | Mean Tree: {trees.mean():.3f} | Min: {trees.min():.3f} | Max: {trees.max():.3f} | >60%: {(trees>=0.6).sum()} | <5%: {(trees<=0.05).sum()}")

print("\n--- Non-forest breakdown (Tree < 5%) ---")
non_forest_df = df[df['frac_10_tree'] <= 0.05]
print(f"Total non-forest tiles (<5% tree): {len(non_forest_df)}")
for aoi, group in non_forest_df.groupby('aoi'):
    print(f"AOI: {aoi:15s} | Count: {len(group):2d} | Dominant: {group['dominant_class'].value_counts().to_dict()}")

print("\n--- Forest breakdown (Tree >= 60%) ---")
forest_df = df[df['frac_10_tree'] >= 0.60]
print(f"Total forest tiles (>=60% tree): {len(forest_df)}")
for aoi, group in forest_df.groupby('aoi'):
    print(f"AOI: {aoi:15s} | Count: {len(group):2d}")
