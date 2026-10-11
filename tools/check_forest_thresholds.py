"""
Detailed breakdown of landcover classes in tiles_index.csv
"""
import pandas as pd

df = pd.read_csv("data/india/tiles_index.csv")

print(df.groupby('aoi')[['frac_10_tree', 'frac_20_shrub', 'frac_30_grass', 'frac_40_crop', 'frac_50_built', 'frac_60_bare', 'frac_80_water', 'frac_90_wetland']].mean().round(3))

print("\n--- Non-forest candidate definitions ---")
# If non-forest is tree <= 0.10 or tree <= 0.15:
for thresh in [0.05, 0.10, 0.15, 0.20]:
    sub = df[df['frac_10_tree'] <= thresh]
    print(f"Tree <= {thresh:.2f}: {len(sub)} tiles across AOIs: {sub['aoi'].value_counts().to_dict()}")

# What if Forest is tree >= 0.50 or 0.60?
for thresh in [0.50, 0.60, 0.70]:
    sub = df[df['frac_10_tree'] >= thresh]
    print(f"Tree >= {thresh:.2f}: {len(sub)} tiles across AOIs: {sub['aoi'].value_counts().to_dict()}")
