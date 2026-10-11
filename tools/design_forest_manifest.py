"""
Design a scientifically robust, geographically separated Forest Cover manifest.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("data/india/tiles_index.csv")

# Filter into Forest and Non-forest
forest_mask = df['frac_10_tree'] >= 0.50
non_forest_mask = df['frac_10_tree'] <= 0.20

df_filtered = df[forest_mask | non_forest_mask].copy()
df_filtered['label'] = np.where(df_filtered['frac_10_tree'] >= 0.50, 'Forest', 'Non-forest')

print(f"Total candidate tiles: {len(df_filtered)}")
print(df_filtered['label'].value_counts())
print("\nBreakdown by AOI and Class:")
print(pd.crosstab(df_filtered['aoi'], df_filtered['label']))

# Geographic split strategy:
# Train AOIs: western_ghats (Forest), thar (Non-forest bare), sardar_sarovar (Non-forest crop), kaziranga (Mixed)
# Val AOIs: kanha (Forest), chilika (Non-forest crop/water + Forest)
# Test AOI: hirakud (Independent held-out test region with both Forest and Non-forest water/crop)
print("\n--- Testing Split Assignments ---")
train_aois = ['western_ghats', 'thar', 'sardar_sarovar', 'kaziranga']
val_aois = ['kanha', 'chilika']
test_aois = ['hirakud']

df_filtered['split'] = 'unknown'
df_filtered.loc[df_filtered['aoi'].isin(train_aois), 'split'] = 'train'
df_filtered.loc[df_filtered['aoi'].isin(val_aois), 'split'] = 'val'
df_filtered.loc[df_filtered['aoi'].isin(test_aois), 'split'] = 'test'

print(pd.crosstab(df_filtered['split'], df_filtered['label']))
print("\nTest AOI Breakdown:")
print(pd.crosstab(df_filtered[df_filtered['split']=='test']['aoi'], df_filtered[df_filtered['split']=='test']['label']))
