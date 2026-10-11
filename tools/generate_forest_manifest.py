"""
Generate scientifically balanced multi-modal Forest Cover manifest.
Ensures negative class covers all three non-forest spectral archetypes:
1. Bare Soil / Desert Sand (Thar)
2. Cropland / Agriculture (Sardar Sarovar)
3. Open Water / Wetlands (Kaziranga & Chilika)
"""
import pandas as pd
import numpy as np

df = pd.read_csv("data/india/tiles_index.csv")

# Filter candidates
forest_mask = df['frac_10_tree'] >= 0.50
non_forest_mask = df['frac_10_tree'] <= 0.20

df_sub = df[forest_mask | non_forest_mask].copy()
df_sub['label'] = np.where(df_sub['frac_10_tree'] >= 0.50, 'Forest', 'Non-forest')

# Assign splits strictly by AOI
# Train AOIs: western_ghats, thar, sardar_sarovar, kaziranga
# Val AOIs: kanha, chilika
# Test AOI: hirakud (Independent held-out geography)
train_aois = ['western_ghats', 'thar', 'sardar_sarovar', 'kaziranga']
val_aois = ['kanha', 'chilika']
test_aois = ['hirakud']

df_sub['split'] = 'unknown'
df_sub.loc[df_sub['aoi'].isin(train_aois), 'split'] = 'train'
df_sub.loc[df_sub['aoi'].isin(val_aois), 'split'] = 'val'
df_sub.loc[df_sub['aoi'].isin(test_aois), 'split'] = 'test'

out_df = df_sub[['tile_path', 'label', 'split', 'location_id']].copy()
out_df.to_csv("data/india/manifest_forest_cover.csv", index=False)

print("Saved manifest_forest_cover.csv with shape:", len(out_df))
print(pd.crosstab(out_df['split'], out_df['label'], margins=True))
