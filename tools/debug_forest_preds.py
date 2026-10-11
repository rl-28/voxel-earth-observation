"""
Inspect logits, probabilities, and spectral features of test tiles under current forest model.
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np
import pandas as pd
from data.datasets import RealSentinel2Dataset
from evaluate import load_evaluation_checkpoint

device = torch.device("cpu")
model, mask, class_list, ckpt = load_evaluation_checkpoint("checkpoints/real_forest_cover/best_fusion_net.pth", device)
print("Mask:", mask)
print("Active channels:", ckpt.get("selected_channels"))

manifest_df = pd.read_csv("data/india/manifest_forest_cover.csv")
ds = RealSentinel2Dataset(manifest_df=manifest_df, split="test", task="forest_cover")
print("Test samples:", len(ds))

for i in range(len(ds)):
    x, y = ds[i]
    x_in = x.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x_in, mask=mask)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
        pred = int(torch.argmax(logits, dim=1).item())
    
    tile_row = ds.records.iloc[i]
    print(f"Sample {i:2d} | True: {class_list[y]:10s} | Pred: {class_list[pred]:10s} | P(Non-forest): {probs[0]:.4f} | P(Forest): {probs[1]:.4f} | Path: {tile_row['tile_path']}")
