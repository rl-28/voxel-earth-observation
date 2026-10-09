# VOXEL Module 3 Checkpoints & Artifacts Directory

> [!WARNING]
> **PLACEHOLDER - generated from random mock data, not valid results; regenerate after real training**
>
> The binary files and visualizations currently stored in this directory (`best_fusion_net.pth`, `terrain_index.faiss`, `terrain_index.faiss.meta.json`, `shap_band_importance.png`) were generated using synthetic / mock data during Module 3 pipeline hardening.
> **They must be regenerated after training on real satellite data once Module 1 (Data Ingestion) and Module 2 (Band Optimization) are integrated.**

---

## Artifact Descriptions

| Filename | Description | Status |
| :--- | :--- | :--- |
| `best_fusion_net.pth` | Fused CNN-ViT model checkpoint weights, optimizer state, registered band mask, class list, and git commit metadata. | PLACEHOLDER - generated from random mock data, not valid results; regenerate after real training |
| `terrain_index.faiss` | FAISS vector database storing 1216-dim normalized embeddings for similarity retrieval. | PLACEHOLDER - generated from random mock data, not valid results; regenerate after real training |
| `terrain_index.faiss.meta.json` | Metadata mapping vector indices to tile identifiers, true classes, predicted classes, and coordinates. | PLACEHOLDER - generated from random mock data, not valid results; regenerate after real training |
| `shap_band_importance.png` | Publication-ready horizontal bar chart quantifying Permutation SHAP spectral attribution. | PLACEHOLDER - generated from random mock data, not valid results; regenerate after real training |
| `splits.json` | Exact tile paths partitioned into reproducible `train`, `val`, and `test` splits. | Generated on training run |

---

## How to Regenerate Artifacts After Integration

### 1. Train Model on Real Ingested Sentinel-2 Tiles
```bash
python train_gpu.py \
  --manifest data/sentinel2_manifest.csv \
  --mask_path config/metaheuristic_mask.json \
  --epochs 50 \
  --batch_size 8
```

### 2. Regenerate SHAP & FAISS Artifacts from Real Validation Data
```bash
python build_explainability.py \
  --checkpoint checkpoints/best_fusion_net.pth \
  --manifest data/sentinel2_manifest.csv \
  --mask_path config/metaheuristic_mask.json
```

### 3. Evaluate Final Performance on Held-Out Test Set
```bash
python evaluate.py \
  --checkpoint checkpoints/best_fusion_net.pth \
  --manifest data/sentinel2_manifest.csv
```

### 4. Run Retrieval & Inference Demo
```bash
python demo_module3.py
```
