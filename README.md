# VOXEL: Earth Observation & Spectral Intelligence

> Physics-Guided Band Selection and Hybrid CNN-ViT Multimodal Fusion for Satellite Earth Observation and Land Cover Classification.

---

## 🛰️ Module 3 Overview (Deep Learning, Explainability & Vector Retrieval)

Module 3 delivers the neural classifier and similarity engine for the VOXEL framework:
- **Hybrid Classifier (FusionNet):** Combines **XNANet CNN** (1024-dim spatial texture representation) with **Tiny-32 Vision Transformer** (192-dim spectral attention representation) into a unified **1216-dimensional embedding** followed by a 10-class land cover head.
- **Dynamic Spectral Slicing:** Dynamically handles any arbitrary active channel count (e.g. 3, 6, 9, 13 bands) selected by metaheuristic optimization.
- **Explainable AI (SHAP):** Permutation SHAP channel attribution quantifying feature importance across active spectral bands.
- **Sub-Second Vector Search (FAISS):** Fast cosine similarity search on 1216-dim tile representations for geospatial scene retrieval.

---

## 🛠️ Environment & Prerequisites

- **Python Version:** `Python 3.10.11` (Recommended for PyTorch CUDA, Numba, and SHAP compatibility)
- **Compute:** NVIDIA CUDA GPU (e.g. RTX 3050 Laptop GPU, CUDA 12.1) or CPU fallback

### Setup
```bash
# 1. Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution & Command Reference

### 1. Synthetic Fake Dataset Generator
Generate synthetic tiles (.npy and .tif) on disk for rapid integration testing:
```bash
python tools/make_fake_dataset.py --output_dir data/fake_dataset --num_samples 40
```

### 2. GPU Model Training
```bash
# Run with synthetic mock data
python train_gpu.py --mock --epochs 50 --batch_size 8

# Run with on-disk satellite dataset
python train_gpu.py \
  --manifest data/fake_dataset/manifest.csv \
  --mask_path config/example_band_mask.json \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --grad_accum_steps 1
```

### 3. Model Evaluation on Test Split
Computes Overall Accuracy, Macro-F1, Mean IoU, per-class metrics, and saves `results/confusion_matrix.png`:
```bash
python evaluate.py --checkpoint checkpoints/best_fusion_net.pth --manifest data/fake_dataset/manifest.csv
# Or with mock data:
python evaluate.py --checkpoint checkpoints/best_fusion_net.pth --mock
```

### 4. Spectral Band Selection Benchmark
Compares (a) All 13 bands vs (b) Optimized mask vs (c) Random control mask and outputs `results/band_selection_comparison.md`:
```bash
python tools/compare_masks.py --mask_path config/example_band_mask.json --mock --epochs 3
```

### 5. Build Explainability & FAISS Retrieval Database
Regenerates `checkpoints/shap_band_importance.png`, `checkpoints/terrain_index.faiss`, and metadata:
```bash
python build_explainability.py \
  --checkpoint checkpoints/best_fusion_net.pth \
  --manifest data/fake_dataset/manifest.csv \
  --mask_path config/example_band_mask.json
```

### 6. End-to-End Inference & Vector Search Demo
Classifies incoming tiles and retrieves top-5 similar terrain scenes:
```bash
python demo_module3.py
```

### 7. Run Test Suite (pytest)
```bash
pytest -q
```

---

## 🤝 How to Integrate (Pre-Merge Handoff)

Module 3 is fully decoupled from teammates' active feature branches via configuration contracts.

For complete data schemas, tensor orderings, and checklist specifications, consult:
👉 **[Integration Contract (docs/INTEGRATION_CONTRACT.md)](docs/INTEGRATION_CONTRACT.md)**

### When Module 1 (Data Ingestion) Arrives:
- Ensure incoming tiles follow the contract: `float32` tensor `(13, 256, 256)` with values in `[0.0, 1.0]`.
- Provide a `manifest.csv` referencing `tile_path`, `label`, and optional `location_id`.
- Run: `python train_gpu.py --manifest <path_to_manifest.csv>`

### When Module 2 (Band Optimization) Arrives:
- Ensure the candidate mask is exported as JSON (see `config/example_band_mask.json`) or binary `.npy`/`.pt` array of shape `(13,)`.
- Run: `python train_gpu.py --manifest <manifest> --mask_path <path_to_mask.json>`

---

## 📁 Repository Structure

```
voxel-earth-observation/
├── checkpoints/             # Checkpoints, FAISS indices & README.md
├── config/                  # Centralized constants & example masks
│   ├── constants.py
│   └── example_band_mask.json
├── data/                    # Dataset loaders & augmentation
│   ├── datasets.py
│   └── fake_dataset/        # Local synthetic testing dataset
├── docs/                    # Integration contract & readiness reports
│   ├── INTEGRATION_CONTRACT.md
│   └── MERGE_READINESS.md
├── explainability/          # SHAP attribution & FAISS vector search
│   ├── faiss_search.py
│   └── shap_engine.py
├── models/                  # CNN, ViT, and FusionNet architectures
│   ├── cnn_backbone.py
│   ├── fusion_net.py
│   └── vit_transformer.py
├── results/                 # Training logs, confusion matrices, benchmarks
├── tests/                   # Pytest automated test suite
├── tools/                   # Benchmarks & dataset generators
│   ├── compare_masks.py
│   └── make_fake_dataset.py
├── utils/                   # Pluggable band mask loaders
│   └── band_mask.py
├── build_explainability.py  # SHAP/FAISS artifact builder CLI
├── demo_module3.py          # Interactive GPU inference demo
├── evaluate.py              # Test split evaluation script
├── inference.py             # Inference API & top-k retrieval
├── train_gpu.py             # Hardened GPU training script
├── requirements.txt         # Pinned working dependencies
└── README.md
```

---

## 👥 Team & Attribution

- **Arya Suryavanshi** - Data Ingestion & Preprocessing (Module 1)
- **Shreeya Joshi** - Physics-Guided & Metaheuristic Band Optimization (Module 2)
- **Riya Lepkar** - Deep Learning Training, Feature Fusion, Explainability & Vector Retrieval (Module 3)

**Academic Mentor:** Dr. Uma Jaishankar  
*Vidyalankar Institute of Technology (VIT)*
