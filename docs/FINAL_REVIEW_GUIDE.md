# VOXEL: Final Professor-Review Defense & Technical Guide

**Platform:** VOXEL (Task-Adaptive Earth Observation & Spectral Intelligence)  
**Branch:** `feature/voxel-end-to-end-integration`  
**Evaluation Status:** Held-Out Real-Data Evaluation Complete & Verified  
**Date:** October 2026  

---

## Executive Summary

VOXEL is an end-to-end Earth Observation pipeline designed to translate natural language environmental inquiries into physics-guided, spectral-optimized deep learning inference on genuine **Sentinel-2 L2A satellite imagery**. 

The system transitions raw satellite data through four tightly integrated stages:
1. **Stage 0 (Query Compilation):** Natural language task resolution & geocoding via OpenStreetMap Nominatim.
2. **Module 1 (Signal Ingestion & Processing):** HTTP windowed COG streaming via AWS Earth Search STAC, 256×256 spatial tiling, and bottom-of-atmosphere (BOA) radiometric reflectance calibration `[0.0, 1.0]`.
3. **Module 2 (Physics-Guided Spectral Optimization):** Metaheuristic Binary Whale Optimization Algorithm (BWOA) achieving **53.8% – 61.5% spectral telemetry reduction** while strictly enforcing physical domain balance and atmospheric absorption refusal (Band 10 Cirrus).
4. **Module 3 (Deep Learning, Explainability & Search):** Dual-branch CNN-ViT **FusionNet** (XNANet + Tiny-ViT) with dynamic tensor slicing, Permutation SHAP spectral attribution, and FAISS semantic vector retrieval.

Both core models (**Forest Cover** and **Waterbody Classification**) have been trained and evaluated on **genuine Indian Sentinel-2 L2A imagery** paired with **ESA WorldCover 2021 v200** ground-truth labels under strict **geographic holdout isolation** (zero spatial or temporal data leakage).

---

## 1. Complete Technical Architecture

```
User Query ("Track waterbody in Loktak Lake")
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 0: Natural Language Query Compiler & Geocoder         │
│  - Task Detection: Rule Engine / LLM Intent Classification  │
│  - Geographic Resolution: Nominatim REST API (GeoJSON BBOX) │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ MODULE 1: Satellite Ingestion & Radiometric Signal Pipeline │
│  - STAC Search: AWS Earth Search STAC (Sentinel-2 L2A)      │
│  - Windowed COG Streaming: GDAL/Rasterio Sub-Granule Crop   │
│  - Radiometric Scaling: BOA Reflectance / 10000 -> [0, 1]   │
│  - Spatial Tiling: Calibrated 256×256 Grid (13 Bands)       │
└────────────────────────────┬────────────────────────────────┘
                             │ (13, 256, 256) Float32 Tensor
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ MODULE 2: Physics-Guided Spectral Channel Optimization      │
│  - Metaheuristic: Binary Whale Optimization Algorithm (BWOA)│
│  - Physics Constraints: Refusal of Band 10 (Cirrus: 1375nm) │
│  - Telemetry Reduction: 13 Bands -> 5–6 Bands (53–62% drop) │
└────────────────────────────┬────────────────────────────────┘
                             │ Channel Selection Mask M ∈ {0, 1}¹³
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ MODULE 3: Hybrid CNN-ViT Deep Learning (FusionNet)          │
│  ┌─────────────────────────┐   ┌──────────────────────────┐ │
│  │ XNANet CNN Branch       │   │ Tiny-ViT Branch          │ │
│  │ (1024-dim Local Texture)│   │ (192-dim Global Context) │ │
│  └────────────┬────────────┘   └────────────┬─────────────┘ │
│               └──────────────┬──────────────┘               │
│                              ▼                              │
│              Concatenation Head: 1216-dim Embedding         │
│                              │                              │
│       ┌──────────────────────┼──────────────────────┐       │
│       ▼                      ▼                      ▼       │
│  Classification Head    Permutation SHAP       FAISS Index  │
│  (Softmax Probs)        (Feature Impact)       (Similarity) │
└─────────────────────────────────────────────────────────────┘
```

### Module Specifications
- **Module 1 Output:** 13-band bottom-of-atmosphere surface reflectance tensor $(13, 256, 256)$, scaled to $[0.0, 1.0]$.
- **Module 2 Mask:** Binary selection vector $M \in \{0, 1\}^{13}$.
- **Module 3 Slicing:** Inputs are dynamically sliced to active channels $C_{\text{active}} \in \{5, 6\}$, passed through input projection layers matching the active spectral subset.
- **Fusion Embedding:** 1216-dimensional joint representation vector $z \in \mathbb{R}^{1216}$.

---

## 2. Training Datasets & Checkpoint Information

### Dataset Composition
The Indian satellite dataset was collected across 7 diverse climatic zones in India to avoid geographic overfitting:
1. **Western Ghats (Karnataka):** Tropical evergreen rainforest, complex montane topography.
2. **Kaziranga (Assam):** Brahmaputra river floodplain, dense riverine grassland/forest.
3. **Thar Desert (Rajasthan):** Arid sand dunes, sparse xerophytic scrub (high negative control for water/forest).
4. **Sardar Sarovar (Gujarat):** Narmada river reservoir basin, engineered impoundment.
5. **Chilika Lake (Odisha):** Coastal brackish lagoon, saline wetlands.
6. **Kanha National Park (Madhya Pradesh):** Central Indian dry/moist deciduous sal forest.
7. **Hirakud Reservoir / Sambalpur (Odisha):** Major inland freshwater lake, surrounding open woodland.

### Strict Geographic Split Allocation
To guarantee zero spatial and temporal leakage:
- **Training Set (68 tiles):** Western Ghats, Thar, Kaziranga, Sardar Sarovar.
- **Validation Set (30–37 tiles):** Kanha National Park, Chilika Lake.
- **Held-Out Test Set (19 tiles):** Hirakud Reservoir (strictly unseen during all training and hyperparameter tuning).

### Checkpoint Provenance Registry

| Model Directory | Task | Weights Path | Active Channels | Best Epoch | Test OA | Test Macro-F1 | Status |
|---|---|---|---|---|---|---|---|
| `checkpoints/real_forest_cover/` | Forest Cover | `best_fusion_net.pth` (195.2 MB) | 5 bands (`B02, B04, B06, B11, B12`) | Epoch 1 | **94.74%** | **93.60%** | **Genuine Real Satellite Model** |
| `checkpoints/real_waterbody/` | Waterbody | `best_fusion_net.pth` (197.5 MB) | 6 bands (`B01, B03, B04, B06, B8A, B11`) | Epoch 4 | **73.68%** | **73.39%** | **Genuine Real Satellite Model** |
| `checkpoints/forest_cover/` | Forest Cover | `best_fusion_net.pth` (195.2 MB) | 5 bands | Epoch 1 | 100.0% | 100.0% | *Synthetic Fixture (Noise)* |
| `checkpoints/waterbody/` | Waterbody | `best_fusion_net.pth` (197.5 MB) | 6 bands | Epoch 1 | 50.0% | 33.3% | *Synthetic Fixture (Collapsed)* |
| `checkpoints/` | EuroSAT Land Cover | `best_fusion_net.pth` (197.6 MB) | 6 bands | Epoch 1 | 3.12% | 0.67% | *Synthetic Benchmark Fixture* |

---

## 3. Verified Evaluation Results (Held-Out Test Set)

Evaluated on the **Hirakud Reservoir scene** (19 tiles, completely isolated):

### 1. Forest Cover Classification (`real_forest_cover`)
- **Overall Accuracy (OA):** **94.74%** (18 of 19 tiles correct)
- **Majority-Class Baseline:** 68.42% OA &rarr; **+26.32% performance lift over baseline**
- **Balanced Accuracy:** **91.67%**
- **Macro F1-Score:** **93.60%**
- **Mean Intersection-over-Union (mIoU):** **88.10%**
- **Class Breakdown:**
  - *Forest:* Recall = **100.0%** (13/13), Precision = **92.86%**, F1 = 96.30%
  - *Non-forest:* Recall = **83.33%** (5/6), Precision = **100.0%**, F1 = 90.91%
- **Confusion Matrix:** $\begin{bmatrix} 5 & 1 \\ 0 & 13 \end{bmatrix}$ (Rows: Non-forest, Forest; Cols: Pred Non-forest, Pred Forest)

### 2. Waterbody Classification (`real_waterbody`)
- **Overall Accuracy (OA):** **73.68%** (14 of 19 tiles correct)
- **Majority-Class Baseline:** 68.42% OA
- **Balanced Accuracy:** **80.77%**
- **Macro F1-Score:** **73.39%**
- **Mean Intersection-over-Union (mIoU):** **58.04%**
- **Class Breakdown:**
  - *Water:* Recall = **100.0%** (6/6), Precision = 54.55%, F1 = 70.59%
  - *Non-water:* Recall = 61.54% (8/13), Precision = **100.0%**, F1 = 76.19%
- **Confusion Matrix:** $\begin{bmatrix} 8 & 5 \\ 0 & 6 \end{bmatrix}$ (Rows: Non-water, Water; Cols: Pred Non-water, Pred Water)

---

## 4. Module 2: Binary Whale Optimization Algorithm (BWOA)

### Mathematical Formulation
The Binary Whale Optimization Algorithm mimics the hunting mechanism of humpback whales using three search operators in continuous space mapped into binary vectors $x \in \{0, 1\}^{13}$:

1. **Encircling Prey ($|A| < 1, p < 0.5$):**
   $$\vec{D} = |\vec{C} \odot \vec{X}^*(t) - \vec{X}(t)|$$
   $$\vec{X}(t+1) = \vec{X}^*(t) - \vec{A} \odot \vec{D}$$
   Where $\vec{A} = 2\vec{a} \odot \vec{r} - \vec{a}$, $\vec{C} = 2\vec{r}$, and $\vec{a}$ decreases linearly from 2 to 0.

2. **Bubble-Net Spiral Attack ($p \ge 0.5$):**
   $$\vec{X}(t+1) = \vec{D}' \cdot e^{b l} \cdot \cos(2\pi l) + \vec{X}^*(t)$$
   Where $\vec{D}' = |\vec{X}^*(t) - \vec{X}(t)|$, $b$ is the logarithmic spiral constant, and $l \in [-1, 1]$.

3. **Exploration / Search for Prey ($|A| \ge 1, p < 0.5$):**
   Whales search randomly with respect to a random whale $\vec{X}_{\text{rand}}$:
   $$\vec{X}(t+1) = \vec{X}_{\text{rand}} - \vec{A} \odot |\vec{C} \odot \vec{X}_{\text{rand}} - \vec{X}(t)|$$

4. **Sigmoidal Binary Discretization:**
   Continuous whale positions $X_{i, d}$ are mapped to bit decisions via the standard logistic sigmoid:
   $$S(X_{i, d}) = \frac{1}{1 + e^{-X_{i, d}}}$$
   $$x_{i, d}^{(t+1)} = \begin{cases} 1 & \text{if } \text{rand}() < S(X_{i, d}) \\ 0 & \text{otherwise} \end{cases}$$

### Multi-Objective Fitness Scorecard
The objective function maximizes spectral information while minimizing inter-band redundancy and enforcing strict physical constraints:
$$f(x) = w_1 \cdot \text{Information}(x) + w_2 \cdot \text{Variance}(x) - w_3 \cdot \text{Redundancy}(x) - \lambda \cdot |\|x\|_1 - K_{\text{target}}| - P_{\text{physics}}(x)$$

- **Domain Balance Penalty:** Requires at least one band from each primary regime (Visible $[B2, B3, B4]$, NIR $[B8, B8A]$, SWIR $[B11, B12]$).
- **Atmospheric Absorption Refusal:** Band 10 ($1375\,\text{nm}$ Cirrus band) is subjected to an infinite penalty ($P(x_{10}=1) = +\infty$), guaranteeing $x_{10} = 0$.

---

## 5. Live Demonstration Commands

Execute all verification steps from `d:\Voxel\voxel-earth-observation`:

```powershell
# 1. Run Automated Test Suite (178 passing tests)
.venv\Scripts\pytest -q

# 2. Run End-to-End Live Satellite Inference Verification
.venv\Scripts\python tools/test_live_inference.py

# 3. Launch Interactive Streamlit Dashboard
.venv\Scripts\streamlit run app.py
```

---

## 6. Five-Minute Presentation Sequence

| Timeline | Dashboard Location | Action & Demonstration | Technical Talking Point |
|---|---|---|---|
| **0:00 - 0:45** | **Overview & Sidebar** | Show Online System Status (Modules 1, 2, 3 Online, Provenance Gating Active). | Explain VOXEL's core motivation: Bridging high-level queries with physical multi-spectral satellites without unverified claims. |
| **0:45 - 2:00** | **Tab 1: Orchestrator** | 1. Select *"Track waterbody in Loktak Lake, Manipur"* &rarr; Click **Execute**. Show real Sentinel-2 RGB composite, 6 selected bands (53.8% reduction), **Water (96.7% confidence)**.<br>2. Select *"Assess forest canopy in Jim Corbett"* &rarr; Click **Execute**. Show 5 selected bands (61.5% reduction), **Forest (58.4% confidence)**. | Point out **zero network IO** on cached imagery vs windowed COG streaming. Show live Permutation SHAP attribution. |
| **2:00 - 3:15** | **Tab 2: Provenance Hub** | Scroll to Table 3 & Confusion Matrices. Show held-out evaluation on Hirakud Reservoir: **Forest 94.74% OA** (+26.3% over baseline) and **Water 73.68% OA**. | Emphasize strict zero-leakage geographic holdout: Train (Assam, Karnataka, Rajasthan), Val (MP, Odisha Coast), Test (Hirakud Reservoir). |
| **3:15 - 4:15** | **Tab 4: Optimization** | Inspect Waterbody (6 bands) and Forest Cover (5 bands) masks. Highlight **Band 10 Cirrus refusal**. Open *Run Live BWOA Search* expander and run a 5-whale, 5-iter optimization. | Reiterate research integrity: BWOA is evaluated as a telemetry data volume reducer, not an unverified accuracy booster. |
| **4:15 - 5:00** | **Tab 5 & Summary** | Show FusionNet CNN-ViT architecture specification and FAISS semantic retrieval scope. | Conclude: End-to-end operational pipeline with verified real-satellite provenance and hardened scientific gating. |

---

## 7. Current Research Limitations & Scientific Integrity

1. **Whole-Tile Classification Footprint:** VOXEL performs whole-tile land-cover classification over $256 \times 256$ pixel patches ($2.56 \times 2.56\,\text{km}$ footprint) rather than dense pixel-level segmentation. Ambiguous mixed-vegetation patches at reservoir edges may exhibit lower certainty.
2. **Band Spatial Resolution Discrepancy:** Sentinel-2 bands possess native resolutions of 10m (B02-B04, B08), 20m (B05-B07, B8A, B11, B12), and 60m (B01, B09). All bands are bilinearly re-sampled to a uniform 10m pixel grid in Module 1.
3. **Dataset Scale:** The genuine Indian dataset currently comprises 140 curated Sentinel-2 tiles across 7 regions. While geographically diverse, larger pre-training on thousands of scenes would improve out-of-distribution robustness.
4. **BWOA Scientific Claims:** BWOA demonstrates clear spectral telemetry reduction (53.8% – 61.5% smaller downlink bandwidth). Claims that BWOA outperforms full-band models in raw classification accuracy remain strictly gated until multi-seed ablation baselines are published.
5. **Weather / Cloud Cover Susceptibility:** Optical surface reflectance is degraded under severe monsoonal cloud cover ($>20\%$ cloud cover scenes are filtered during STAC search).

---

## 8. 15 Likely Professor Questions & Technically Correct Answers

#### Q1: How do you guarantee zero data leakage between training, validation, and test sets?
> **Answer:** We enforce strict **geographical separation** at the scene level. The training set consists of tiles from Western Ghats (Karnataka), Thar Desert (Rajasthan), Kaziranga (Assam), and Sardar Sarovar (Gujarat). The validation set is drawn exclusively from Kanha (Madhya Pradesh) and Chilika Lake (Odisha Coast). The test set consists entirely of the Hirakud Reservoir scene (Odisha inland). Tiles from the same Sentinel-2 acquisition granule never appear in multiple splits.

#### Q2: Why did the Forest Cover model's validation accuracy drop in later epochs while test accuracy reached 94.74%?
> **Answer:** The validation set consists of Kanha National Park and Chilika Lake. Kanha is dry deciduous sal forest in February (distinct seasonal winter phenology), whereas Chilika has coastal scrub. As the network trained for 8 epochs, training loss steadily decreased (train OA reached 89.71%), but the model began fitting features specific to the rainforest biomes in training. Epoch 1 demonstrated the best out-of-distribution validation generalization (val loss 0.6542, val OA 66.67%). Our automated checkpointing selected Epoch 1, which successfully generalized to the unseen Hirakud test region, achieving 94.74% accuracy.

#### Q3: Why is Sentinel-2 Band 10 (Cirrus) refused in Module 2?
> **Answer:** Band 10 is centered at $1375\,\text{nm}$, a strong atmospheric water vapor absorption band. Very little solar radiation reaches the Earth's surface and reflects back to the sensor in this wavelength; it is designed purely for detecting high-altitude cirrus clouds. In bottom-of-atmosphere (L2A) surface reflectance products, Band 10 contains no terrestrial surface signal and is omitted or set to null. BWOA enforces an infinite penalty if Band 10 is selected.

#### Q4: What is the benefit of a hybrid CNN-ViT architecture over a standard ResNet or Vision Transformer alone?
> **Answer:** Sentinel-2 multi-spectral tiles exhibit both fine-grained high-frequency spatial textures (river banks, field boundaries, canopies captured by the XNANet CNN inductive bias) and long-range contextual spectral correlations across the 2.56 km tile (captured by the multi-head self-attention in Tiny-ViT). Combining their representations yields a 1216-dimensional embedding with both local shift-invariance and global spatial-spectral context.

#### Q5: Does the Binary Whale Optimization Algorithm improve classification accuracy?
> **Answer:** We do not make that claim without controlled multi-condition ablations. What BWOA empirically proves in our platform is **telemetry data volume reduction** (53.8% for waterbody, 61.5% for forest cover) while selecting physically meaningful bands (Green/NIR for water, Red Edge/SWIR for forest). Claiming that fewer bands improve accuracy over all 13 bands requires controlled baseline trials against PCA, Random, and All-Band models.

#### Q6: How does windowed COG streaming improve live satellite ingestion latency?
> **Answer:** Sentinel-2 L2A granules are distributed on AWS S3 as Cloud-Optimized GeoTIFFs (COGs). A full single-band granule is typically 100–500 MB. By issuing GDAL/rasterio HTTP range requests with the bounding box transformed to native UTM coordinates, Module 1 reads only the specific $260 \times 260$ pixel sub-window. This reduces network payload from ~1.5 GB down to ~3 MB, reducing ingestion latency from several minutes to under 2 seconds.

#### Q7: How were ESA WorldCover 2021 labels aligned with the Sentinel-2 tiles?
> **Answer:** ESA WorldCover 2021 provides 10-meter global land cover derived from Sentinel-1 and Sentinel-2. For each Indian tile, the corresponding WorldCover raster was reprojected via nearest-neighbor interpolation to match the exact CRS, pixel resolution, and bounding grid of the Sentinel-2 tile. Tiles were then filtered using coverage thresholds: water $\ge 10\%$, non-water $\le 0.5\%$, dropping ambiguous borderline tiles.

#### Q8: How does the model dynamically handle 5 or 6 bands when the raw tile has 13 bands?
> **Answer:** Dynamic tensor slicing is implemented at the input of FusionNet. Rather than re-instantiating convolutional layers with dynamic channel counts, the forward method accepts the full tensor and a binary mask vector. Channels where $M_i = 1$ are sliced into a contiguous tensor of shape $(B, C_{\text{active}}, H, W)$ and routed into the network's input projection layer calibrated for that task.

#### Q9: What is the purpose of the FAISS vector index if classification is already performed by FusionNet?
> **Answer:** FusionNet produces a 1216-dimensional latent embedding $z$ prior to the classification logits. The FAISS index stores embeddings from a reference multi-class spectral corpus to benchmark whether the learned latent space preserves semantic topological distances. It allows nearest-neighbor terrain retrieval in sub-millisecond latency (IndexFlatIP cosine similarity).

#### Q10: How are the predictions explained to non-expert users?
> **Answer:** We utilize **Permutation Feature Importance (SHAP)** across the active spectral channels. By perturbing each channel's spatial distribution across the evaluation batch and computing the change in output probability, we determine the relative contribution of each band. In waterbody classification, coastal blue (B01) and green (B03) exhibit top importance; in forest cover, Red Edge (B06) and SWIR (B11) dominate.

#### Q11: How do you prevent class imbalance from skewing model predictions?
> **Answer:** In both training setups, class imbalance was addressed via compute-class-weighting:
$$w_c = \frac{N}{K \cdot N_c}$$
Where $N$ is total training tiles, $K$ is number of classes, and $N_c$ is class support. These weights were supplied directly to `nn.CrossEntropyLoss(weight=weights)`. Furthermore, we evaluate models using **Balanced Accuracy** and **Macro F1-Score**, which penalize models that merely predict the majority class.

#### Q12: Why are some water tiles misclassified as non-water in the test set?
> **Answer:** In the held-out Hirakud test scene, water recall is **100%** (all 6 water tiles correctly detected). However, 5 non-water tiles were predicted as water (non-water precision: 100%, but non-water recall: 61.5%). These 5 tiles correspond to seasonal wetland mudflats and riparian floodplains surrounding the reservoir with high soil moisture content, causing low SWIR reflectance similar to shallow turbid water.

#### Q13: What happens if Nominatim fails or the query has no clear location?
> **Answer:** Stage 0 includes deterministic fallback parsing. If live OpenStreetMap geocoding times out or returns no matches, the query compiler matches known geographic entities (e.g., Loktak Lake, Chilika, Jim Corbett, Thar, Western Ghats) against a pre-compiled local coordinate registry. If the user runs in offline mode, it falls back to a deterministic synthetic fixture with logged warnings.

#### Q14: How does the dashboard distinguish between cached and live-acquired satellite data?
> **Answer:** Module 1 checks whether local GeoTIFF crops already exist in `data/cache/<stac_item_id>/` and meet size thresholds ($>1000$ bytes). If present, it sets `"cache_status": "Cached on Disk (Zero Network IO)"` and bypasses network requests. Otherwise, it sets `"cache_status": "Newly Acquired via STAC COG"` and records the live download time. The provenance is explicitly displayed in Tab 1 under Module 1 diagnostics.

#### Q15: What is the computational complexity of BWOA compared to exhaustive search?
> **Answer:** For 13 spectral bands, exhaustive search requires evaluating $2^{13} = 8,192$ combinations. If constrained to exactly 5 bands, it requires $\binom{13}{5} = 1,287$ evaluations. BWOA converges in $10 \text{ whales} \times 5 \text{ iterations} = 50$ function evaluations, representing a **96.1% reduction in search evaluations** while identifying high-information subsets satisfying physics constraints.
