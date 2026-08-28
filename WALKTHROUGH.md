# Probability Based Expert Variational Autoencoder Package (`pbe-vae`)
## Complete Technical Walkthrough & Architecture Guide

---

## 📖 1. Overview & Key Innovations

The **Probability Based Expert Variational Autoencoder (`pbe-vae`)** is a modular Python library designed for extreme-scale hyperspectral remote sensing (e.g. NASA AVIRIS campaign data) and digital elevation model (DEM) integration.

Instead of monolithic script files, all functions are strictly organized into single-responsibility domain modules within the `pbe_vae` package. The system features:

1. **Layer-by-Layer YAML Architecture System (`[from, number, module, args]`)**: Define custom deep neural network layers, activations, and bottleneck dimensions directly in YAML files (e.g. nano, standard, deep).
2. **Physics-Guided Expert Prior Gate**: Dynamically weights VNIR/SWIR spectral bands based on 2nd-derivative spectral curvature and variance priors before entering the latent encoder.
3. **Latent Space Clustering**: Fits Gaussian Mixture Models (GMM) on latent spectral representations to compute soft subclass probabilities.
4. **DEM Elevation Integration**: Automatically aligns and resamples DEM rasters, scaling elevation features after the VAE latent bottleneck (`[soft_cluster_probs, dem * dem_weight]`).
5. **Selectable Supervised Detectors**: Train calibrated Gradient Boosting (GBDT), Hist-GBDT, Random Forest, XGBoost, or LightGBM models with spatial Leave-One-Group-Out (LOGO) cross-validation and isotonic probability calibration.
6. **Headless PyQGIS Map Publishing**: Automatically builds multi-tile GDAL VRT mosaics, burns cartographic markers (bullseyes and stars), and renders print-ready 300 DPI PDF maps.

---

## 📁 2. Repository & Subpackage Layout

```
PBE-VAE/
├── pbe_vae/                 # Core Python Library Package
│   ├── __init__.py          # Exports PBEVAE class and library version
│   ├── detector.py          # High-level PBEVAE detector class API
│   │
│   ├── data/                # Data Loading & Geospatial Processing Subpackage
│   │   ├── __init__.py      # Clean exports for pbe_vae.data
│   │   ├── geo.py           # Geotransform, coordinates & ENVI header parser
│   │   ├── mosaic.py        # Tile reprojection, VRT assembly & mosaic builder
│   │   ├── harvester.py     # Multi-flightline ROI cube extraction
│   │   └── dataset.py       # PyTorch Dataset wrappers & background sampler
│   │
│   ├── models/              # Neural Networks & Machine Learning Subpackage
│   │   ├── __init__.py      # Clean exports for pbe_vae.models
│   │   ├── vae.py           # DynamicPBEVAE & ExpertGate layer-by-layer engine
│   │   └── classifier.py    # Selectable GBDT/RF/XGB detectors & calibration
│   │
│   ├── utils/               # Cartography, Styling & Metrics Subpackage
│   │   ├── __init__.py      # Clean exports for pbe_vae.utils
│   │   ├── cartography.py   # Multi-ring bullseye & star marker rasterizers
│   │   ├── qgis_renderer.py # Headless PyQGIS map composer & PDF exporter
│   │   ├── config.py        # YAML configuration manager
│   │   ├── metrics.py       # Separability & classification metric formulas
│   │   └── visualization.py # Diagnostic dashboards & per-flightline PDF reports
│   │
│   └── cli/                 # Terminal CLI Entrypoint
│       └── main.py          # `pbe-vae` CLI subcommands
│
├── configs/
│   ├── default.yaml         # Campaign configuration & target ROI coordinates
│   └── models/              # Layer-by-layer YAML model definitions
│       ├── pbe_vae_nano.yaml     # Ultra-fast compact preset
│       ├── pbe_vae_standard.yaml # Balanced standard preset (default)
│       └── pbe_vae_deep.yaml     # High-capacity deep representation preset
│
├── test_full_pipeline.py    # Master end-to-end deployment pipeline script
├── train.py                 # Top-level training runner script
├── predict.py               # Top-level inference runner script
├── pyproject.toml           # Packaging metadata & console script config
├── setup.py                 # Pip installation setup script
└── requirements.txt         # Package dependencies
```

---

## 🏛️ 3. Layer-by-Layer Model Definition (`[from, number, module, args]`)

Model architectures are defined in YAML configuration files using the standard `[from, number, module, args]` layer format. The dynamic parser in `pbe_vae.models.vae` instantiates and connects all layers:

### Standard Model Architecture ([`configs/models/pbe_vae_standard.yaml`](file:///Users/aramayo/Documents/DeepLearning_Python/Arizona_Project/PBE-VAE/configs/models/pbe_vae_standard.yaml))

```yaml
name: "pbe_vae_standard"
version: "v1.0"

# Hyperparameters
latent_dim: 5                  # Latent bottleneck dimension
reconstruction_loss_weight: 10.0 # MSE spectral reconstruction weight
kld_loss_weight: 0.0001        # KL divergence regularization weight
prior_loss_weight: 5.0         # BCE physics-guided prior loss weight
sparsity_loss_weight: 1.0      # Band retention penalty weight
min_bands_loss_weight: 50.0    # Minimum active band constraint penalty
min_bands_threshold: 19.0      # Minimum active spectral bands
lr: 0.001                      # Adam optimizer learning rate
clip_grad_norm: 1.0            # Gradient clipping max norm

# Expert Prior Gate
gate:
  # [from, number, module, args]
  - [-1, 1, ExpertGate, [0.5, 4, 3.0, 0.05]] # [temp, reduction_ratio, scale_init, min_gate_threshold]

# Encoder (Backbone)
encoder:
  # [from, number, module, args]
  - [-1, 1, Linear, [64]]
  - [-1, 1, LeakyReLU, [0.2]]
  - [-1, 1, BatchNorm1d, [64]]
  - [-1, 1, Linear, [32]]
  - [-1, 1, LeakyReLU, [0.2]]

# Latent Projection Layer
latent:
  # [from, number, module, args]
  - [-1, 1, Linear, [latent_dim]] # fc_mu
  - [-2, 1, Linear, [latent_dim]] # fc_logvar

# Decoder (Head)
decoder:
  # [from, number, module, args]
  - [-1, 1, Linear, [32]]
  - [-1, 1, LeakyReLU, [0.2]]
  - [-1, 1, BatchNorm1d, [32]]
  - [-1, 1, Linear, [64]]
  - [-1, 1, LeakyReLU, [0.2]]
  - [-1, 1, Linear, [input_dim]]
```

### Available Architecture Presets:
- **`pbe_vae_nano.yaml`**: 4-D latent space, compact `[32, 16]` encoder layers for high-throughput exploration.
- **`pbe_vae_standard.yaml`**: 5-D latent space, `[64, 32]` encoder layers (default balanced baseline).
- **`pbe_vae_deep.yaml`**: 8-D latent space, `[128, 64, 32]` encoder layers with batch normalization.

---

## 📊 4. Probability Conversion & Calibration Methods

The library provides selectable mathematical transformations for both **latent subclass soft probabilities** and **supervised detector probability calibration**:

### A. Latent Subclass Soft Probability Methods (`processing.probability_method`)
Converts standardised latent embeddings $\mathbf{z} \in \mathbb{R}^d$ into soft subclass probability vectors $\mathbf{p} \in \mathbb{R}^K$:

1. **`"gmm"`** (Gaussian Mixture Posterior - Default):
   $$p(c_k \mid \mathbf{z}) = \frac{\pi_k \mathcal{N}(\mathbf{z} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{z} \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

2. **`"softmax"` / `"euclidean_softmax"`** (Negative Euclidean Distance Softmax):
   $$p(c_k \mid \mathbf{z}) = \frac{\exp(-\|\mathbf{z} - \boldsymbol{\mu}_k\|_2)}{\sum_{j=1}^K \exp(-\|\mathbf{z} - \boldsymbol{\mu}_j\|_2)}$$

3. **`"rbf"`** (Radial Basis Function / Gaussian Kernel Softmax):
   $$p(c_k \mid \mathbf{z}) = \frac{\exp\left(-\frac{\|\mathbf{z} - \boldsymbol{\mu}_k\|_2^2}{2\sigma^2}\right)}{\sum_{j=1}^K \exp\left(-\frac{\|\mathbf{z} - \boldsymbol{\mu}_j\|_2^2}{2\sigma^2}\right)}$$

4. **`"student_t"` / `"dec"`** (Student-t / Deep Embedded Clustering Kernel):
   $$p(c_k \mid \mathbf{z}) = \frac{\left(1 + \|\mathbf{z} - \boldsymbol{\mu}_k\|_2^2\right)^{-1}}{\sum_{j=1}^K \left(1 + \|\mathbf{z} - \boldsymbol{\mu}_j\|_2^2\right)^{-1}}$$

5. **`"cosine"`** (Directional Cosine Similarity Softmax):
   $$p(c_k \mid \mathbf{z}) = \frac{\exp\left(\tau \frac{\mathbf{z} \cdot \boldsymbol{\mu}_k}{\|\mathbf{z}\|_2 \|\boldsymbol{\mu}_k\|_2}\right)}{\sum_{j=1}^K \exp\left(\tau \frac{\mathbf{z} \cdot \boldsymbol{\mu}_j}{\|\mathbf{z}\|_2 \|\boldsymbol{\mu}_j\|_2}\right)}$$

---

### B. Supervised Detector Probability Calibration (`classifier.calibration`)
Transforms raw decision function margins or tree ensemble votes into calibrated posterior probabilities:

1. **`"isotonic"`** (Non-parametric Isotonic Regression - Default):
   Fits a non-decreasing step function minimizing MSE on cross-validation folds. Recommended for rich datasets.
2. **`"sigmoid"`** (Platt Scaling Logistic Calibration):
   Fits a parametric sigmoid $P(Y=1 \mid \hat{y}) = \frac{1}{1 + \exp(A \hat{y} + B)}$. Recommended for smaller datasets.
3. **`null` / `"raw"`**:
   Uses direct model `predict_proba()` output without post-hoc calibration.

---

## 🌲 5. Supervised Target Detectors (`pbe_vae.models.classifier`)

The package provides selectable classifier architectures to distinguish targets from background pixels:

### Supported Detector Models:
- **`"gradient_boosting"`** (Default): Standard scikit-learn Gradient Boosting Trees.
- **`"hist_gradient_boosting"`**: Fast histogram-based gradient boosting optimized for large-scale datasets.
- **`"random_forest"`**: Multi-tree ensemble with parallelized tree training (`n_jobs=-1`).
- **`"extra_trees"`**: Extremely randomized trees classifier.
- **`"xgboost"`**: High-performance distributed gradient boosting (when `xgboost` is installed).
- **`"lightgbm"`**: Fast gradient boosting decision tree (when `lightgbm` is installed).

### Configuration in `configs/default.yaml`:

```yaml
classifier:
  model: "gradient_boosting" # Options: gradient_boosting, hist_gradient_boosting, random_forest, xgboost, lightgbm
  n_estimators: 300          # Number of boosting trees
  learning_rate: 0.05        # Learning rate step shrinkage
  max_depth: 4               # Tree depth
  subsample: 0.8             # Subsampling fraction
  min_samples_leaf: 5        # Minimum leaf samples
  calibration: "isotonic"    # Probability calibration ("isotonic", "sigmoid", null)
```

---

## 🏔️ 6. DEM Elevation Integration & Feature Pipeline

1. **Harvest Stage**: When `use_dem: true`, the DEM raster is warped to match each hyperspectral cube's bounds and resolution.
2. **Latent Representation**: The VAE extracts latent embeddings and cluster soft-membership probabilities from the spectral channels.
3. **Post-VAE Weighting**: The elevation channel is multiplied by `dem_weight` and appended to form the final feature vector:
   $$\mathbf{x}_{\text{final}} = \big[\mathbf{p}_{\text{cluster\_1}}, \dots, \mathbf{p}_{\text{cluster\_K}},\; \text{dem} \times \text{dem\_weight}\big]$$

---

## 🎯 7. Target ROI Discovery & Morphological Spatial Clustering Engine

Rather than simply plotting points, the candidate discovery and annotation pipeline deploys a spatial morphology and connected-component discovery algorithm across multi-gigabyte flightline probability rasters:

### Discovery Algorithm Workflow:
1. **Pixel Probability Inference**: The calibrated detector evaluates all non-zero hyperspectral pixels, outputting a continuous 2D probability surface $P(\text{Target} \mid \mathbf{x})$.
2. **Confidence Thresholding**: Pixels exceeding threshold (`conf=0.85`) form initial binary target masks.
3. **Morphological Filtering**:
   - **Binary Opening (`structure=11×11`)**: Strips single-pixel noise, atmospheric streaks, and sensor artifacts.
   - **Binary Hole Filling**: Consolidates solid, contiguous mineralized target cores.
4. **Connected-Component Clustering**: `scipy.ndimage.label` groups contiguous pixels into unique spatial blobs.
5. **Spatial Scale & Significance Filtering**: Discards sub-threshold noise blobs ($< 500$ pixels) and verifies cluster mean probability.
6. **Spatial Deduplication vs Benchmark ROIs**: Filters out known training targets within an 80-pixel radius, isolating genuinely novel geographical discoveries.
7. **Geodetic Coordinate Projection**: Inverts raster coordinates $(r_c, c_c)$ through GDAL affine geotransforms and WKT projections to compute precise geodetic (Latitude, Longitude) coordinates.
8. **Ranked Coordinate Export**: Outputs `new_roi_coordinates.txt` sorted by mean probability confidence and cluster area.
9. **Cartographic Marker Rasterization**:
   - **Known Benchmark ROIs**: Gold 5-point vector star markers.
   - **Discovered New ROIs**: Cyan-orange concentric multi-ring bullseyes with ranked IDs.
10. **Headless PyQGIS Map Publishing**: Renders 300 DPI publication-ready geospatial PDF maps with continuous Magma colormaps and county boundary shapefile overlays.

---

## 💻 8. Python API Usage Guide

### A. High-Level Object-Oriented Workflow (`PBEVAE`)

```python
from pbe_vae import PBEVAE

# ====================================================
# 1. Initialize Model
# ====================================================
# Load configuration (data paths, ROI coordinates, DEM settings):
model = PBEVAE("configs/default.yaml")

# (Optional: choose a specific model preset or detector type):
# model = PBEVAE("configs/default.yaml", model_cfg="configs/models/pbe_vae_nano.yaml", classifier="hist_gradient_boosting")

# ====================================================
# 2. End-to-End Training
# ====================================================
# Specify the training budget (epochs and cluster count):
results = model.train(epochs=100, n_clusters=10)

# ====================================================
# 3. Flightline Inference & Target Detection
# ====================================================
# Run target discovery with a confidence threshold:
detections = model.predict(conf=0.85, save=True)

# ====================================================
# 4. Large-Scale Georeferenced Mosaicing
# ====================================================
mosaic_info = model.mosaic()

# ====================================================
# 5. Burn Markers & Export Print-Ready PDF Maps
# ====================================================
# Render top candidate detections with cartographic styling:
annot_info = model.annotate(top_n=20)
```

---

### B. Using Individual Single-Responsibility Functions Directly

```python
# 1. Data Harvesting & Dataset Sampling
from pbe_vae.data.harvester import harvest_rois
from pbe_vae.data.dataset import harvest_background_data

# 2. Geospatial & Mosaicing
from pbe_vae.data.mosaic import assemble_probability_mosaic, warp_probability_tile
from pbe_vae.data.geo import latlon_to_pixel, pixel_to_latlon, read_roi_detections

# 3. Dynamic Model Architecture & VAE
from pbe_vae.models.vae import build_vae, train_and_extract_vae, apply_vae

# 4. Supervised Detectors & Feature Engineering
from pbe_vae.models.classifier import PBEVAEClassifier, make_features, build_base_classifier

# 5. Cartography & Map Composer
from pbe_vae.utils.cartography import burn_markers_to_raster, draw_bullseye_marker, draw_star_marker
from pbe_vae.utils.qgis_renderer import export_annotated_map_pdf
```

---

## ⚡ 7. Sequential Deployment Script (`test_full_pipeline.py`)

Run the complete 6-stage end-to-end operational pipeline from top to bottom with zero CLI argument hassles:

```bash
python test_full_pipeline.py
```

```python
from pathlib import Path
from pbe_vae import PBEVAE

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / "configs" / "default.yaml"

# Initialize Detector Model
model = PBEVAE(CONFIG_FILE)

# 1. Harvest ROI Cubes & Align DEM
model.harvest(roi_radius_meters=200, use_dem=True)

# 2. Probability Based Expert VAE & Latent Clustering
model.cluster(n_clusters=10, epochs=100, prob_method="gmm")

# 3. Train Calibrated Target Classifier
clf_stats = model.fit_classifier(classifier="gradient_boosting", calibration="isotonic")

# 4. Flightline Prediction & Target Candidate Discovery
detections = model.predict(
    conf=0.85,
    min_cluster_pixels=500,
    max_cluster_pixels=50000,
    opening_kernel=11,
    dedup_radius=80,
    save=True,
)

# 5. Georeferenced Probability Mosaic & VRT
mosaic_info = model.mosaic()

# 6. Burn Cartographic Markers & Export PDF Map
annot_info = model.annotate(top_n=20)
```

---

## 🖥️ 8. Command Line Interface (CLI)

```bash
# End-to-end model training
pbe-vae train --config configs/default.yaml --epochs 100 --clusters 10

# Multi-flightline target inference
pbe-vae predict --config configs/default.yaml --weights trained_model.pkl --conf 0.85

# Georeferenced mosaic assembly
pbe-vae mosaic --config configs/default.yaml

# Cartographic annotation & high-DPI PDF map publishing
pbe-vae annotate --config configs/default.yaml --top-n 20
```
