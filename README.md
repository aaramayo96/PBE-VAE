# PBE-VAE: Probability Based Expert Variational Autoencoder Package

`pbe-vae` is a modular, high-performance PyTorch and Machine Learning target-detection library designed for extreme-scale hyperspectral remote sensing (e.g., AVIRIS campaign data) and DEM elevation integration. It models environmental signatures, spectral curvature, and surface elevation to detect target sites without requiring heavy CNN architectures.

---

## 🌟 Key Features

- **Probability Based Expert Gate**: An unsupervised gate module that dynamically weighs and selects VNIR/SWIR spectral bands based on 2nd-derivative spectral curvature and variance priors.
- **Lightweight VAE Latent Encoding**: Compresses high-dimensional hyperspectral bands down to a low-dimensional (e.g. 5-D) latent embedding.
- **Unsupervised Latent Clustering**: Evaluates separability across GMM, MiniBatchKMeans, HDBSCAN, Birch, and BisectingKMeans.
- **Calibrated GBDT Classifier**: Gradient Boosting Trees paired with Isotonic probability calibration and Spatial Leave-One-Group-Out (LOGO) cross-validation.
- **Georeferenced Mosaicing & Cartography**: Sub-pixel accurate raster reprojection, multi-ring high-contrast cartographic marker burning, and publication-ready PyQGIS PDF map export.
- **Intuitive Python API & CLI**: Clean object-oriented model workflow in Python (`PBEVAE`) and terminal commands (`pbe-vae`).

---

## 📁 Repository Structure

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
│   │   ├── vae.py           # ExpertGate & LightweightVAE PyTorch modules
│   │   └── classifier.py    # GMM soft-clustering & GBDT classifier wrapper
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
│   └── default.yaml         # Campaign configuration & target ROI coordinates
├── test_full_pipeline.py    # Master end-to-end deployment pipeline script
├── train.py                 # Top-level training runner script
├── predict.py               # Top-level inference runner script
├── pyproject.toml           # Packaging metadata & console script config
├── setup.py                 # Pip installation setup script
└── requirements.txt         # Package dependencies
```

---

## 🚀 Quick Start

### 1. Installation

Install the package in editable mode:

```bash
cd PBE-VAE
pip install -e .
```

### 2. Python API Usage

```python
from pbe_vae import PBEVAE

# ==========================================
# 1. Load / Initialize Model from Layer-by-Layer YAML
# ==========================================
model = PBEVAE("configs/default.yaml")                                               # Default (pbe_vae_standard.yaml)
model = PBEVAE("configs/default.yaml", model_cfg="configs/models/pbe_vae_nano.yaml") # Fast compact model
model = PBEVAE("configs/default.yaml", model_cfg="configs/models/pbe_vae_deep.yaml") # Deep high-capacity model
model = PBEVAE("trained_model.pkl")                                                  # Load pretrained model bundle

# ==========================================
# 2. Train the Model
# ==========================================
# Specify training epochs and cluster count:
results = model.train(epochs=100, n_clusters=10)

# ==========================================
# 3. Predict / Detect
# ==========================================
# Run inference with desired confidence threshold:
detections = model.predict(conf=0.85, save=True)

# ==========================================
# 4. Single-spectrum detection (independent of ROI training)
# ==========================================
single_spectrum_results = model.run_single_spectrum(
    target_spectrum="target_signature.npy",
    source="/path/to/flightlines",
    methods=("ace", "sam"),
    conf={"ace": 0.02, "sam": 0.02},
    output_dir="/path/to/single_spectrum_results",
)

# ==========================================
# 5. Master ROI detection pipeline
# ==========================================
roi_results = model.run_roi_pipeline(
    train_kwargs={"epochs": 100, "n_clusters": 1},
    predict_kwargs={"conf": 0.95, "save": True},
)

# ==========================================
# 6. Georeferenced Mosaic & Annotation Maps
# ==========================================
mosaic_res = model.mosaic()
annot_res = model.annotate(top_n=20)
```

### Probability-map tuning

The ROI pipeline reads its score and candidate controls from `configs/default.yaml`.
For a one-cluster GMM, tune `processing.single_cluster_activation`,
`single_cluster_temperature`, and `single_cluster_offset`, then retrain so the
chosen settings are saved in `cluster_bundle.pkl`. Classifier calibration is set
with `classifier.calibration`. Background balance uses
`background_sample_ratio`, `background_max_pixels`, and
`background_negative_weight`. Candidate filtering uses the `discovery` settings,
including `min_mean_probability`, `min_compactness`, and `candidate_sort`.

### 3. Command Line Interface (CLI)

```bash
# Train end-to-end model
pbe-vae train --config configs/default.yaml

# Run inference across flight lines
pbe-vae predict --weights trained_model.pkl

# Reproject and build georeferenced master mosaic
pbe-vae mosaic

# Burn cartographic markers & publish annotated PDF
pbe-vae annotate --top-n 20
```

---

## 📊 Outputs & Diagnostics

1. `harvested_data.npz` & `roi_cubes/`: Extracted spectral-DEM features and spatial ROI cubes.
2. `plots/`: Multi-panel diagnostic dashboards showing PCA distributions and cluster mean spectra.
3. `trained_model.pkl` & `vae_bundle.pt`: Saved PyTorch VAE weights and calibrated GBDT model bundle.
4. `*_PROB.tif` & `*_detection_report.pdf`: Probability GeoTIFF rasters and per-flightline multi-panel PDF reports with highlighted detections.
5. `new_roi_coordinates.txt`: Georeferenced coordinates (Lat, Lon) of newly discovered target candidate sites.
6. `prob_mosaic/`: Master merged probability mosaic GeoTIFF (`prob_mosaic_masked.tif`), transparent marker overlay (`roi_annotations.tif`), and publication-quality annotated maps (`roi_annotated_map.pdf`).
