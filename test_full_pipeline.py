from pathlib import Path
from pbe_vae import PBEVAE

# Paths & Setup
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / "configs" / "default.yaml"

# Initialize Detector Model
model = PBEVAE(CONFIG_FILE)

# 1. Harvest ROI Spectral Cubes & Align DEM (Options: roi_radius_pixels=40, use_dem=True, vnir_range=(0.38, 1.2), swir_range=(2.0, 2.5))
model.harvest(roi_radius_pixels=40, use_dem=True)

# 2. Probability Based Expert VAE & Latent Subclass Clustering (Prob Methods: "gmm", "softmax", "rbf", "student_t", "cosine")
model.cluster(n_clusters=10, epochs=100, prob_method="gmm")

# 3. Train Calibrated Target Classifier (Classifiers: "gradient_boosting", "hist_gradient_boosting", "random_forest", "xgboost", "lightgbm" | Calibration: "isotonic", "sigmoid", null)
clf_stats = model.fit_classifier(classifier="gradient_boosting", calibration="isotonic")

# 4. Flightline Pixel Inference & Morphological Target Candidate Discovery (Tuning: conf=0.85, min_cluster_pixels=500, max_cluster_pixels=50000, opening_kernel=11, dedup_radius=80)
detections = model.predict(
    conf=0.85,
    min_cluster_pixels=500,
    max_cluster_pixels=50000,
    opening_kernel=11,
    dedup_radius=80,
    save=True,
)

# 5. Multi-Flightline Georeferenced Probability Mosaic & VRT Assembly
mosaic_info = model.mosaic()

# 6. Target Discovery Cartography & High-DPI Map Publishing (Ranks Top New ROIs with Bullseye Markers & Benchmark Stars)
annot_info = model.annotate(top_n=20)
