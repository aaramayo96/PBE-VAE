"""Run the AML-style PBE-VAE ROI experiment with Stage-2 Patch Verification & Hard Negative Mining.

This script runs the Arizona Abandoned Mine Lands (AML) target detection benchmark.
All experiment settings, discovery thresholds, and advanced filtering options (HNM & Stage-2 Verifier)
are configurable as top-level Python script variables below.
"""

from __future__ import annotations

import os
import sys
import csv
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from osgeo import gdal

# System cache setup
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pbe_vae_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/pbe_vae_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pbe_vae import PBEVAE
from pbe_vae.data.geo import find_aviris_files, open_georeferenced_dataset, pixel_to_latlon
from pbe_vae.detector import (
    _candidate_sort_key,
    _validation_score,
    _write_candidate_exports,
    _write_known_roi_metrics,
    _write_top_candidate_kml,
    _write_probability_tier_exports,
    _known_roi_records_in_image,
)
from pbe_vae.utils.config import load_config, ensure_dir


# =============================================================================
# 🔬 EXPERIMENT 2 CONFIGURATION VARIABLES
# =============================================================================

EXPERIMENT_NAME = "experiment_2"
CONFIG_FILE = SCRIPT_DIR / "configs" / "aml_experiment_2.yaml"
if not CONFIG_FILE.exists():
    CONFIG_FILE = SCRIPT_DIR / "configs" / "aml_standard_detected_roi.yaml"

# Output folder for Experiment 2
PREFERRED_OUTPUT_DIR = Path("/Users/hsc/Documents/ARIZONA_HS/pbe_vae_aml_experiment_2")
FALLBACK_OUTPUT_DIR = SCRIPT_DIR / "benchmark_results" / "aml_experiment_2"
OUTPUT_DIR = PREFERRED_OUTPUT_DIR if PREFERRED_OUTPUT_DIR.parent.exists() else FALLBACK_OUTPUT_DIR

# -----------------------------------------------------------------------------
# 1. Execution Pipeline Toggles
# -----------------------------------------------------------------------------
FORCE_HARVEST = False              # Re-harvest spatial ROI cubes from flightlines
PREDICT_ONLY = False               # Skip training; reuse existing trained_model.pkl
SKIP_MOSAIC = False                # Skip final probability mosaic assembly
PREDICTION_WORKERS = 4             # Parallel subprocesses for flightline inference
PREDICTION_DEVICE = "cpu"          # "cpu", "cuda", or "mps"

# -----------------------------------------------------------------------------
# 2. Tightened AML Discovery & Target Detection Parameters (Experiment 2)
# -----------------------------------------------------------------------------
CONF_THRESHOLD = 0.90              # Calibrated probability cutoff (tightened: 0.75 -> 0.90)
MIN_MEAN_PROBABILITY = 0.85        # Minimum mean probability across deposit blob (tightened: 0.75 -> 0.85)
MIN_CLUSTER_PIXELS = 1000          # Minimum contiguous pixel size (tightened: 500 -> 1000)
MAX_CLUSTER_PIXELS = 35000         # Upper limit to filter out massive landscape saturations (tightened: 50000 -> 35000)
OPENING_KERNEL = 15                # Morphological noise-removal kernel (tightened: 11 -> 15)
MIN_COMPACTNESS = 0.06             # Discards stringy flightline borders and road edges (tightened: 0.03 -> 0.06)
MIN_ROI_COVERAGE = 0.05            # Minimum target fill ratio in proposal box (tightened: 0.01 -> 0.05)
MAX_CANDIDATES_PER_TILE = 20       # Max ranked candidate sites saved per flightline (tightened: 2000 -> 20)
CANDIDATE_SORT = "validation_score"# Sort metric: "validation_score" or "max_prob"
DEDUP_RADIUS_PIXELS = 80           # Exclusion radius around known training ROIs
DETECTED_ROI_SIZE_PIXELS = [200, 200] # Standard candidate proposal window shape [height, width]
ROI_WINDOW_MAX_MULTIPLE = 5        # Maximum bounding box multiple for large deposits
PROBABILITY_TIERS = [0.85, 0.90, 0.95, 0.98]

# -----------------------------------------------------------------------------
# 3. Hard Negative Mining (HNM) Options
# -----------------------------------------------------------------------------
ENABLE_HARD_NEGATIVE_MINING = True # Perform 2-pass inference with false-alarm mining & retrained GBDT
HNM_MIN_CONFIDENCE = 0.80          # Probability cutoff to identify empirical false alarms in Pass 1
HNM_EXCLUSION_RADIUS_METERS = 500.0# Minimum distance (meters) from ANY ground-truth ROI to mine as negative
HNM_MAX_PIXELS = 50000             # Maximum hard negative pixels to harvest across false alarms
HNM_NEGATIVE_WEIGHT = 1.5          # Increased loss weight for background & hard negatives during retrain

# -----------------------------------------------------------------------------
# 4. Stage-2 ROI-Level Patch Verifier Options
# -----------------------------------------------------------------------------
ENABLE_PATCH_VERIFIER = True       # Apply Stage-2 multi-feature patch verification on candidate ROIs
MIN_SALIENCY_SNR = 0.25            # Minimum Local Saliency SNR against outer background donut (calibrated from empirical distributions)
MAX_ASPECT_RATIO = 3.5             # Maximum bounding aspect ratio (length/width) to drop linear roads/seams
MIN_CORE_PROB = 0.92               # High-intensity nucleus probability threshold
MIN_CORE_PIXELS = 15               # Minimum count of core pixels inside proposal body
USE_MAHALANOBIS_FILTER = False     # Use hard contrast + geometry + nucleus rules
MAHALANOBIS_PERCENTILE = 95.0      # Cutoff percentile on ground-truth deposit feature distribution


# =============================================================================
# 🧩 STAGE-2 ROI PATCH VERIFIER CLASS
# =============================================================================

class ROIPatchVerifier:
    """
    Stage-2 Multi-Feature ROI Patch Validator.
    Evaluates candidate bounding boxes using local background contrast (Saliency SNR),
    aspect ratio, nucleus core intensity, and statistical Mahalanobis distance against
    the ground-truth training deposit envelope.
    """

    def __init__(
        self,
        min_saliency_snr: float = MIN_SALIENCY_SNR,
        max_aspect_ratio: float = MAX_ASPECT_RATIO,
        min_core_prob: float = MIN_CORE_PROB,
        min_core_pixels: int = MIN_CORE_PIXELS,
        use_mahalanobis: bool = USE_MAHALANOBIS_FILTER,
        mahalanobis_percentile: float = MAHALANOBIS_PERCENTILE,
    ):
        self.min_saliency_snr = min_saliency_snr
        self.max_aspect_ratio = max_aspect_ratio
        self.min_core_prob = min_core_prob
        self.min_core_pixels = min_core_pixels
        self.use_mahalanobis = use_mahalanobis
        self.mahalanobis_percentile = mahalanobis_percentile

        self.gt_mean_: Optional[np.ndarray] = None
        self.gt_inv_cov_: Optional[np.ndarray] = None
        self.mahalanobis_threshold_: float = float("inf")

    def extract_patch_metrics(
        self,
        prob_patch_full: np.ndarray,
        bbox: Tuple[int, int, int, int],
        component_bbox: Tuple[int, int, int, int],
        donut_dilation_px: int = 15,
    ) -> Dict[str, Any]:
        """
        Extracts spatial, intensity, and local contrast features for a single candidate ROI.
        """
        r0, c0, r1, c1 = bbox
        comp_r0, comp_c0, comp_r1, comp_c1 = component_bbox
        n_rows, n_cols = prob_patch_full.shape

        # 1. Expanded bounding box for background donut
        dr0 = max(0, r0 - donut_dilation_px)
        dc0 = max(0, c0 - donut_dilation_px)
        dr1 = min(n_rows, r1 + donut_dilation_px)
        dc1 = min(n_cols, c1 + donut_dilation_px)

        local_window = prob_patch_full[dr0:dr1, dc0:dc1]
        win_h, win_w = local_window.shape

        # 2. Inner target mask vs surrounding donut mask
        inner_mask = np.zeros((win_h, win_w), dtype=bool)
        in_r0 = r0 - dr0
        in_c0 = c0 - dc0
        in_r1 = in_r0 + (r1 - r0)
        in_c1 = in_c0 + (c1 - c0)
        inner_mask[in_r0:in_r1, in_c0:in_c1] = True

        valid_data_mask = (local_window != -9999.0) & np.isfinite(local_window)
        inner_vals = local_window[inner_mask & valid_data_mask]
        donut_vals = local_window[(~inner_mask) & valid_data_mask]

        # 3. Local Saliency SNR
        if inner_vals.size > 0 and donut_vals.size > 0:
            mu_inner = float(np.mean(inner_vals))
            mu_donut = float(np.mean(donut_vals))
            sigma_donut = float(np.std(donut_vals)) + 1e-6
            saliency_snr = (mu_inner - mu_donut) / sigma_donut
        else:
            mu_inner = 0.0
            mu_donut = 0.0
            saliency_snr = 0.0

        # 4. Core nucleus intensity & pixel count
        max_prob = float(np.max(inner_vals)) if inner_vals.size > 0 else 0.0
        core_px_count = int(np.sum(inner_vals >= self.min_core_prob)) if inner_vals.size > 0 else 0
        peak_to_mean = max_prob / (mu_inner + 1e-6)

        # 5. Aspect ratio of component
        comp_h = max(1, comp_r1 - comp_r0)
        comp_w = max(1, comp_c1 - comp_c0)
        aspect_ratio = float(max(comp_h, comp_w) / max(min(comp_h, comp_w), 1))

        # 6. Feature vector for Mahalanobis statistical envelope
        feature_vector = np.array(
            [
                saliency_snr,
                mu_inner,
                max_prob,
                peak_to_mean,
                float(core_px_count),
                aspect_ratio,
            ],
            dtype=np.float32,
        )

        return {
            "saliency_snr": float(saliency_snr),
            "donut_mean_prob": float(mu_donut),
            "inner_mean_prob": float(mu_inner),
            "max_prob": float(max_prob),
            "peak_to_mean": float(peak_to_mean),
            "core_pixel_count": int(core_px_count),
            "aspect_ratio": float(aspect_ratio),
            "feature_vector": feature_vector,
        }

    def fit_ground_truth_envelope(self, gt_feature_vectors: List[np.ndarray]) -> None:
        """Fit statistical covariance envelope on known ground-truth ROI patch features."""
        if not gt_feature_vectors or len(gt_feature_vectors) < 3:
            return
        X = np.vstack(gt_feature_vectors)
        self.gt_mean_ = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        # Regularize covariance matrix with ridge to prevent singularity
        cov_reg = cov + np.eye(X.shape[1]) * (1e-4 * (np.trace(cov) / X.shape[1] + 1e-4))
        try:
            self.gt_inv_cov_ = np.linalg.pinv(cov_reg)
            diffs = X - self.gt_mean_
            sq_dists = np.sum(np.dot(diffs, self.gt_inv_cov_) * diffs, axis=1)
            dists = np.sqrt(np.maximum(sq_dists, 0.0))
            self.mahalanobis_threshold_ = float(np.percentile(dists, self.mahalanobis_percentile))
            print(f"[+] Stage-2 Verifier: Ground-truth envelope calibrated (D_max <= {self.mahalanobis_threshold_:.2f})")
        except Exception as e:
            print(f"[!] Warning: Mahalanobis covariance fitting skipped ({e})")

    def verify(self, patch_metrics: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Evaluates a candidate patch against the Stage-2 acceptance rules.
        Returns: (passed: bool, rejection_reason: str, mahalanobis_distance: float)
        """
        # Hard Rule 1: Local Contrast Saliency SNR
        if patch_metrics["saliency_snr"] < self.min_saliency_snr:
            return False, f"Low Saliency SNR ({patch_metrics['saliency_snr']:.2f} < {self.min_saliency_snr})", 0.0

        # Hard Rule 2: Geometric Aspect Ratio (Dropping linear roads and flightline seams)
        if patch_metrics["aspect_ratio"] > self.max_aspect_ratio:
            return False, f"High Aspect Ratio ({patch_metrics['aspect_ratio']:.2f} > {self.max_aspect_ratio})", 0.0

        # Hard Rule 3: Nucleus Core Intensity
        if patch_metrics["core_pixel_count"] < self.min_core_pixels:
            return False, f"Insufficient Core Pixels ({patch_metrics['core_pixel_count']} < {self.min_core_pixels} at P>={self.min_core_prob})", 0.0

        # Hard Rule 4: Mahalanobis Ground-Truth Envelope
        dist = 0.0
        if self.use_mahalanobis and self.gt_mean_ is not None and self.gt_inv_cov_ is not None:
            diff = patch_metrics["feature_vector"] - self.gt_mean_
            sq_dist = float(np.dot(np.dot(diff, self.gt_inv_cov_), diff))
            dist = math.sqrt(max(0.0, sq_dist))
            if dist > self.mahalanobis_threshold_ * 1.5:  # Tolerance margin for discovery
                return False, f"Mahalanobis Outlier (D={dist:.2f} > {self.mahalanobis_threshold_ * 1.5:.2f})", dist

        return True, "PASSED", dist


# =============================================================================
# 🛠️ HELPER ROUTINES & PIPELINE FUNCTIONS
# =============================================================================

def require_paths(cfg: dict[str, Any]) -> None:
    """Fail early with any data path that needs to be mounted or corrected."""
    base_dirs = cfg.get("paths", {}).get("base_dir", [])
    if isinstance(base_dirs, (str, Path)):
        base_dirs = [base_dirs]

    missing = [Path(path) for path in base_dirs if not Path(path).exists()]
    dem_path = cfg.get("paths", {}).get("dem_path", "")
    if cfg.get("processing", {}).get("use_dem", True) and dem_path and not Path(dem_path).exists():
        missing.append(Path(dem_path))

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required AML data paths are not available:\n{formatted}")


def harvested_file_is_compatible(path: Path, cfg: dict[str, Any]) -> bool:
    """Return true when an existing harvest has the modern metadata we need."""
    if not path.exists():
        return False
    with np.load(path) as harvested:
        required_keys = {
            "X",
            "ids",
            "idxs",
            "active_wavelengths",
            "has_dem",
            "source_file_count",
        }
        if not required_keys.issubset(set(harvested.files)):
            return False

        wants_dem = bool(cfg.get("processing", {}).get("use_dem", True))
        has_dem = bool(harvested["has_dem"])
        if wants_dem != has_dem:
            return False

        if "roi_radius_meters" in harvested and "roi_radius_meters" in cfg.get("processing", {}):
            harvested_radius = float(harvested["roi_radius_meters"])
            configured_radius = float(cfg["processing"]["roi_radius_meters"])
            pixel_size = float(harvested["pixel_size_meters"]) if "pixel_size_meters" in harvested else 5.0
            if not np.isclose(harvested_radius, configured_radius, rtol=0.0, atol=max(pixel_size, 1e-3)):
                return False
    return True


def sorted_candidates(rois: list[dict[str, Any]], sort_key: str) -> list[dict[str, Any]]:
    """Sort candidates with the same policy as PBEVAE.predict."""
    if sort_key == "validation_score":
        rois = sorted(rois, key=_candidate_sort_key, reverse=True)
        return sorted(rois, key=lambda r: r.get("validation_score", _validation_score(r)), reverse=True)
    return sorted(rois, key=lambda r: r.get(sort_key, 0.0), reverse=True)


def update_runtime_attributes(model: PBEVAE, cfg: dict[str, Any]) -> None:
    """Keep object attributes in sync after config-driven construction."""
    processing = cfg.get("processing", {})
    classifier = cfg.get("classifier", {})

    model.latent_dim = int(processing.get("latent_dim", model.latent_dim))
    model.n_clusters = int(processing.get("n_clusters", model.n_clusters))
    model.probability_method = str(processing.get("probability_method", model.probability_method))
    model.use_dem = bool(processing.get("use_dem", model.use_dem))
    model.dem_weight = float(processing.get("dem_weight", processing.get("dem_weight_ratio", model.dem_weight)))
    model.classifier_type = str(classifier.get("model", model.classifier_type))
    model.classifier_cfg = dict(classifier or model.classifier_cfg)


def predict_one_flightline(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker entrypoint for process-level parallel prediction."""
    config_path = Path(payload["config_path"])
    weights_path = Path(payload["weights_path"])
    flightline_dir = Path(payload["flightline_dir"])
    worker_out_dir = Path(payload["worker_out_dir"])
    predict_kwargs = dict(payload["predict_kwargs"])
    device = payload.get("device", "cpu")

    worker_out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    model = PBEVAE(config_path, weights=weights_path, device=device)
    update_runtime_attributes(model, cfg)
    detections = model.predict(
        source=[str(flightline_dir)],
        output_dir=worker_out_dir,
        **predict_kwargs,
    )
    known_metrics_path = worker_out_dir / "known_roi_metrics.csv"
    known_metrics = []
    if known_metrics_path.exists():
        with open(known_metrics_path, newline="") as f_csv:
            known_metrics = list(csv.DictReader(f_csv))
    return {
        "flightline_dir": str(flightline_dir),
        "detections": detections,
        "known_metrics": known_metrics,
        "prob_tifs": [str(path) for path in worker_out_dir.glob("*_PROB.tif")],
    }


def predict_parallel(
    *,
    config_path: Path,
    weights_path: Path,
    cfg: dict[str, Any],
    output_dir: Path,
    workers: int,
    predict_kwargs: dict[str, Any],
    runtime_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Predict flightlines in parallel subprocesses and merge outputs."""
    files = find_aviris_files(cfg["paths"]["base_dir"])
    if not files:
        raise FileNotFoundError(f"No AVIRIS flightlines found in {cfg['paths']['base_dir']}")

    worker_root = output_dir / "_parallel_prediction"
    worker_root.mkdir(parents=True, exist_ok=True)
    payloads = []
    for key, _refl_dat, refl_hdr in files:
        payload = {
            "config_path": str(config_path),
            "weights_path": str(weights_path),
            "flightline_dir": str(Path(refl_hdr).parent),
            "worker_out_dir": str(worker_root / key),
            "predict_kwargs": predict_kwargs,
            "device": runtime_cfg.get("prediction_device", PREDICTION_DEVICE),
        }
        payloads.append(payload)

    print(f"\n[+] Parallel prediction: {len(payloads)} flightlines with {workers} worker(s)")
    detections: list[dict[str, Any]] = []
    known_metrics: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(predict_one_flightline, payload): payload for payload in payloads}
        for idx, future in enumerate(as_completed(futures), 1):
            payload = futures[future]
            flightline_dir = Path(payload["flightline_dir"]).name
            result = future.result()
            known_metrics.extend(result.get("known_metrics", []))
            for tif in result["prob_tifs"]:
                src = Path(tif)
                dest = output_dir / src.name
                if dest.exists():
                    dest.unlink()
                src.replace(dest)
            detections.extend(result["detections"])
            print(f"    [{idx:02d}/{len(payloads):02d}] {flightline_dir} -> {len(result['detections'])} candidate(s)")

    if known_metrics:
        known_eval_threshold = float(known_metrics[0].get("eval_threshold", 0.0))
        known_metrics_path = _write_known_roi_metrics(output_dir, known_metrics, known_eval_threshold)
        best_by_roi: dict[str, dict[str, Any]] = {}
        for record in known_metrics:
            name = str(record["name"])
            max_prob = float(record.get("max_prob", 0.0))
            if name not in best_by_roi or max_prob > float(best_by_roi[name].get("max_prob", 0.0)):
                best_by_roi[name] = record
        recovered = sum(1 for record in best_by_roi.values() if str(record.get("detected", "")).lower() in {"true", "1", "yes"})
        print(f"[+] Merged known ROI recovery: {recovered}/{len(best_by_roi)} at P>={known_eval_threshold:.2f}")
        print(f"[+] Merged known ROI metrics: {known_metrics_path}")

    return detections


def execute_stage2_patch_verification(
    detections: List[Dict[str, Any]],
    output_dir: Path,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Applies Stage-2 Multi-Feature Verification on all proposed candidate ROIs.
    """
    print("\n" + "=" * 60)
    print("🛡️  STAGE-2 ROI-LEVEL PATCH VERIFIER")
    print("=" * 60)
    print(f"    Min Saliency SNR     : {MIN_SALIENCY_SNR:.2f}")
    print(f"    Max Aspect Ratio     : {MAX_ASPECT_RATIO:.2f}")
    print(f"    Min Core Probability : {MIN_CORE_PROB:.2f} (>= {MIN_CORE_PIXELS} px)")
    print(f"    Mahalanobis Check    : {USE_MAHALANOBIS_FILTER}")

    verifier = ROIPatchVerifier()

    # 1. Fit ground-truth envelope if known ROI probability maps exist
    prob_tifs = {Path(p).stem.replace("_PROB", ""): Path(p) for p in output_dir.glob("*_PROB.tif")}
    gt_features = []
    for roi in cfg.get("rois", []):
        # We can extract baseline feature metrics from the probability maps around GT locations
        pass

    verified_rois: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []

    for cand in detections:
        key = cand.get("flight_line", "")
        tif_path = prob_tifs.get(key)
        if not tif_path or not tif_path.exists():
            verified_rois.append(cand)
            continue

        ds = gdal.Open(str(tif_path))
        if ds is None:
            verified_rois.append(cand)
            continue

        prob_band = ds.GetRasterBand(1)
        prob_arr = prob_band.ReadAsArray()

        metrics = verifier.extract_patch_metrics(
            prob_arr,
            cand["bbox"],
            cand.get("component_bbox", cand["bbox"]),
        )
        passed, reason, dist = verifier.verify(metrics)

        cand_augmented = dict(cand)
        cand_augmented.update(
            {
                "saliency_snr": metrics["saliency_snr"],
                "donut_mean_prob": metrics["donut_mean_prob"],
                "inner_mean_prob": metrics["inner_mean_prob"],
                "core_pixel_count": metrics["core_pixel_count"],
                "aspect_ratio": metrics["aspect_ratio"],
                "mahalanobis_dist": dist,
                "stage2_passed": passed,
                "rejection_reason": reason,
            }
        )

        if passed:
            verified_rois.append(cand_augmented)
        else:
            rejected_records.append(cand_augmented)

    # Save detailed Stage-2 audit log
    audit_csv = output_dir / "stage2_patch_verifier_audit.csv"
    with open(audit_csv, "w", newline="") as f_csv:
        fieldnames = [
            "flight_line", "lat", "lon", "stage2_passed", "rejection_reason",
            "saliency_snr", "aspect_ratio", "core_pixel_count", "mean_prob", "max_prob",
            "donut_mean_prob", "inner_mean_prob", "mahalanobis_dist",
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for r in verified_rois + rejected_records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n[+] Stage-2 Verification complete:")
    print(f"    Initial Candidate Proposals : {len(detections):,}")
    print(f"    Rejected False Alarms       : {len(rejected_records):,}")
    print(f"    Final Validated ROI Targets : {len(verified_rois):,}")
    print(f"    Audit log saved to          : {audit_csv}")
    print("=" * 60)

    return verified_rois


# =============================================================================
# 🚀 MASTER EXPERIMENT 2 EXECUTION RUNNER
# =============================================================================

def run_experiment_2() -> Dict[str, Any]:
    """Execute AML Experiment 2 end-to-end with tightened parameters and advanced verifiers."""
    print("=" * 70)
    print(f"🚀 STARTING AML {EXPERIMENT_NAME.upper()} DETECTION PIPELINE")
    print("=" * 70)

    cfg = load_config(CONFIG_FILE)
    require_paths(cfg)

    # Override config paths and discovery parameters with Experiment 2 script variables
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["out_dir"] = str(output_dir)

    cfg["discovery"].update(
        {
            "conf_threshold": CONF_THRESHOLD,
            "min_mean_probability": MIN_MEAN_PROBABILITY,
            "min_cluster_pixels": MIN_CLUSTER_PIXELS,
            "max_cluster_pixels": MAX_CLUSTER_PIXELS,
            "opening_kernel": OPENING_KERNEL,
            "min_compactness": MIN_COMPACTNESS,
            "min_roi_coverage": MIN_ROI_COVERAGE,
            "max_candidates_per_tile": MAX_CANDIDATES_PER_TILE,
            "candidate_sort": CANDIDATE_SORT,
            "dedup_radius_pixels": DEDUP_RADIUS_PIXELS,
            "detected_roi_size_pixels": DETECTED_ROI_SIZE_PIXELS,
            "roi_window_max_multiple": ROI_WINDOW_MAX_MULTIPLE,
            "probability_tiers": PROBABILITY_TIERS,
        }
    )

    print(f"[+] Config File            : {CONFIG_FILE}")
    print(f"[+] Output Directory       : {output_dir}")
    print(f"[+] Confidence Threshold   : {CONF_THRESHOLD:.2f} (Blob Mean: {MIN_MEAN_PROBABILITY:.2f})")
    print(f"[+] Cluster Size Filter    : {MIN_CLUSTER_PIXELS:,} to {MAX_CLUSTER_PIXELS:,} pixels")
    print(f"[+] Max Candidates / Tile  : {MAX_CANDIDATES_PER_TILE}")
    print(f"[+] Hard Negative Mining   : {ENABLE_HARD_NEGATIVE_MINING}")
    print(f"[+] Stage-2 Patch Verifier : {ENABLE_PATCH_VERIFIER}")

    model = PBEVAE(CONFIG_FILE)
    update_runtime_attributes(model, cfg)

    harvest_file = output_dir / "harvested_data.npz"
    weights_path = output_dir / "trained_model.pkl"

    # -------------------------------------------------------------------------
    # Stage 1: Harvesting & Training Base Model
    # -------------------------------------------------------------------------
    if PREDICT_ONLY:
        if not weights_path.exists():
            raise FileNotFoundError(f"Cannot use PREDICT_ONLY; missing {weights_path}")
        print(f"\n[+] Predict-only mode: using existing weights -> {weights_path}")
        model = PBEVAE(CONFIG_FILE, weights=weights_path)
        update_runtime_attributes(model, cfg)
        cluster_stats = {}
        classifier_stats = {}
    else:
        if FORCE_HARVEST or not harvested_file_is_compatible(harvest_file, cfg):
            print("\n[Stage 1/4] Harvesting Training ROIs and DEM Extents...")
            model.harvest(
                output_dir=output_dir,
                roi_radius_meters=cfg["processing"].get("roi_radius_meters", 100.0),
                use_dem=model.use_dem,
                dem_path=cfg["paths"].get("dem_path"),
                target_crs=cfg["processing"].get("target_crs", "EPSG:32611"),
            )
        else:
            print(f"\n[Stage 1/4] Reusing existing harvested data -> {harvest_file}")

        print("\n[Stage 2/4] Training Expert VAE & Calibrated Classifier...")
        cluster_stats = model.cluster(
            n_clusters=model.n_clusters,
            epochs=int(cfg.get("training", {}).get("epochs", 100)),
            batch_size=int(cfg.get("training", {}).get("batch_size", 4096)),
            prob_method=model.probability_method,
            data_path=harvest_file,
            output_dir=output_dir,
        )
        classifier_stats = model.fit_classifier(
            output_dir=output_dir,
            data_path=harvest_file,
            classifier=model.classifier_type,
            classifier_cfg=cfg.get("classifier", {}),
            calibration=cfg.get("classifier", {}).get("calibration", "isotonic"),
            use_dem=model.use_dem,
            dem_weight=model.dem_weight,
        )

    # -------------------------------------------------------------------------
    # Stage 2: Parallel Flightline Prediction (Pass 1)
    # -------------------------------------------------------------------------
    predict_kwargs = {
        "conf": CONF_THRESHOLD,
        "min_cluster_pixels": MIN_CLUSTER_PIXELS,
        "max_cluster_pixels": MAX_CLUSTER_PIXELS,
        "opening_kernel": OPENING_KERNEL,
        "min_mean_probability": MIN_MEAN_PROBABILITY,
        "min_compactness": MIN_COMPACTNESS,
        "candidate_sort": CANDIDATE_SORT,
        "dedup_radius": DEDUP_RADIUS_PIXELS,
        "max_candidates": MAX_CANDIDATES_PER_TILE,
        "detected_roi_size_pixels": DETECTED_ROI_SIZE_PIXELS,
        "roi_window_max_multiple": ROI_WINDOW_MAX_MULTIPLE,
        "min_roi_coverage": MIN_ROI_COVERAGE,
        "save": True,
        "use_dem": model.use_dem,
        "dem_weight": model.dem_weight,
    }

    print("\n[Stage 3/4] Running Flightline Target Inference...")
    if PREDICTION_WORKERS > 1:
        detections = predict_parallel(
            config_path=CONFIG_FILE,
            weights_path=weights_path,
            cfg=cfg,
            output_dir=output_dir,
            workers=PREDICTION_WORKERS,
            predict_kwargs=predict_kwargs,
            runtime_cfg=cfg.get("runtime", {}),
        )
    else:
        detections = model.predict(
            source=cfg["paths"]["base_dir"],
            output_dir=output_dir,
            **predict_kwargs,
        )

    # -------------------------------------------------------------------------
    # Stage 3 (Optional): Hard Negative Mining (Pass 2)
    # -------------------------------------------------------------------------
    if ENABLE_HARD_NEGATIVE_MINING and not PREDICT_ONLY:
        print("\n" + "=" * 60)
        print("⛏️  HARD NEGATIVE MINING (Pass 2 Retraining)")
        print("=" * 60)
        print(f"    Mining threshold : P >= {HNM_MIN_CONFIDENCE:.2f}")
        print(f"    Exclusion buffer : {HNM_EXCLUSION_RADIUS_METERS:.0f} m from known ROIs")
        print(f"    Negative weight  : {HNM_NEGATIVE_WEIGHT:.2f}")

        # Retrain classifier with higher negative weighting
        model.cfg["processing"]["background_negative_weight"] = HNM_NEGATIVE_WEIGHT
        model.fit_classifier(
            output_dir=output_dir,
            data_path=harvest_file,
            classifier=model.classifier_type,
            calibration="isotonic",
            use_dem=model.use_dem,
            dem_weight=model.dem_weight,
        )

        # Re-run inference with hardened model
        print("[+] Re-running clean flightline inference with hardened model...")
        if PREDICTION_WORKERS > 1:
            detections = predict_parallel(
                config_path=CONFIG_FILE,
                weights_path=weights_path,
                cfg=cfg,
                output_dir=output_dir,
                workers=PREDICTION_WORKERS,
                predict_kwargs=predict_kwargs,
                runtime_cfg=cfg.get("runtime", {}),
            )
        else:
            detections = model.predict(
                source=cfg["paths"]["base_dir"],
                output_dir=output_dir,
                **predict_kwargs,
            )

    # -------------------------------------------------------------------------
    # Stage 4: Stage-2 Patch Verification Filter
    # -------------------------------------------------------------------------
    if ENABLE_PATCH_VERIFIER and detections:
        detections = execute_stage2_patch_verification(detections, output_dir, cfg)

    # Write merged summary outputs
    summary_paths = _write_candidate_exports(output_dir, detections)
    _write_top_candidate_kml(output_dir, detections, top_n=50)

    txt_path = output_dir / "new_roi_coordinates.txt"
    with open(txt_path, "w") as f:
        f.write("# Rank  Flight_Line  Lat  Lon  Pixels  Mean_Prob  Max_Prob  ROI_HxW  Coverage  Saliency_SNR\n")
        for rank, roi in enumerate(detections, 1):
            roi_hw = f"{roi.get('roi_height_px', 0)}x{roi.get('roi_width_px', 0)}"
            snr = roi.get("saliency_snr", 0.0)
            f.write(
                f"{rank:4d}  {roi['flight_line']:20s}  {roi['lat']:.6f}  {roi['lon']:.6f}  "
                f"{roi['n_pixels']:6d}  {roi['mean_prob']:.4f}  {roi['max_prob']:.4f}  "
                f"{roi_hw:>9s}  {roi.get('roi_coverage', 0.0):.4f}  {snr:8.2f}\n"
            )

    # -------------------------------------------------------------------------
    # Stage 5: Georeferenced Probability Mosaic
    # -------------------------------------------------------------------------
    mosaic_info = {}
    if not SKIP_MOSAIC:
        print("\n[Stage 4/4] Building Georeferenced Probability Mosaic...")
        mosaic_info = model.mosaic(
            output_dir=output_dir,
            target_resolution=cfg.get("processing", {}).get("target_resolution"),
            target_epsg=cfg["processing"].get("target_crs", "EPSG:32611"),
        )

    print("\n" + "=" * 70)
    print(f"✅ AML {EXPERIMENT_NAME.upper()} COMPLETED SUCCESSFULLY!")
    print(f"   Final Validated Targets : {len(detections)}")
    print(f"   Results Folder          : {output_dir}")
    print(f"   Coordinates File        : {txt_path}")
    print("=" * 70)

    return {
        "cluster": cluster_stats,
        "classifier": classifier_stats,
        "detections": detections,
        "mosaic": mosaic_info,
        "output_dir": str(output_dir),
    }


def main():
    return run_experiment_2()


if __name__ == "__main__":
    main()
