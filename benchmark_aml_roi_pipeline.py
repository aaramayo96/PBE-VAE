"""Run the PBE-VAE ROI detector with the reference AML project configuration.

This transfers the base AML workflow into the PBE-VAE wrapper:
  1. harvest all AML latitude/longitude target ROIs and aligned DEM pixels;
  2. train a spectral-only VAE, then append weighted DEM outside the VAE;
  3. train the calibrated GBDT target detector with AML background sampling;
  4. run strict 0.95 probability inference and candidate discovery;
  5. build the PBE-VAE mosaic products.

The input flightlines, DEM, output directory, and ROIs are read directly from
the authoritative AML config, not from the Cuprite benchmark configuration.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from pbe_vae import PBEVAE
from pbe_vae.utils.config import load_config


SCRIPT_DIR = Path(__file__).resolve().parent
PBE_CONFIG = SCRIPT_DIR / "configs" / "default.yaml"
AML_CONFIG = Path("/Users/aramayo/Downloads/HSC_COmputer/config.yaml")

# AML base-workflow overrides. Keep inference strict as requested.
VAE_EPOCHS = 100
PROBABILITY_METHOD = "gmm"
CLASSIFIER = "gradient_boosting"
CALIBRATION = "isotonic"
PBE_CONFIDENCE = 0.95

# Reference 4_predict_map.py object-discovery settings supported by PBE-VAE.
MIN_CLUSTER_PIXELS = 500
OPENING_KERNEL = 11
MIN_COMPACTNESS = 0.03
DEDUP_RADIUS_PIXELS = 80
MAX_CANDIDATES_PER_TILE = 2000


def build_pbe_config(aml_cfg: dict[str, Any]) -> dict[str, Any]:
    """Map the authoritative AML settings onto the PBE-VAE configuration."""
    cfg = deepcopy(load_config(PBE_CONFIG))
    aml_processing = aml_cfg["processing"]
    aml_paths = aml_cfg["paths"]

    # Preserve AML ROI locations and mounted-volume paths exactly.
    cfg["rois"] = deepcopy(aml_cfg["rois"])
    cfg["paths"].update(
        {
            "base_dir": deepcopy(aml_paths["base_dir"]),
            "out_dir": str(aml_paths["out_dir"]),
            "dem_path": str(aml_paths["dem_path"]),
        }
    )

    # PBE-VAE's harvester gives pixels precedence over meters. Removing its
    # default pixel radius ensures AML's 100 m ROI radius is actually used.
    cfg["processing"].pop("roi_radius_pixels", None)
    cfg["processing"].update(
        {
            "roi_radius_meters": float(aml_processing["roi_radius_meters"]),
            "target_crs": str(aml_processing["target_crs"]),
            "latent_dim": int(aml_processing["vae_latent_dim"]),
            "n_clusters": int(aml_processing["n_clusters"]),
            "probability_method": PROBABILITY_METHOD,
            "vnir_range": [0.38, 1.20],
            "swir_range": [2.00, 2.50],
            "excluded_wavelength_ranges": [[1.30, 1.50], [1.75, 2.00]],
            "curvature_threshold": 4.5,
            "use_dem": bool(aml_processing["use_dem"]),
            # Both names are retained because the PBE API reads dem_weight,
            # while its persisted cluster bundle records dem_weight_ratio.
            "dem_weight": float(aml_processing["dem_weight_ratio"]),
            "dem_weight_ratio": float(aml_processing["dem_weight_ratio"]),
            "background_sample_ratio": float(aml_processing["background_sample_ratio"]),
            "background_max_pixels": int(aml_processing["background_max_pixels"]),
            "background_max_flightlines": int(aml_processing["background_max_flightlines"]),
            "background_negative_weight": float(aml_processing["background_negative_weight"]),
        }
    )
    cfg["classifier"].update(
        {
            "model": CLASSIFIER,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "min_samples_leaf": 5,
            "calibration": CALIBRATION,
        }
    )
    cfg["discovery"].update(
        {
            "conf_threshold": PBE_CONFIDENCE,
            "min_cluster_pixels": MIN_CLUSTER_PIXELS,
            "opening_kernel": OPENING_KERNEL,
            "min_mean_probability": PBE_CONFIDENCE,
            "min_compactness": MIN_COMPACTNESS,
            "candidate_sort": "max_prob",
            "dedup_radius_pixels": DEDUP_RADIUS_PIXELS,
            "max_candidates_per_tile": MAX_CANDIDATES_PER_TILE,
        }
    )
    return cfg


def require_paths(cfg: dict[str, Any]) -> None:
    """Fail early with the AML path that needs to be mounted or corrected."""
    missing = [Path(path) for path in cfg["paths"]["base_dir"] if not Path(path).exists()]
    if cfg["processing"]["use_dem"] and not Path(cfg["paths"]["dem_path"]).exists():
        missing.append(Path(cfg["paths"]["dem_path"]))
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"The AML project volumes are not available:\n{formatted}")


def main() -> dict[str, Any]:
    if not AML_CONFIG.exists():
        raise FileNotFoundError(f"Reference AML config not found: {AML_CONFIG}")

    aml_cfg = load_config(AML_CONFIG)
    cfg = build_pbe_config(aml_cfg)
    require_paths(cfg)

    output_dir = Path(cfg["paths"]["out_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = PBEVAE(PBE_CONFIG)
    model.cfg = cfg
    model.latent_dim = int(cfg["processing"]["latent_dim"])
    model.n_clusters = int(cfg["processing"]["n_clusters"])
    model.probability_method = PROBABILITY_METHOD
    model.use_dem = bool(cfg["processing"]["use_dem"])
    model.dem_weight = float(cfg["processing"]["dem_weight"])
    model.classifier_type = CLASSIFIER
    model.classifier_cfg = dict(cfg["classifier"])

    print("[+] AML reference configuration loaded")
    print(f"[+] Target ROIs: {len(cfg['rois'])}")
    print(f"[+] Flightlines: {cfg['paths']['base_dir']}")
    print(f"[+] DEM: {cfg['paths']['dem_path']}")
    print(f"[+] Output: {output_dir}")
    print(
        f"[+] HSI-only VAE: latent_dim={model.latent_dim}, "
        f"clusters={model.n_clusters}, DEM weight={model.dem_weight:.2f}"
    )

    harvest_file = output_dir / "harvested_data.npz"
    has_aligned_wavelengths = False
    if harvest_file.exists():
        with np.load(harvest_file) as harvested:
            has_aligned_wavelengths = "active_wavelengths" in harvested
    if harvest_file.exists() and has_aligned_wavelengths:
        print(f"[+] Reusing harvested AML ROI data -> {harvest_file}")
    else:
        if harvest_file.exists():
            print("[+] Existing harvest lacks canonical wavelengths; rebuilding it for header-driven alignment.")
        model.harvest(
            output_dir=output_dir,
            roi_radius_meters=float(cfg["processing"]["roi_radius_meters"]),
            use_dem=model.use_dem,
            dem_path=cfg["paths"]["dem_path"],
            target_crs=cfg["processing"]["target_crs"],
        )

    cluster_stats = model.cluster(
        n_clusters=model.n_clusters,
        epochs=VAE_EPOCHS,
        prob_method=PROBABILITY_METHOD,
        data_path=harvest_file,
        output_dir=output_dir,
    )
    classifier_stats = model.fit_classifier(
        output_dir=output_dir,
        data_path=harvest_file,
        classifier=CLASSIFIER,
        calibration=CALIBRATION,
        use_dem=model.use_dem,
        dem_weight=model.dem_weight,
    )
    detections = model.predict(
        source=cfg["paths"]["base_dir"],
        output_dir=output_dir,
        conf=PBE_CONFIDENCE,
        min_cluster_pixels=MIN_CLUSTER_PIXELS,
        opening_kernel=OPENING_KERNEL,
        min_mean_probability=PBE_CONFIDENCE,
        min_compactness=MIN_COMPACTNESS,
        candidate_sort="max_prob",
        dedup_radius=DEDUP_RADIUS_PIXELS,
        max_candidates=MAX_CANDIDATES_PER_TILE,
        save=True,
        use_dem=model.use_dem,
        dem_weight=model.dem_weight,
    )
    mosaic_info = model.mosaic(
        output_dir=output_dir,
        target_resolution=None,
        target_epsg=cfg["processing"]["target_crs"],
    )

    print("\n[+] AML ROI PBE-VAE pipeline completed.")
    print(f"[+] Candidate regions: {len(detections)}")
    print(f"[+] Results: {output_dir}")
    return {
        "cluster": cluster_stats,
        "classifier": classifier_stats,
        "detections": detections,
        "mosaic": mosaic_info,
    }


if __name__ == "__main__":
    main()
