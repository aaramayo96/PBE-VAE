"""Run the master ROI-trained PBE-VAE detection pipeline."""

from pathlib import Path
import numpy as np

from pbe_vae import PBEVAE


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / "configs" / "default.yaml"
CUPRITE_DIR = "/Users/aramayo/Documents/DeepLearning_Python/HSI/7.Cuprite_data/PROSPECTIR/!REFLECTANCE"
DEM_OUTPUT_DIR = SCRIPT_DIR / "benchmark_results" / "cuprite_roi_pipeline"
NO_DEM_OUTPUT_DIR = SCRIPT_DIR / "benchmark_results" / "cuprite_roi_pipeline_no_dem"
DEM_PATH = SCRIPT_DIR / "data" / "cuprite_dem" / "cuprite_usgs_1m.vrt"

# ROI-pipeline variables.
CUPRITE_ROIS = [
    {"name": "Cuprite_ROI_01", "x": 481155, "y": 4153002},
    {"name": "Cuprite_ROI_02", "x": 481205.45, "y": 4152978.75},
]
ROI_RADIUS_PIXELS = 4
PBE_CONFIDENCE = 0.90
ACE_CONFIDENCE = 0.02  # ACE scores below this threshold are hidden in the detection map.
SAM_CONFIDENCE = 0.06  # SAM angles above this threshold are hidden in the detection map.
N_CLUSTERS = 1
VAE_EPOCHS = 300
PROBABILITY_METHOD = "gmm"
SINGLE_CLUSTER_ACTIVATION = "sigmoid"  # "sigmoid" or "exp_density"
SINGLE_CLUSTER_TEMPERATURE = 1.0
SINGLE_CLUSTER_OFFSET = 0.0
CALIBRATION = "isotonic"                # "isotonic", "sigmoid", or None
DEM_WEIGHT = 1  # Maximum DEM influence: a poor terrain match can reduce a spectral score by up to 80%.
DEM_SIMILARITY_WIDTH = 1.0  # DEM z-score width: 1.0 means one target-ROI std is a close terrain match.
RUN_WITH_DEM = True
RUN_WITHOUT_DEM = False


def run_pipeline(*, use_dem: bool, output_dir: Path):
    """Train and predict one Cuprite ROI experiment configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = "HSI + DEM" if use_dem else "HSI only"
    print(f"\n[+] Starting Cuprite ROI pipeline: {experiment_name}")

    if use_dem and not DEM_PATH.exists():
        raise FileNotFoundError(
            f"Cuprite DEM VRT not found: {DEM_PATH}\n"
            "Run: /opt/anaconda3/envs/pytorch_py310/bin/python prepare_cuprite_dem.py"
        )

    model = PBEVAE(
        CONFIG_FILE,
        use_dem=use_dem,
        dem_weight=DEM_WEIGHT,
    )
    model.cfg["paths"]["base_dir"] = [CUPRITE_DIR]
    model.cfg["paths"]["out_dir"] = str(output_dir)
    model.cfg["paths"]["dem_path"] = str(DEM_PATH)
    model.cfg["processing"]["target_crs"] = "EPSG:32611"
    model.cfg["rois"] = CUPRITE_ROIS
    model.cfg["processing"].update(
        {
            "probability_method": PROBABILITY_METHOD,
            "single_cluster_activation": SINGLE_CLUSTER_ACTIVATION,
            "single_cluster_temperature": SINGLE_CLUSTER_TEMPERATURE,
            "single_cluster_offset": SINGLE_CLUSTER_OFFSET,
            "use_dem": use_dem,
            "dem_weight": DEM_WEIGHT,
            "dem_integration": "spectral_gate",
            "dem_similarity_width": DEM_SIMILARITY_WIDTH,
        }
    )
    model.probability_method = PROBABILITY_METHOD
    print(
        f"[+] Effective DEM configuration: use_dem={model.use_dem}, "
        f"weight={model.dem_weight:.2f}, integration=spectral_gate"
    )

    harvest_file = output_dir / "harvested_data.npz"
    needs_harvest = not harvest_file.exists()
    if not needs_harvest:
        data = np.load(harvest_file)
        has_dem_column = data["X"].shape[1] == len(data["idxs"]) + 1
        needs_harvest = use_dem and not has_dem_column
        data.close()

    if needs_harvest:
        if harvest_file.exists():
            print("[+] Re-harvesting ROI data to include the DEM feature.")
        harvest_stats = model.harvest(
            output_dir=output_dir,
            roi_radius_pixels=ROI_RADIUS_PIXELS,
            use_dem=use_dem,
            dem_path=DEM_PATH,
        )
        data = np.load(harvest_stats["harvested_path"])
    else:
        data = np.load(harvest_file)

    model.cluster(
        n_clusters=N_CLUSTERS,
        epochs=VAE_EPOCHS,
        prob_method=PROBABILITY_METHOD,
        data_path=harvest_file,
        output_dir=output_dir,
    )
    model.fit_classifier(
        output_dir=output_dir,
        data_path=harvest_file,
        classifier="gradient_boosting",
        calibration=CALIBRATION,
        use_dem=use_dem,
        dem_weight=DEM_WEIGHT,
    )
    pbe_detections = model.predict(
        source=[CUPRITE_DIR],
        output_dir=output_dir,
        conf=PBE_CONFIDENCE,
        save=True,
        use_dem=use_dem,
        dem_weight=DEM_WEIGHT,
    )

    n_bands = len(data["idxs"])
    target_signature = np.mean(data["X"][:, :n_bands], axis=0)
    ace_detections = model.predict_ace(
        target_spectrum=target_signature,
        source=[CUPRITE_DIR],
        out_dir=output_dir,
        conf=ACE_CONFIDENCE,
        save=True,
    )
    sam_detections = model.predict_sam(
        target_spectrum=target_signature,
        source=[CUPRITE_DIR],
        out_dir=output_dir,
        conf=SAM_CONFIDENCE,
        save=True,
    )

    mosaic_info = model.mosaic(
        output_dir=output_dir,
        target_resolution=None,
        target_epsg="EPSG:32611",
        plot_dem=True,
        ace_threshold=ACE_CONFIDENCE,
        sam_threshold=SAM_CONFIDENCE,
    )

    print(f"\n[+] {experiment_name} ROI pipeline complete: PBE-VAE, ACE, and SAM")
    print(f"[+] Outputs: {output_dir}")
    print(f"[+] Mosaic results: {list(mosaic_info)}")
    return {
        "pbe_vae": pbe_detections,
        "ace": ace_detections,
        "sam": sam_detections,
        "mosaic": mosaic_info,
    }


def main():
    results = {}
    if RUN_WITHOUT_DEM:
        results["without_dem"] = run_pipeline(
            use_dem=False,
            output_dir=NO_DEM_OUTPUT_DIR,
        )
    if RUN_WITH_DEM:
        results["with_dem"] = run_pipeline(
            use_dem=True,
            output_dir=DEM_OUTPUT_DIR,
        )
    return results


if __name__ == "__main__":
    main()
