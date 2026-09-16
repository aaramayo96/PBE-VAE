"""
pbe_vae.detector
================
PBE-VAE: Probability Based Expert Variational Autoencoder Package
High-level Object-Oriented Detector API for hyperspectral environmental target detection.
"""

from __future__ import annotations
import os
import sys
import glob
import math
import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple
from xml.sax.saxutils import escape
from osgeo import gdal
from tqdm import tqdm
from scipy.ndimage import binary_dilation, label as scipy_label, binary_fill_holes, binary_opening

try:
    import torch
except ImportError:
    torch = None

try:
    import joblib
except ImportError:
    joblib = None

from pbe_vae.utils.config import load_config, save_config, ensure_dir
from pbe_vae.data.geo import (
    find_aviris_files,
    get_wavelengths,
    pixel_to_latlon,
    roi_centres_in_image,
    read_roi_detections,
    open_georeferenced_dataset,
)
from pbe_vae.data.harvester import harvest_rois
from pbe_vae.data.dataset import harvest_background_data
from pbe_vae.data.mosaic import assemble_probability_mosaic, mask_zeros_to_nodata
from pbe_vae.models.vae import (
    train_and_extract_vae,
    build_vae,
    build_gate,
    LightweightVAE,
    ExpertGate,
)
from pbe_vae.models.classifier import PBEVAEClassifier, make_features
from pbe_vae.utils.metrics import compute_separability_metrics
from pbe_vae.utils.cartography import burn_markers_to_raster
from pbe_vae.utils.qgis_renderer import export_annotated_map_pdf
from pbe_vae.utils.visualization import plot_diagnostic_dashboard, build_pdf_report

_REFL_CLIP = (0.0, 1.5)


def _roi_radius_meters(roi: Any, cfg: Dict[str, Any]) -> float:
    """Return per-ROI radius when present, otherwise the global configured radius."""
    default_radius = float(cfg.get("processing", {}).get("roi_radius_meters", 200.0))
    if isinstance(roi, dict):
        for key in ("roi_radius_meters", "radius_meters", "radius_m"):
            if key in roi and roi[key] is not None:
                return float(roi[key])
    return default_radius


def _configured_roi_diameter_meters(cfg: Dict[str, Any]) -> float:
    """Representative harvested ROI diameter for proposal-window sizing."""
    radii = [_roi_radius_meters(roi, cfg) for roi in cfg.get("rois", [])]
    if not radii:
        radii = [float(cfg.get("processing", {}).get("roi_radius_meters", 200.0))]
    return 2.0 * float(np.median(radii))


def _choose_roi_window_shape(
    component_height: int,
    component_width: int,
    base_height_px: int,
    base_width_px: int,
    max_multiple: int = 1,
) -> Tuple[int, int, int]:
    """Return the smallest base-window multiple that covers a detected blob."""
    base_height_px = max(1, int(base_height_px))
    base_width_px = max(1, int(base_width_px))
    max_multiple = max(1, int(max_multiple))
    multiple = max(
        1,
        int(math.ceil(component_height / base_height_px)),
        int(math.ceil(component_width / base_width_px)),
    )
    multiple = min(multiple, max_multiple)
    return base_height_px * multiple, base_width_px * multiple, multiple


def _centered_bbox(
    row_c: int,
    col_c: int,
    height_px: int,
    width_px: int,
    n_rows: int,
    n_cols: int,
) -> Tuple[int, int, int, int]:
    """Build a centered exclusive ROI bbox, clipped while preserving size when possible."""
    height_px = int(min(max(1, height_px), n_rows))
    width_px = int(min(max(1, width_px), n_cols))
    half_h = height_px // 2
    half_w = width_px // 2

    r0 = int(row_c) - half_h
    c0 = int(col_c) - half_w
    r1 = r0 + height_px
    c1 = c0 + width_px

    if r0 < 0:
        r1 -= r0
        r0 = 0
    if c0 < 0:
        c1 -= c0
        c0 = 0
    if r1 > n_rows:
        r0 -= r1 - n_rows
        r1 = n_rows
    if c1 > n_cols:
        c0 -= c1 - n_cols
        c1 = n_cols

    return max(0, r0), max(0, c0), min(n_rows, r1), min(n_cols, c1)


def _bbox_to_lonlat_polygon(
    bbox: Tuple[int, int, int, int],
    geotransform: Tuple[float, float, float, float, float, float],
    projection_wkt: str,
) -> List[List[float]]:
    """Convert an exclusive pixel bbox to a closed GeoJSON lon/lat polygon."""
    r0, c0, r1, c1 = bbox
    polygon = []
    for row, col in ((r0, c0), (r0, c1), (r1, c1), (r1, c0), (r0, c0)):
        lat, lon = pixel_to_latlon(row, col, geotransform, projection_wkt)
        polygon.append([float(lon), float(lat)])
    return polygon


def _candidate_sort_key(roi: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Rank candidates by peak probability, then region quality."""
    return (
        float(roi.get("max_prob", 0.0)),
        float(roi.get("mean_prob", 0.0)),
        float(roi.get("roi_coverage", 0.0)),
        float(roi.get("component_area_m2", 0.0)),
    )


def _validation_score(roi: Dict[str, Any]) -> float:
    """Score field-validation candidates using confidence plus practical footprint."""
    mean_prob = float(roi.get("mean_prob", 0.0))
    max_prob = float(roi.get("max_prob", mean_prob))
    coverage = float(roi.get("roi_coverage", 0.0))
    area_m2 = max(float(roi.get("component_area_m2", 0.0)), 1.0)
    coverage_score = min(1.0, math.sqrt(max(coverage, 0.0)))
    area_score = min(1.0, math.log10(area_m2 + 1.0) / 6.0)
    return (
        0.45 * mean_prob
        + 0.25 * max_prob
        + 0.20 * coverage_score
        + 0.10 * area_score
    )


def _candidate_window_base_pixels(
    cfg: Dict[str, Any],
    geotransform: Tuple[float, float, float, float, float, float],
) -> Tuple[int, int]:
    """
    Determine the standard detected-ROI proposal size as (height_px, width_px).

    discovery.detected_roi_size_pixels takes precedence. If omitted, a meter
    size can be supplied; otherwise the harvested ROI footprint is used.
    """
    discovery = cfg.get("discovery", {})
    px_size = discovery.get(
        "detected_roi_size_pixels",
        discovery.get("roi_window_base_pixels"),
    )
    if px_size is not None:
        if isinstance(px_size, (list, tuple)) and len(px_size) == 2:
            return max(1, int(px_size[0])), max(1, int(px_size[1]))
        px = max(1, int(px_size))
        return px, px

    px_w_m = abs(geotransform[1]) if geotransform[1] != 0 else 5.0
    px_h_m = abs(geotransform[5]) if geotransform[5] != 0 else px_w_m
    meter_size = discovery.get(
        "detected_roi_size_meters",
        discovery.get("roi_window_base_meters", _configured_roi_diameter_meters(cfg)),
    )
    if isinstance(meter_size, (list, tuple)) and len(meter_size) == 2:
        h_m, w_m = float(meter_size[0]), float(meter_size[1])
    else:
        h_m = w_m = float(meter_size)
    return max(1, int(round(h_m / px_h_m))), max(1, int(round(w_m / px_w_m)))


def _known_roi_records_in_image(cfg: Dict[str, Any], ds_hsi: gdal.Dataset) -> List[Dict[str, Any]]:
    """Find configured ROIs inside a scene with their configured radius."""
    radii_by_name: Dict[str, float] = {}
    for i, roi in enumerate(cfg.get("rois", []), 1):
        if isinstance(roi, dict):
            name = str(roi.get("name", f"ROI_{i:02d}"))
            radii_by_name[name] = _roi_radius_meters(roi, cfg)
        else:
            radii_by_name[f"ROI_{i:02d}"] = _roi_radius_meters(roi, cfg)

    return [
        {
            "name": name,
            "col": col,
            "row": row,
            "radius_m": radii_by_name.get(name, _roi_radius_meters({}, cfg)),
        }
        for name, col, row in roi_centres_in_image(cfg, ds_hsi)
    ]


def _evaluate_known_roi_records(
    prob_map: np.ndarray,
    valid_mask: np.ndarray,
    known_roi_records: List[Dict[str, Any]],
    threshold: float,
    pixel_size_m: float,
) -> List[Dict[str, Any]]:
    """Evaluate whether known ROI windows are recovered by the probability map."""
    n_rows, n_cols = prob_map.shape
    valid_prob = np.where(valid_mask, prob_map, 0.0)
    records = []

    for roi in known_roi_records:
        row = int(roi["row"])
        col = int(roi["col"])
        rad_px = max(1, int(round(float(roi["radius_m"]) / max(pixel_size_m, 1.0))))
        r0, r1 = max(0, row - rad_px), min(n_rows, row + rad_px)
        c0, c1 = max(0, col - rad_px), min(n_cols, col + rad_px)

        patch_prob = valid_prob[r0:r1, c0:c1]
        patch_valid = valid_mask[r0:r1, c0:c1]
        valid_px = int(np.sum(patch_valid))
        if valid_px > 0:
            valid_values = patch_prob[patch_valid]
            max_p = float(np.max(valid_values))
            mean_p = float(np.mean(valid_values))
        else:
            max_p = 0.0
            mean_p = 0.0

        detected_mask = (patch_prob >= threshold) & patch_valid
        detected_px = int(np.sum(detected_mask))
        coverage = detected_px / valid_px if valid_px > 0 else 0.0

        if detected_px > 0:
            y_det, x_det = np.where(detected_mask)
            p_r0 = r0 + int(y_det.min())
            p_r1 = r0 + int(y_det.max())
            p_c0 = c0 + int(x_det.min())
            p_c1 = c0 + int(x_det.max())
            inter_r0, inter_r1 = max(r0, p_r0), min(r1, p_r1)
            inter_c0, inter_c1 = max(c0, p_c0), min(c1, p_c1)
            intersection = max(0, inter_r1 - inter_r0) * max(0, inter_c1 - inter_c0)
            gt_area = max(0, r1 - r0) * max(0, c1 - c0)
            pred_area = max(0, p_r1 - p_r0) * max(0, p_c1 - p_c0)
            union = gt_area + pred_area - intersection
            iou = intersection / union if union > 0 else 0.0
        else:
            iou = 0.0

        records.append(
            {
                "name": roi["name"],
                "row": row,
                "col": col,
                "radius_m": float(roi["radius_m"]),
                "radius_px": rad_px,
                "valid_px": valid_px,
                "detected_px": detected_px,
                "max_prob": max_p,
                "mean_prob": mean_p,
                "coverage": coverage,
                "iou": iou,
                "detected": bool(max_p >= threshold),
            }
        )

    return records


def _write_known_roi_metrics(
    output_dir: Path,
    known_roi_metrics: List[Dict[str, Any]],
    threshold: float,
) -> Optional[Path]:
    """Write per-flightline known-ROI recovery metrics."""
    if not known_roi_metrics:
        return None

    csv_path = output_dir / "known_roi_metrics.csv"
    fields = [
        "flight_line", "name", "detected", "max_prob", "mean_prob",
        "coverage", "iou", "detected_px", "valid_px", "row", "col",
        "radius_m", "radius_px", "eval_threshold",
    ]
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fields)
        writer.writeheader()
        for row in known_roi_metrics:
            record = dict(row)
            record["eval_threshold"] = threshold
            writer.writerow(record)
    return csv_path


def _write_top_candidate_kml(
    output_dir: Path,
    rois: List[Dict[str, Any]],
    top_n: int = 50,
    stem: str = "top_50_new_roi_candidates",
) -> Path:
    """Write top new ROI candidates as Google Earth KML point+area placemarks."""
    kml_path = output_dir / f"{stem}.kml"
    if rois and "validation_score" in rois[0]:
        ranked_rois = sorted(
            rois,
            key=lambda r: (
                float(r.get("validation_score", 0.0)),
                float(r.get("max_prob", 0.0)),
                float(r.get("mean_prob", 0.0)),
            ),
            reverse=True,
        )[: max(0, int(top_n))]
    else:
        ranked_rois = sorted(
            rois,
            key=lambda r: (
                float(r.get("max_prob", 0.0)),
                float(r.get("mean_prob", 0.0)),
                float(r.get("validation_score", 0.0)),
            ),
            reverse=True,
        )[: max(0, int(top_n))]

    def data(name: str, value: Any) -> str:
        return (
            f'        <Data name="{escape(str(name))}">'
            f"<value>{escape(str(value))}</value></Data>\n"
        )

    with open(kml_path, "w") as f_kml:
        f_kml.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f_kml.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
        f_kml.write("  <Document>\n")
        f_kml.write("    <name>Top 50 New ROI Candidates</name>\n")
        f_kml.write("    <Style id=\"candidate_area\">\n")
        f_kml.write("      <IconStyle><scale>0.8</scale></IconStyle>\n")
        f_kml.write("      <LineStyle><color>ff0055ff</color><width>2</width></LineStyle>\n")
        f_kml.write("      <PolyStyle><color>550055ff</color></PolyStyle>\n")
        f_kml.write("    </Style>\n")

        for rank, roi in enumerate(ranked_rois, 1):
            lat = float(roi.get("lat", float("nan")))
            lon = float(roi.get("lon", float("nan")))
            if not np.isfinite(lat) or not np.isfinite(lon):
                continue

            blob_pixels = int(roi.get("n_pixels", roi.get("blob_pixels", 0)))
            roi_detected_px = int(roi.get("roi_detected_px", blob_pixels))
            component_area_m2 = float(roi.get("component_area_m2", 0.0))
            roi_area_m2 = float(roi.get("roi_area_m2", 0.0))
            max_prob = float(roi.get("max_prob", 0.0))
            mean_prob = float(roi.get("mean_prob", 0.0))
            coverage = float(roi.get("roi_coverage", 0.0))

            name = (
                f"{rank:02d} {roi.get('flight_line', '')} "
                f"Pmax={max_prob:.4f} area={component_area_m2:.0f} m2"
            )
            f_kml.write("    <Placemark>\n")
            f_kml.write(f"      <name>{escape(name)}</name>\n")
            f_kml.write("      <styleUrl>#candidate_area</styleUrl>\n")
            f_kml.write("      <ExtendedData>\n")
            f_kml.write(data("rank", rank))
            f_kml.write(data("flight_line", roi.get("flight_line", "")))
            f_kml.write(data("max_prob", f"{max_prob:.6f}"))
            f_kml.write(data("mean_prob", f"{mean_prob:.6f}"))
            f_kml.write(data("blob_pixels", blob_pixels))
            f_kml.write(data("roi_detected_px", roi_detected_px))
            f_kml.write(data("roi_coverage", f"{coverage:.6f}"))
            f_kml.write(data("component_area_m2", f"{component_area_m2:.2f}"))
            f_kml.write(data("roi_area_m2", f"{roi_area_m2:.2f}"))
            f_kml.write(data("lat", f"{lat:.8f}"))
            f_kml.write(data("lon", f"{lon:.8f}"))
            f_kml.write("      </ExtendedData>\n")
            f_kml.write("      <MultiGeometry>\n")
            f_kml.write(f"        <Point><coordinates>{lon:.8f},{lat:.8f},0</coordinates></Point>\n")

            polygon = roi.get("roi_polygon_lonlat", [])
            if polygon:
                coords = " ".join(
                    f"{float(pt[0]):.8f},{float(pt[1]):.8f},0"
                    for pt in polygon
                    if len(pt) >= 2
                )
                if coords:
                    f_kml.write("        <Polygon>\n")
                    f_kml.write("          <outerBoundaryIs><LinearRing>\n")
                    f_kml.write(f"            <coordinates>{coords}</coordinates>\n")
                    f_kml.write("          </LinearRing></outerBoundaryIs>\n")
                    f_kml.write("        </Polygon>\n")

            f_kml.write("      </MultiGeometry>\n")
            f_kml.write("    </Placemark>\n")

        f_kml.write("  </Document>\n")
        f_kml.write("</kml>\n")

    return kml_path


def _write_candidate_exports(
    output_dir: Path,
    rois: List[Dict[str, Any]],
    stem: str = "new_roi_candidates",
) -> Dict[str, str]:
    """Write probability-first ROI candidate CSV and GeoJSON outputs."""
    csv_path = output_dir / f"{stem}.csv"
    geojson_path = output_dir / f"{stem}.geojson"

    csv_fields = [
        "rank", "validation_score", "flight_line", "lat", "lon",
        "row_c", "col_c", "blob_pixels", "component_area_m2",
        "roi_height_px", "roi_width_px", "roi_multiple", "roi_area_m2",
        "mean_prob", "max_prob", "roi_coverage", "roi_detected_px",
        "roi_valid_px", "bbox_r0", "bbox_c0", "bbox_r1", "bbox_c1",
        "component_bbox_r0", "component_bbox_c0", "component_bbox_r1",
        "component_bbox_c1",
    ]

    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
        writer.writeheader()
        for rank, roi in enumerate(rois, 1):
            r0, c0, r1, c1 = roi["bbox"]
            cr0, cc0, cr1, cc1 = roi.get("component_bbox", (r0, c0, r1, c1))
            writer.writerow(
                {
                    "rank": rank,
                    "validation_score": roi.get("validation_score", _validation_score(roi)),
                    "flight_line": roi.get("flight_line", ""),
                    "lat": roi["lat"],
                    "lon": roi["lon"],
                    "row_c": roi["row_c"],
                    "col_c": roi["col_c"],
                    "blob_pixels": roi["n_pixels"],
                    "component_area_m2": roi.get("component_area_m2", 0.0),
                    "roi_height_px": roi.get("roi_height_px", 0),
                    "roi_width_px": roi.get("roi_width_px", 0),
                    "roi_multiple": roi.get("roi_multiple", 0),
                    "roi_area_m2": roi.get("roi_area_m2", 0.0),
                    "mean_prob": roi.get("mean_prob", 0.0),
                    "max_prob": roi.get("max_prob", 0.0),
                    "roi_coverage": roi.get("roi_coverage", 0.0),
                    "roi_detected_px": roi.get("roi_detected_px", 0),
                    "roi_valid_px": roi.get("roi_valid_px", 0),
                    "bbox_r0": r0,
                    "bbox_c0": c0,
                    "bbox_r1": r1,
                    "bbox_c1": c1,
                    "component_bbox_r0": cr0,
                    "component_bbox_c0": cc0,
                    "component_bbox_r1": cr1,
                    "component_bbox_c1": cc1,
                }
            )

    features = []
    for rank, roi in enumerate(rois, 1):
        polygon = roi.get("roi_polygon_lonlat", [])
        if not polygon:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "rank": rank,
                    "validation_score": roi.get("validation_score", _validation_score(roi)),
                    "flight_line": roi.get("flight_line", ""),
                    "lat": roi["lat"],
                    "lon": roi["lon"],
                    "blob_px": roi["n_pixels"],
                    "comp_area_m2": roi.get("component_area_m2", 0.0),
                    "roi_h_px": roi.get("roi_height_px", 0),
                    "roi_w_px": roi.get("roi_width_px", 0),
                    "roi_mult": roi.get("roi_multiple", 0),
                    "roi_area_m2": roi.get("roi_area_m2", 0.0),
                    "mean_prob": roi.get("mean_prob", 0.0),
                    "max_prob": roi.get("max_prob", 0.0),
                    "coverage": roi.get("roi_coverage", 0.0),
                },
                "geometry": {"type": "Polygon", "coordinates": [polygon]},
            }
        )

    with open(geojson_path, "w") as f_geojson:
        json.dump(
            {
                "type": "FeatureCollection",
                "name": stem,
                "features": features,
            },
            f_geojson,
            indent=2,
        )

    return {"candidate_csv": str(csv_path), "candidate_geojson": str(geojson_path)}


def _write_probability_tier_exports(
    output_dir: Path,
    rois: List[Dict[str, Any]],
    probability_tiers: Optional[List[float]],
) -> List[Tuple[float, Dict[str, str]]]:
    """Write candidate exports filtered by max probability thresholds."""
    if not probability_tiers:
        return []

    tier_paths = []
    for tier in sorted({float(t) for t in probability_tiers}):
        label = f"prob_{int(round(tier * 100))}"
        tier_rois = [roi for roi in rois if float(roi.get("max_prob", 0.0)) >= tier]
        paths = _write_candidate_exports(
            output_dir,
            tier_rois,
            stem=f"new_roi_candidates_{label}",
        )
        tier_paths.append((tier, paths))
    return tier_paths


class PBEVAE:
    """
    PBE-VAE (Probability Based Expert Variational Autoencoder) Detector.

    Usage:
        # 1. Build a new model from YAML config
        model = PBEVAE("configs/default.yaml")

        # 2. Select specific architecture versions
        model = PBEVAE("configs/default.yaml", vae_version="v1.0", gate_version="v1.0", latent_dim=5)

        # 3. Load a pretrained model directly
        model = PBEVAE("trained_model.pkl")

        # 4. Train the model (single wrapped function call)
        results = model.train(data="configs/default.yaml", epochs=100, n_clusters=10)

        # 5. Run inference / prediction
        detections = model.predict(source="/path/to/flightlines", conf=0.85)

        # 6. Mosaic & Annotate maps
        mosaic_res = model.mosaic()
        annot_res = model.annotate(top_n=20)
    """

    def __init__(
        self,
        model: Union[str, Path] = "configs/default.yaml",
        weights: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        use_dem: Optional[bool] = None,
        dem_weight: Optional[float] = None,
        vae_version: Optional[str] = None,
        gate_version: Optional[str] = None,
        latent_dim: Optional[int] = None,
        model_cfg: Optional[Union[str, Path, Dict[str, Any]]] = None,
        classifier: Optional[str] = None,
        classifier_cfg: Optional[Dict[str, Any]] = None,
    ):
        model_path = Path(model)
        self.device = device or (
            "cuda"
            if (torch is not None and torch.cuda.is_available())
            else "mps"
            if (torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            else "cpu"
        )

        self.vae_model = None
        self.vae_X_mean = None
        self.vae_X_std = None
        self.cluster_bundle = None
        self.classifier = None
        self.cfg = {}
        pending_weights: Optional[Path] = None

        # Detect if model is a weights file or a config YAML
        if model_path.suffix.lower() in [".pkl", ".pt", ".pth", ".bin"]:
            # Model argument is a weights file
            self.cfg = load_config("configs/default.yaml")
            pending_weights = model_path
        else:
            # Model argument is a configuration YAML
            self.cfg = load_config(model_path)
            if weights and Path(weights).exists():
                pending_weights = Path(weights)

        # Architecture & Model Configuration
        model_setting = model_cfg or self.cfg.get("model", "configs/models/pbe_vae_standard.yaml")
        if isinstance(model_setting, (str, Path)):
            self.model_cfg = load_config(model_setting)
        elif isinstance(model_setting, dict):
            self.model_cfg = model_setting
        else:
            self.model_cfg = {}

        self.vae_version = str(
            vae_version or self.model_cfg.get("version", "v1.0")
        )
        self.gate_version = str(
            gate_version or self.model_cfg.get("gate_version", "v1.0")
        )
        self.latent_dim = int(
            latent_dim
            if latent_dim is not None
            else self.model_cfg.get("latent_dim", self.cfg.get("processing", {}).get("latent_dim", 5))
        )

        # Supervised Detector / Classifier Configuration
        self.classifier_type = str(
            classifier or self.cfg.get("classifier", {}).get("model", "gradient_boosting")
        )
        self.classifier_cfg = (
            classifier_cfg
            if classifier_cfg is not None
            else self.cfg.get("classifier", {})
        )

        # DEM Configuration
        self.use_dem = (
            use_dem
            if use_dem is not None
            else self.cfg.get("processing", {}).get("use_dem", True)
        )
        self.dem_weight = float(
            dem_weight
            if dem_weight is not None
            else self.cfg.get("processing", {}).get(
                "dem_weight", self.cfg.get("processing", {}).get("dem_weight_ratio", 1.0)
            )
        )

        # Probability and Clustering Settings
        self.probability_method = str(
            self.cfg.get("processing", {}).get("probability_method", "gmm")
        )

        # Training and Processing Defaults
        self.epochs = int(self.model_cfg.get("epochs", self.cfg.get("training", {}).get("epochs", 100)))
        self.batch_size = int(self.model_cfg.get("batch_size", self.cfg.get("training", {}).get("batch_size", 4096)))
        self.n_clusters = int(self.cfg.get("processing", {}).get("n_clusters", 10))

        if pending_weights is not None:
            self.load(pending_weights)

    def load(self, weights_path: Union[str, Path]) -> PBEVAE:
        """
        Loads trained model bundle weights. Supports chaining:
            model = PBEVAE("configs/default.yaml").load("trained_model.pkl")
        """
        if joblib is None:
            raise ImportError("joblib is required to load model weights. Please install joblib.")
        if torch is None:
            raise ImportError("torch is required to load PyTorch VAE weights. Please install torch.")

        w_path = Path(weights_path)
        if not w_path.exists():
            raise FileNotFoundError(f"Weights bundle not found at: {w_path}")

        bundle = joblib.load(w_path)
        self.classifier = PBEVAEClassifier(
            model=bundle["model"],
            scaler=bundle["scaler"],
            idxs=bundle["idxs"],
            best_threshold=bundle.get("best_threshold", 0.85),
        )

        vae_path = bundle.get("vae_bundle_path", "")
        if not Path(vae_path).exists():
            vae_path = w_path.parent / "vae_bundle.pt"

        vae_bundle = torch.load(vae_path, map_location=self.device, weights_only=False)
        self.vae_version = vae_bundle.get("vae_version", self.vae_version)
        self.gate_version = vae_bundle.get("gate_version", self.gate_version)
        self.latent_dim = vae_bundle.get("latent_dim", self.latent_dim)

        prior = 1.0 / (1.0 + np.exp(-vae_bundle["expert_prior"]))
        self.vae_model = build_vae(
            version=self.vae_version,
            gate_version=self.gate_version,
            input_dim=vae_bundle["input_dim"],
            expert_prior=prior,
            latent_dim=self.latent_dim,
        ).to(self.device)
        self.vae_model.load_state_dict(vae_bundle["state_dict"])
        self.vae_model.eval()
        self.vae_X_mean = vae_bundle["X_mean"]
        self.vae_X_std = vae_bundle["X_std"]

        cluster_path = bundle.get("cluster_bundle_path", "")
        if not Path(cluster_path).exists():
            cluster_path = w_path.parent / "cluster_bundle.pkl"

        self.cluster_bundle = joblib.load(cluster_path)
        if "use_dem" in bundle:
            self.use_dem = bundle["use_dem"]
        if "dem_weight" in bundle:
            self.dem_weight = bundle["dem_weight"]

        print(
            f"[+] Successfully loaded PBE-VAE weights from {w_path} "
            f"(VAE: {self.vae_version}, Gate: {self.gate_version}, latent_dim: {self.latent_dim}, use_dem: {self.use_dem}, dem_weight: {self.dem_weight})"
        )
        return self

    def harvest(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        roi_radius_pixels: Optional[int] = None,
        roi_radius_meters: Optional[float] = None,
        use_dem: Optional[bool] = None,
        dem_path: Optional[Union[str, Path]] = None,
        vnir_range: Optional[Tuple[float, float]] = None,
        swir_range: Optional[Tuple[float, float]] = None,
        target_crs: Optional[str] = None,
        refl_clip: Optional[Tuple[float, float]] = None,
        save_cubes: bool = True,
    ) -> Dict[str, Any]:
        """Harvests spectral-DEM cubes from flight lines across configured ROIs."""
        out_dir = output_dir or self.cfg.get("paths", {}).get("out_dir", ".")
        use_d = use_dem if use_dem is not None else self.use_dem
        return harvest_rois(
            cfg=self.cfg,
            output_dir=out_dir,
            roi_radius_pixels=roi_radius_pixels,
            roi_radius_meters=roi_radius_meters,
            use_dem=use_d,
            dem_path=dem_path,
            vnir_range=vnir_range,
            swir_range=swir_range,
            target_crs=target_crs,
            refl_clip=refl_clip,
            save_cubes=save_cubes,
        )

    def cluster(
        self,
        n_clusters: Optional[int] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        prob_method: Optional[str] = None,
        data_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Trains the Probability Based Expert VAE and fits latent space clustering."""
        if torch is None or joblib is None:
            raise ImportError("PyTorch and scikit-learn are required for VAE clustering.")

        if prob_method is not None:
            self.probability_method = str(prob_method)

        from sklearn.mixture import GaussianMixture
        from sklearn.cluster import (
            MiniBatchKMeans,
            HDBSCAN,
            Birch,
            BisectingKMeans,
        )
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        out_dir = Path(output_dir or self.cfg.get("paths", {}).get("out_dir", "."))
        ensure_dir(out_dir)

        n_c = int(n_clusters if n_clusters is not None else self.n_clusters)
        eps = int(epochs if epochs is not None else self.epochs)
        bs = int(batch_size if batch_size is not None else self.batch_size)

        d_path = Path(data_path) if data_path else (out_dir / "harvested_data.npz")
        if not d_path.exists():
            d_path = Path("harvested_data.npz")
        data = np.load(d_path)
        X_target = data["X"].astype(np.float32, copy=False)
        fused_idxs = data["idxs"]
        roi_ids = data["ids"]
        data_has_dem = bool(data["has_dem"]) if "has_dem" in data else (X_target.shape[1] > len(fused_idxs))
        has_dem = bool(self.use_dem and data_has_dem)

        files = find_aviris_files(self.cfg["paths"]["base_dir"])
        full_wvl = get_wavelengths(files[0][2])
        active_wvls = (
            np.asarray(data["active_wavelengths"], dtype=np.float32)
            if "active_wavelengths" in data
            else full_wvl[fused_idxs]
        )
        vnir_mask = active_wvls <= 1.2
        swir_mask = active_wvls > 1.2

        # The VAE is spectral-only. DEM remains a separately scaled feature,
        # matching the reference HSC pipeline.
        n_bands = len(fused_idxs)
        # Reflectance is bounded, but elevation is expressed in metres.  Never
        # apply reflectance clipping to the external DEM column.
        X_spectral = np.clip(X_target[:, :n_bands], 0.0, 1.5)
        dem_raw = X_target[:, n_bands : n_bands + 1] if has_dem else None
        X_vnir = X_spectral[:, vnir_mask] if np.any(vnir_mask) else X_spectral
        X_swir = X_spectral[:, swir_mask] if np.any(swir_mask) else X_spectral

        vae_model_cfg = dict(self.model_cfg)
        if "curvature_threshold" in self.cfg.get("processing", {}):
            vae_model_cfg["curvature_threshold"] = float(
                self.cfg["processing"]["curvature_threshold"]
            )

        X_target_vae, vae_model, vae_X_mean, vae_X_std = train_and_extract_vae(
            X_spectral,
            active_wvls=active_wvls,
            latent_dim=self.latent_dim,
            epochs=eps,
            batch_size=bs,
            device=self.device,
            vae_version=self.vae_version,
            gate_version=self.gate_version,
            model_cfg=vae_model_cfg,
        )
        self.vae_model = vae_model
        self.vae_X_mean = vae_X_mean
        self.vae_X_std = vae_X_std

        vae_bundle_path = out_dir / "vae_bundle.pt"
        torch.save(
            {
                "state_dict": vae_model.state_dict(),
                "input_dim": X_spectral.shape[1],
                "latent_dim": self.latent_dim,
                "vae_version": self.vae_version,
                "gate_version": self.gate_version,
                "vae_input": "hsi_only",
                "dem_outside_vae": True,
                "use_dem": has_dem,
                "dem_weight_ratio": self.dem_weight if has_dem else 0.0,
                "expert_prior": vae_model.gate.static_logits.detach().cpu().numpy(),
                "X_mean": vae_X_mean,
                "X_std": vae_X_std,
            },
            vae_bundle_path,
        )

        # Do not put the DEM into the clustering distance.  GMM covariance
        # fitting cancels a simple feature multiplier, making DEM_WEIGHT unable
        # to control the learned DEM influence.
        dem_scaler = StandardScaler().fit(dem_raw) if has_dem else None
        dem_weight = self.dem_weight if has_dem else 0.0
        vae_latent_scaler = StandardScaler()
        X_target_vae = vae_latent_scaler.fit_transform(X_target_vae)

        req_method = str(prob_method).lower().strip() if prob_method is not None else str(self.probability_method).lower().strip()
        if req_method in ["gmm", "hdbscan", "kmeans", "bisecting_kmeans", "birch"]:
            methods_to_run = [req_method]
        else:
            methods_to_run = ["gmm", "kmeans", "birch"]

        metrics_summary = []
        clustering_models = {}

        for method in methods_to_run:
            if method == "kmeans":
                model = MiniBatchKMeans(n_clusters=n_c, random_state=42, batch_size=2048)
                y_target = model.fit_predict(X_target_vae) + 1
                n_clusters_cur = n_c
            elif method == "bisecting_kmeans":
                if n_c <= 1:
                    model = MiniBatchKMeans(n_clusters=1, random_state=42, batch_size=2048)
                else:
                    model = BisectingKMeans(n_clusters=n_c, random_state=42)
                y_target = model.fit_predict(X_target_vae) + 1
                n_clusters_cur = n_c
            elif method == "birch":
                model = Birch(n_clusters=n_c if n_c > 1 else None)
                y_target = model.fit_predict(X_target_vae) + 1
                n_clusters_cur = n_c
            elif method == "hdbscan":
                model = HDBSCAN(min_cluster_size=100, min_samples=15)
                y_target_raw = model.fit_predict(X_target_vae)
                y_target = y_target_raw + 2
                n_clusters_cur = len(np.unique(y_target))
            else:
                model = GaussianMixture(n_components=n_c, random_state=42, reg_covar=1e-2)
                y_target = model.fit_predict(X_target_vae) + 1
                n_clusters_cur = n_c

            sil, db = compute_separability_metrics(X_target_vae, y_target)
            clustering_models[method] = (model, n_clusters_cur)
            metrics_summary.append((method, sil, db))

        best_idx = np.argmax([m[1] for m in metrics_summary])
        best_method = metrics_summary[best_idx][0]
        best_cluster_model, best_n_clusters = clustering_models[best_method]

        density_stats = {}
        if best_n_clusters == 1 and hasattr(best_cluster_model, "score_samples"):
            train_log_dens = best_cluster_model.score_samples(X_target_vae)
            processing_cfg = self.cfg.get("processing", {})
            activation = str(processing_cfg.get("single_cluster_activation", "sigmoid")).lower().strip()
            if activation not in {"sigmoid", "exp_density", "exponential"}:
                raise ValueError(
                    "processing.single_cluster_activation must be 'sigmoid' or 'exp_density'."
                )
            density_stats = {
                "density_mean": float(np.mean(train_log_dens)),
                "density_std": float(np.std(train_log_dens) + 1e-8),
                "density_max": float(np.max(train_log_dens)),
                "single_cluster_activation": activation,
                "single_cluster_temperature": float(
                    processing_cfg.get("single_cluster_temperature", 1.0)
                ),
                "single_cluster_offset": float(processing_cfg.get("single_cluster_offset", 0.0)),
            }

        cluster_bundle_path = out_dir / "cluster_bundle.pkl"
        self.cluster_bundle = {
            "model": best_cluster_model,
            "scaler": vae_latent_scaler,
            "method": best_method,
            "prob_method": self.probability_method,
            "n_clusters": best_n_clusters,
            "has_dem": has_dem,
            "vae_input": "hsi_only",
            "dem_outside_vae": True,
            "dem_scaler": dem_scaler,
            "dem_weight_ratio": dem_weight,
            "dem_integration": "spectral_gate",
            "dem_similarity_width": float(
                self.cfg.get("processing", {}).get("dem_similarity_width", 1.0)
            ),
            **density_stats,
        }
        joblib.dump(self.cluster_bundle, cluster_bundle_path)
        print(
            f"[+] Best clustering: '{best_method.upper()}' (Silhouette: {metrics_summary[best_idx][1]:.3f}, Prob Method: '{self.probability_method}')"
        )

        return {"best_method": best_method, "metrics": metrics_summary, "prob_method": self.probability_method}

    def fit_classifier(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        data_path: Optional[Union[str, Path]] = None,
        classifier: Optional[str] = None,
        classifier_cfg: Optional[Dict[str, Any]] = None,
        calibration: Optional[str] = None,
        use_dem: Optional[bool] = None,
        dem_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Trains the calibrated target classifier on VAE features + background samples."""
        if joblib is None:
            raise ImportError("scikit-learn and joblib are required for classifier training.")

        out_dir = Path(output_dir or self.cfg.get("paths", {}).get("out_dir", "."))
        ensure_dir(out_dir)

        if self.vae_model is None or self.cluster_bundle is None:
            vae_bundle_path = out_dir / "vae_bundle.pt"
            cluster_bundle_path = out_dir / "cluster_bundle.pkl"
            if not (vae_bundle_path.exists() and cluster_bundle_path.exists()):
                raise FileNotFoundError("Missing VAE or cluster bundle. Run .cluster() first.")

            v_bundle = torch.load(vae_bundle_path, map_location=self.device, weights_only=False)
            prior = 1.0 / (1.0 + np.exp(-v_bundle["expert_prior"]))
            self.vae_model = build_vae(
                version=v_bundle.get("vae_version", "v1.0"),
                gate_version=v_bundle.get("gate_version", "v1.0"),
                input_dim=v_bundle["input_dim"],
                expert_prior=prior,
                latent_dim=v_bundle["latent_dim"],
            ).to(self.device)
            self.vae_model.load_state_dict(v_bundle["state_dict"])
            self.vae_model.eval()
            self.vae_X_mean = v_bundle["X_mean"]
            self.vae_X_std = v_bundle["X_std"]
            self.cluster_bundle = joblib.load(cluster_bundle_path)

        has_dem_cb = bool(self.cluster_bundle.get("has_dem", False)) if self.cluster_bundle else False
        use_d = use_dem if use_dem is not None else has_dem_cb
        dem_w = float(dem_weight if dem_weight is not None else self.dem_weight)
        c_type = classifier or self.classifier_type
        c_cfg = dict(classifier_cfg or self.classifier_cfg)
        c_cfg.setdefault(
            "background_negative_weight",
            float(self.cfg.get("processing", {}).get("background_negative_weight", 1.0)),
        )
        if calibration is not None:
            c_cfg["calibration"] = calibration

        d_path = Path(data_path) if data_path else (out_dir / "harvested_data.npz")
        if not d_path.exists():
            d_path = Path("harvested_data.npz")
        data = np.load(d_path)
        fused_idxs = data["idxs"]
        active_wavelengths = (
            np.asarray(data["active_wavelengths"], dtype=np.float32)
            if "active_wavelengths" in data
            else None
        )

        from pbe_vae.data.harvester import normalize_reflectance

        n_bands = len(fused_idxs)
        X_spectral = normalize_reflectance(data["X"][:, :n_bands], _REFL_CLIP)
        if use_d and data["X"].shape[1] > n_bands:
            dem_raw = np.nan_to_num(data["X"][:, n_bands : n_bands + 1]).astype(np.float32)
            X_raw = np.hstack([X_spectral, dem_raw])
        else:
            X_raw = X_spectral
        roi_ids_arr = np.array(data["ids"])
        X_target_arr = make_features(
            X_raw,
            self.vae_model,
            self.vae_X_mean,
            self.vae_X_std,
            self.cluster_bundle,
            device=self.device,
            use_dem=use_d,
            dem_weight=dem_w,
        )

        files = find_aviris_files(self.cfg["paths"]["base_dir"])
        X_background = harvest_background_data(
            files=files,
            cfg=self.cfg,
            fused_idxs=fused_idxs,
            active_wavelengths=active_wavelengths,
            vae_model=self.vae_model,
            vae_X_mean=self.vae_X_mean,
            vae_X_std=self.vae_X_std,
            cluster_bundle=self.cluster_bundle,
            target_size=len(X_target_arr),
            device=self.device,
            use_dem=use_d,
            dem_weight=dem_w,
        )

        X_all = np.vstack((X_target_arr, X_background))
        target_flag = np.concatenate(
            (
                np.ones(len(X_target_arr), dtype=np.int32),
                np.zeros(len(X_background), dtype=np.int32),
            )
        )
        groups = np.concatenate((roi_ids_arr, np.zeros(len(X_background))))

        clf_model = PBEVAEClassifier(
            idxs=fused_idxs,
            classifier_type=c_type,
            classifier_cfg=c_cfg,
        )
        stats = clf_model.fit(
            X_all,
            target_flag,
            groups,
            classifier_type=c_type,
            classifier_cfg=c_cfg,
        )
        self.classifier = clf_model

        model_bundle_path = out_dir / "trained_model.pkl"
        joblib.dump(
            {
                "model": self.classifier.model,
                "scaler": self.classifier.scaler,
                "idxs": fused_idxs,
                "active_wavelengths": active_wavelengths,
                "target_classes": [1],
                "best_threshold": self.classifier.best_threshold,
                "classifier_type": c_type,
                "vae_bundle_path": str(out_dir / "vae_bundle.pt"),
                "cluster_bundle_path": str(out_dir / "cluster_bundle.pkl"),
                "use_dem": use_d,
                "dem_weight": dem_w,
            },
            model_bundle_path,
        )

        print(
            f"[+] Model bundle saved -> {model_bundle_path} "
            f"(Classifier: {c_type}, use_dem={use_d}, dem_weight={dem_w})"
        )
        return stats

    def train(
        self,
        data: Optional[Union[str, Path, Dict[str, Any]]] = None,
        epochs: Optional[int] = None,
        n_clusters: Optional[int] = None,
        batch_size: Optional[int] = None,
        out_dir: Optional[Union[str, Path]] = None,
        harvest: bool = True,
        use_dem: Optional[bool] = None,
        dem_weight: Optional[float] = None,
        vae_version: Optional[str] = None,
        gate_version: Optional[str] = None,
        latent_dim: Optional[int] = None,
        classifier: Optional[str] = None,
        classifier_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Complete end-to-end model training function.
        Wraps data harvesting, PGE-VAE latent space clustering, and calibrated detector training into a single call.

        Zero boilerplate example (uses model config):
            results = model.train()

        Override example:
            results = model.train(epochs=100, classifier="hist_gradient_boosting")
        """
        if data is not None:
            if isinstance(data, (str, Path)):
                self.cfg = load_config(data)
            elif isinstance(data, dict):
                self.cfg = data

        if epochs is not None:
            self.epochs = int(epochs)
        if n_clusters is not None:
            self.n_clusters = int(n_clusters)
        if batch_size is not None:
            self.batch_size = int(batch_size)

        if vae_version is not None:
            self.vae_version = str(vae_version)
        if gate_version is not None:
            self.gate_version = str(gate_version)
        if latent_dim is not None:
            self.latent_dim = int(latent_dim)
        if classifier is not None:
            self.classifier_type = str(classifier)
        if classifier_cfg is not None:
            self.classifier_cfg = classifier_cfg

        if use_dem is not None:
            self.use_dem = use_dem
        if dem_weight is not None:
            self.dem_weight = float(dem_weight)

        output_dir = Path(out_dir or self.cfg.get("paths", {}).get("out_dir", "."))
        ensure_dir(output_dir)

        print("==================================================")
        print(
            f"🚀 PBE-VAE: Training Pipeline (VAE: {self.vae_version}, Gate: {self.gate_version}, "
            f"latent_dim: {self.latent_dim}, Classifier: {self.classifier_type}, use_dem: {self.use_dem}, dem_weight: {self.dem_weight})"
        )
        print("==================================================")

        # Step 1: Data Harvesting
        data_file = output_dir / "harvested_data.npz"
        if harvest or not data_file.exists():
            print("\n[Stage 1/3] Harvesting ROI Cubes & Spectral-DEM Alignments...")
            self.harvest(output_dir=output_dir, use_dem=self.use_dem)
        else:
            print(f"\n[Stage 1/3] Using existing harvested dataset at {data_file}")

        # Step 2: VAE & Latent Space Clustering
        print("\n[Stage 2/3] Training Probability Based Expert VAE & Fitting Clusters...")
        cluster_results = self.cluster(
            n_clusters=self.n_clusters,
            epochs=self.epochs,
            batch_size=self.batch_size,
            output_dir=output_dir,
        )

        # Step 3: Calibrated Target Classifier
        print(f"\n[Stage 3/3] Training Calibrated {self.classifier_type} Target Detection Model...")
        train_results = self.fit_classifier(
            output_dir=output_dir,
            use_dem=self.use_dem,
            dem_weight=self.dem_weight,
            classifier=self.classifier_type,
            classifier_cfg=self.classifier_cfg,
        )

        print("\n==================================================")
        print("✅ PBE-VAE: Training Complete!")
        print(f"   Best Threshold : {self.classifier.best_threshold:.3f}")
        print(f"   Model Weights  : {output_dir / 'trained_model.pkl'}")
        print("==================================================")

        return {
            "cluster_results": cluster_results,
            "train_stats": train_results,
            "weights": str(output_dir / "trained_model.pkl"),
            "best_threshold": self.classifier.best_threshold,
        }

    def run_single_spectrum(
        self,
        target_spectrum: Union[np.ndarray, str, Path],
        source: Optional[Union[str, Path, List[str]]] = None,
        methods: Union[str, List[str], Tuple[str, ...]] = ("ace", "sam"),
        conf: Union[float, Dict[str, float]] = 0.02,
        output_dir: Optional[Union[str, Path]] = None,
        out_dir: Optional[Union[str, Path]] = None,
        save: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run independent spectral detectors for one supplied target spectrum.

        This mode does not harvest ROIs, train the PBE-VAE classifier, or create
        spatial candidate regions. It is intended for one known spectrum and
        writes each detector's output to its own subdirectory.
        """
        if isinstance(methods, str):
            selected = [methods]
        else:
            selected = list(methods)
        selected = [str(method).lower().strip() for method in selected]
        unknown = sorted(set(selected) - {"ace", "sam"})
        if unknown:
            raise ValueError(f"Unsupported single-spectrum detector(s): {unknown}. Choose 'ace' or 'sam'.")

        def detector_conf(name: str) -> float:
            if isinstance(conf, dict):
                return float(conf.get(name, 0.02))
            return float(conf)

        results: Dict[str, List[Dict[str, Any]]] = {}
        common = {
            "target_spectrum": target_spectrum,
            "source": source,
            "output_dir": output_dir,
            "out_dir": out_dir,
            "save": save,
        }
        if "ace" in selected:
            results["ace"] = self.predict_ace(conf=detector_conf("ace"), **common)
        if "sam" in selected:
            results["sam"] = self.predict_sam(conf=detector_conf("sam"), **common)
        return results

    def run_roi_pipeline(
        self,
        train_kwargs: Optional[Dict[str, Any]] = None,
        predict_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the master ROI detection workflow end to end.

        The ROI mode preserves the full supervised workflow: ROI harvesting,
        VAE/clustering, background sampling, calibrated classifier training,
        spatial candidate discovery, and optional raster output.
        """
        training = self.train(**dict(train_kwargs or {}))
        detections = self.predict(**dict(predict_kwargs or {}))
        return {
            "training": training,
            "detections": detections,
        }

    def predict(
        self,
        source: Optional[Union[str, Path, List[str]]] = None,
        conf: Optional[float] = None,
        min_cluster_pixels: Optional[int] = None,
        max_cluster_pixels: Optional[int] = None,
        opening_kernel: Optional[int] = None,
        min_mean_probability: Optional[float] = None,
        min_compactness: Optional[float] = None,
        candidate_sort: Optional[str] = None,
        dedup_radius: Optional[int] = None,
        max_candidates: Optional[int] = None,
        detected_roi_size_pixels: Optional[Union[int, Tuple[int, int], List[int]]] = None,
        roi_window_max_multiple: Optional[int] = None,
        min_roi_coverage: Optional[float] = None,
        output_dir: Optional[Union[str, Path]] = None,
        out_dir: Optional[Union[str, Path]] = None,
        save: bool = True,
        use_dem: Optional[bool] = None,
        dem_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs pixel-level target detection inference on flight lines and applies
        spatial morphological clustering to discover discrete candidate target bodies.
        """
        if self.classifier is None:
            raise RuntimeError("Model is not loaded or trained. Call .train() or .load(weights).")

        use_d = use_dem if use_dem is not None else self.use_dem
        dem_w = float(dem_weight if dem_weight is not None else self.dem_weight)
        dem_path = self.cfg.get("paths", {}).get("dem_path", "")
        has_dem = bool(use_d and dem_path and Path(dem_path).exists())

        threshold = conf if conf is not None else self.classifier.best_threshold
        known_eval_threshold = max(
            float(self.cfg.get("discovery", {}).get("known_eval_threshold", self.classifier.best_threshold)),
            0.40,
        )
        min_px_threshold = int(min_cluster_pixels if min_cluster_pixels is not None else self.cfg.get("discovery", {}).get("min_cluster_pixels", 500))
        max_px_cfg = self.cfg.get("discovery", {}).get("max_cluster_pixels", None)
        max_px_threshold = int(max_cluster_pixels) if max_cluster_pixels is not None else (int(max_px_cfg) if max_px_cfg is not None else None)
        kernel_sz = int(opening_kernel if opening_kernel is not None else self.cfg.get("discovery", {}).get("opening_kernel", 11))
        min_mean = float(
            min_mean_probability
            if min_mean_probability is not None
            else self.cfg.get("discovery", {}).get("min_mean_probability", threshold)
        )
        min_compact = float(
            min_compactness
            if min_compactness is not None
            else self.cfg.get("discovery", {}).get("min_compactness", 0.0)
        )
        sort_key = str(candidate_sort or self.cfg.get("discovery", {}).get("candidate_sort", "mean_prob")).lower()
        if sort_key not in {"mean_prob", "max_prob", "validation_score"}:
            raise ValueError("candidate_sort must be 'mean_prob', 'max_prob', or 'validation_score'.")
        dedup_r = int(dedup_radius if dedup_radius is not None else self.cfg.get("discovery", {}).get("dedup_radius_pixels", 80))
        max_cand = int(max_candidates if max_candidates is not None else self.cfg.get("discovery", {}).get("max_candidates_per_tile", 50))
        roi_max_multiple = int(
            roi_window_max_multiple
            if roi_window_max_multiple is not None
            else self.cfg.get("discovery", {}).get("roi_window_max_multiple", 1)
        )
        min_coverage = float(
            min_roi_coverage
            if min_roi_coverage is not None
            else self.cfg.get("discovery", {}).get("min_roi_coverage", 0.0)
        )

        if detected_roi_size_pixels is not None:
            self.cfg.setdefault("discovery", {})["detected_roi_size_pixels"] = detected_roi_size_pixels

        output_directory = Path(output_dir or out_dir or self.cfg.get("paths", {}).get("out_dir", "."))
        ensure_dir(output_directory)

        base_dirs = source or self.cfg.get("paths", {}).get("base_dir", "")
        files = find_aviris_files(base_dirs)
        if not files:
            raise FileNotFoundError(f"No AVIRIS flightlines found in source: {base_dirs}")

        all_new_rois = []
        all_known_roi_metrics = []

        data_file = Path(self.cfg.get("paths", {}).get("out_dir", ".")) / "harvested_data.npz"
        active_wavelengths = None
        if data_file.exists():
            harvested = np.load(data_file)
            if "active_wavelengths" in harvested:
                active_wavelengths = np.asarray(harvested["active_wavelengths"], dtype=np.float32)

        from pbe_vae.data.geo import open_georeferenced_dataset
        from pbe_vae.data.spectral import align_spectral_cube

        for key, refl_dat, refl_hdr in tqdm(files, desc="Running PBE-VAE inference"):
            pdf_path = output_directory / f"{key}_detection_report.pdf"
            tif_path = output_directory / f"{key}_PROB.tif"

            ds_hsi = open_georeferenced_dataset(refl_dat, refl_hdr)
            if ds_hsi is None:
                continue

            rows_total, cols_total = ds_hsi.RasterYSize, ds_hsi.RasterXSize
            source_wavelengths = get_wavelengths(refl_hdr)
            gt = ds_hsi.GetGeoTransform()
            proj_wkt = ds_hsi.GetProjection()
            roi_base_h_px, roi_base_w_px = _candidate_window_base_pixels(self.cfg, gt)
            pixel_area_m2 = abs(gt[1] * gt[5] - gt[2] * gt[4])

            drv = None
            prob_full = np.full((rows_total, cols_total), -9999.0, dtype=np.float32)

            if has_dem:
                bounds = (
                    gt[0],
                    gt[3] + rows_total * gt[5],
                    gt[0] + cols_total * gt[1],
                    gt[3],
                )
                ds_dem_full = gdal.Warp(
                    f"/vsimem/dem_full_{key}.tif",
                    dem_path,
                    options=gdal.WarpOptions(
                        dstSRS=proj_wkt,
                        width=cols_total,
                        height=rows_total,
                        outputBounds=bounds,
                    ),
                )
                dem_full = (
                    ds_dem_full.ReadAsArray().astype(np.float32)
                    if ds_dem_full
                    else np.zeros((rows_total, cols_total), np.float32)
                )
                gdal.Unlink(f"/vsimem/dem_full_{key}.tif")
            else:
                dem_full = None

            CHUNK_ROWS = 512
            valid_mask_full = np.zeros((rows_total, cols_total), dtype=bool)

            for y_off in range(0, rows_total, CHUNK_ROWS):
                y_sz = min(CHUNK_ROWS, rows_total - y_off)
                h_raw = ds_hsi.ReadAsArray(0, y_off, cols_total, y_sz)
                if h_raw is None:
                    continue
                h_raw = h_raw.astype(np.float32).transpose(1, 2, 0)
                d_chunk = dem_full[y_off : y_off + y_sz, :] if dem_full is not None else None

                mid_band = h_raw.shape[2] // 2
                mask_chunk = (h_raw[..., mid_band] != 0.0) & (h_raw[..., mid_band] != -9999.0)

                H, W, _ = h_raw.shape
                prob_out = np.zeros((H, W), dtype=np.float32)

                if np.any(mask_chunk):
                    from pbe_vae.data.harvester import normalize_reflectance

                    h_sub = normalize_reflectance(
                        align_spectral_cube(h_raw, source_wavelengths, active_wavelengths)
                        if active_wavelengths is not None
                        else h_raw[:, :, self.classifier.idxs],
                        _REFL_CLIP,
                    )
                    if has_dem and d_chunk is not None:
                        d_sub = np.nan_to_num(d_chunk)[..., np.newaxis]
                        raw_flat = np.concatenate(
                            [h_sub.reshape(-1, h_sub.shape[2]), d_sub.reshape(-1, 1)],
                            axis=1,
                        )
                    else:
                        raw_flat = h_sub.reshape(-1, h_sub.shape[2])

                    feats = make_features(
                        raw_pixels=raw_flat,
                        vae_model=self.vae_model,
                        vae_X_mean=self.vae_X_mean,
                        vae_X_std=self.vae_X_std,
                        cluster_bundle=self.cluster_bundle,
                        device=self.device,
                        use_dem=has_dem,
                        dem_weight=dem_w,
                    )
                    feats_2d = feats.reshape(H, W, -1)
                    valid_feats = feats_2d.reshape(-1, feats.shape[-1])[mask_chunk.ravel()]

                    if len(valid_feats) > 0:
                        prob_target = self.classifier.predict_proba(valid_feats)
                        prob_out[mask_chunk] = prob_target.astype(np.float32)

                prob_full[y_off : y_off + y_sz, :] = prob_out
                valid_mask_full[y_off : y_off + y_sz, :] = mask_chunk

            if save:
                drv = gdal.GetDriverByName("GTiff")
                out_ds = drv.Create(
                    str(tif_path),
                    cols_total,
                    rows_total,
                    1,
                    gdal.GDT_Float32,
                    options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"],
                )
                out_ds.SetGeoTransform(gt)
                out_ds.SetProjection(proj_wkt)
                out_band = out_ds.GetRasterBand(1)
                out_band.SetNoDataValue(-9999.0)
                out_band.WriteArray(prob_full)
                out_band.FlushCache()
                out_ds = None

            known_roi_records = _known_roi_records_in_image(self.cfg, ds_hsi)
            known_rois_px = [(r["name"], r["col"], r["row"]) for r in known_roi_records]
            px_w_m = abs(gt[1]) if gt[1] != 0 else 5.0
            px_h_m = abs(gt[5]) if gt[5] != 0 else px_w_m
            px_size_m = max(px_w_m, px_h_m, 1.0)
            known_roi_pixels_set = [
                (
                    int(r["row"]),
                    int(r["col"]),
                    max(1, int(round(float(r["radius_m"]) / px_size_m))),
                )
                for r in known_roi_records
            ]

            valid_prob = np.where(valid_mask_full, prob_full, 0.0)
            for record in _evaluate_known_roi_records(
                prob_full,
                valid_mask_full,
                known_roi_records,
                known_eval_threshold,
                px_size_m,
            ):
                record["flight_line"] = key
                all_known_roi_metrics.append(record)

            # Connected component discovery
            binary = valid_prob >= threshold
            binary = binary_opening(binary, structure=np.ones((kernel_sz, kernel_sz), dtype=bool))
            binary = binary_fill_holes(binary)

            new_rois = []
            if np.any(binary):
                labeled, n_features = scipy_label(binary, structure=np.ones((3, 3), dtype=int))
                for blob_id in range(1, n_features + 1):
                    blob_mask = labeled == blob_id
                    n_px = int(np.sum(blob_mask))
                    if n_px < min_px_threshold or (max_px_threshold is not None and n_px > max_px_threshold):
                        continue
                    rows_b, cols_b = np.where(blob_mask)
                    row_c, col_c = int(np.mean(rows_b)), int(np.mean(cols_b))
                    comp_r0, comp_r1 = int(rows_b.min()), int(rows_b.max()) + 1
                    comp_c0, comp_c1 = int(cols_b.min()), int(cols_b.max()) + 1
                    comp_h = comp_r1 - comp_r0
                    comp_w = comp_c1 - comp_c0

                    roi_h_px, roi_w_px, roi_multiple = _choose_roi_window_shape(
                        comp_h,
                        comp_w,
                        roi_base_h_px,
                        roi_base_w_px,
                        max_multiple=roi_max_multiple,
                    )
                    r0, c0, r1, c1 = _centered_bbox(
                        row_c,
                        col_c,
                        roi_h_px,
                        roi_w_px,
                        rows_total,
                        cols_total,
                    )
                    roi_prob = valid_prob[r0:r1, c0:c1]
                    roi_valid = valid_mask_full[r0:r1, c0:c1]
                    roi_detected = (roi_prob >= threshold) & roi_valid
                    roi_valid_px = int(np.sum(roi_valid))
                    roi_detected_px = int(np.sum(roi_detected))
                    roi_coverage = roi_detected_px / roi_valid_px if roi_valid_px > 0 else 0.0
                    if roi_coverage < min_coverage:
                        continue

                    mean_p = float(np.mean(valid_prob[blob_mask]))
                    if mean_p < min_mean:
                        continue
                    max_p = float(np.max(valid_prob[blob_mask]))
                    dilated = binary_dilation(blob_mask, structure=np.ones((3, 3), dtype=bool))
                    perimeter = int(np.sum(dilated & ~blob_mask))
                    compactness = (4.0 * np.pi * n_px) / (perimeter ** 2) if perimeter else 1.0
                    if compactness < min_compact:
                        continue

                    is_known = any(
                        abs(row_c - known[0]) <= (known[2] if len(known) > 2 else dedup_r)
                        and abs(col_c - known[1]) <= (known[2] if len(known) > 2 else dedup_r)
                        for known in known_roi_pixels_set
                    )
                    if is_known:
                        continue

                    lat, lon = pixel_to_latlon(row_c, col_c, gt, proj_wkt)
                    try:
                        roi_polygon = _bbox_to_lonlat_polygon((r0, c0, r1, c1), gt, proj_wkt)
                    except Exception:
                        roi_polygon = []

                    candidate = {
                        "row_c": row_c,
                        "col_c": col_c,
                        "lat": lat,
                        "lon": lon,
                        "n_pixels": n_px,
                        "mean_prob": mean_p,
                        "max_prob": max_p,
                        "compactness": compactness,
                        "roi_height_px": int(r1 - r0),
                        "roi_width_px": int(c1 - c0),
                        "roi_multiple": roi_multiple,
                        "roi_valid_px": roi_valid_px,
                        "roi_detected_px": roi_detected_px,
                        "roi_coverage": roi_coverage,
                        "component_area_m2": float(n_px * pixel_area_m2),
                        "roi_area_m2": float(roi_valid_px * pixel_area_m2),
                        "roi_polygon_lonlat": roi_polygon,
                        "component_bbox": (comp_r0, comp_c0, comp_r1, comp_c1),
                        "bbox": (r0, c0, r1, c1),
                        "flight_line": key,
                    }
                    candidate["validation_score"] = _validation_score(candidate)
                    new_rois.append(candidate)

            if sort_key == "validation_score":
                new_rois = sorted(new_rois, key=_candidate_sort_key, reverse=True)
                new_rois = sorted(new_rois, key=lambda r: r["validation_score"], reverse=True)
            else:
                new_rois = sorted(new_rois, key=lambda r: r[sort_key], reverse=True)
            new_rois = new_rois[:max_cand]
            all_new_rois.extend(new_rois)
            if save and self.cfg.get("discovery", {}).get("save_individual_reports", False):
                from pbe_vae.utils.visualization import build_pdf_report
                build_pdf_report(
                    pdf_path,
                    key,
                    valid_prob,
                    known_rois_px,
                    new_rois,
                    threshold,
                    valid_mask_full,
                )

        if save:
            if sort_key == "validation_score":
                all_new_rois = sorted(all_new_rois, key=_candidate_sort_key, reverse=True)
                all_new_rois = sorted(all_new_rois, key=lambda r: r["validation_score"], reverse=True)
            else:
                all_new_rois = sorted(all_new_rois, key=lambda r: r[sort_key], reverse=True)
            for rank, roi in enumerate(all_new_rois, 1):
                roi["prob_rank"] = rank
                roi["validation_score"] = _validation_score(roi)

            known_metrics_path = _write_known_roi_metrics(
                output_directory,
                all_known_roi_metrics,
                known_eval_threshold,
            )
            if known_metrics_path is not None:
                best_by_roi: Dict[str, Dict[str, Any]] = {}
                for record in all_known_roi_metrics:
                    name = str(record["name"])
                    if name not in best_by_roi or record["max_prob"] > best_by_roi[name]["max_prob"]:
                        best_by_roi[name] = record
                recovered = sum(1 for record in best_by_roi.values() if record["detected"])
                print(
                    f"[+] Known ROI recovery: {recovered}/{len(best_by_roi)} "
                    f"at P>={known_eval_threshold:.2f}"
                )
                print(f"[+] Known ROI metrics saved to {known_metrics_path}")

            txt_out = output_directory / "new_roi_coordinates.txt"
            with open(txt_out, "w") as f:
                f.write("# Rank  Flight_Line  Lat  Lon  Pixels  Mean_Prob  Max_Prob  ROI_HxW  Coverage\n")
                for i, r in enumerate(all_new_rois, 1):
                    roi_hw = f"{r.get('roi_height_px', 0)}x{r.get('roi_width_px', 0)}"
                    f.write(
                        f"{i:4d}  {r['flight_line']:20s}  {r['lat']:.6f}  {r['lon']:.6f}  "
                        f"{r['n_pixels']:6d}  {r['mean_prob']:.4f}  "
                        f"{r['max_prob']:.4f}  {roi_hw:>9s}  {r.get('roi_coverage', 0.0):.4f}\n"
                    )
            export_paths = _write_candidate_exports(output_directory, all_new_rois)
            print(f"[+] Inference finished. Candidates saved to {txt_out}")
            print(f"[+] Candidate CSV saved to {export_paths['candidate_csv']}")
            print(f"[+] Candidate GeoJSON saved to {export_paths['candidate_geojson']}")
            kml_path = _write_top_candidate_kml(output_directory, all_new_rois, top_n=50)
            print(f"[+] Top 50 candidate KML saved to {kml_path}")
            tier_paths = _write_probability_tier_exports(
                output_directory,
                all_new_rois,
                self.cfg.get("discovery", {}).get("probability_tiers"),
            )
            for tier, paths in tier_paths:
                print(
                    f"[+] Candidate P>={tier:.2f} CSV saved to {paths['candidate_csv']}"
                )
                print(
                    f"[+] Candidate P>={tier:.2f} GeoJSON saved to {paths['candidate_geojson']}"
                )

        return all_new_rois

    def mosaic(
        self,
        source: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        out_dir: Optional[Union[str, Path]] = None,
        target_epsg: str = "EPSG:32611",
        target_resolution: Optional[float] = None,
        shp_path: Optional[Union[str, Path]] = None,
        auto_predict: bool = False,
        plot_dem: bool = False,
        ace_threshold: Optional[float] = None,
        sam_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Reprojects individual flight line GeoTIFFs, builds georeferenced master mosaics (VRT and GeoTIFF),
        and exports high-resolution comparison PDF maps for PBE-VAE, ACE, and SAM detections.
        Set plot_dem=True to create a separate elevation figure for the PBE mosaic footprint.
        ACE/SAM thresholds are reapplied after raster reprojection to prevent interpolation from restoring rejected values.
        """
        in_dir = Path(source or output_dir or out_dir or self.cfg.get("paths", {}).get("out_dir", "."))
        out_d = Path(output_dir or out_dir or in_dir / "prob_mosaic")
        out_d.mkdir(parents=True, exist_ok=True)

        res = target_resolution if target_resolution is not None else self.cfg.get("processing", {}).get("target_resolution", None)

        results: Dict[str, Any] = {}

        # 1. PBE-VAE Probability Mosaic
        prob_tifs = sorted(p for p in in_dir.glob("*_PROB.tif") if not p.name.startswith("._"))
        if not prob_tifs:
            prob_tifs = sorted(p for p in in_dir.rglob("*_PROB.tif") if not p.name.startswith("._"))

        if not prob_tifs and auto_predict:
            print("[i] No probability tiles found. Running flightline inference...")
            self.predict(output_dir=in_dir, save=True)
            prob_tifs = sorted(p for p in in_dir.glob("*_PROB.tif") if not p.name.startswith("._"))

        if prob_tifs:
            pbe_out_d = out_d / "pbe_mosaic" if (out_d / "ace_mosaic").exists() else out_d
            pbe_res = assemble_probability_mosaic(
                prob_dir=in_dir,
                out_dir=pbe_out_d,
                pattern="*_PROB.tif",
                prefix="prob_mosaic",
                target_epsg=target_epsg,
                target_resolution=res,
            )
            pdf_pbe = pbe_out_d / "prob_mosaic_map.pdf"
            export_annotated_map_pdf(
                mosaic_tif=pbe_res["masked_tif"],
                out_pdf=pdf_pbe,
            )
            pbe_res["pdf_map"] = pdf_pbe
            results["pbe_vae"] = pbe_res
            results["masked_tif"] = pbe_res["masked_tif"]
            results["pdf_map"] = pdf_pbe
            print(f"[+] PBE-VAE Mosaic assembled: {pbe_res['masked_tif']}")
            print(f"[+] PBE-VAE Mosaic PDF map: {pdf_pbe}")

        # 2. ACE Detector Mosaic
        ace_dir = in_dir / "ace_detections" if (in_dir / "ace_detections").exists() else in_dir
        ace_tifs = sorted(p for p in ace_dir.glob("*_ACE.tif") if not p.name.startswith("._"))
        if not ace_tifs:
            ace_tifs = sorted(p for p in ace_dir.rglob("*_ACE.tif") if not p.name.startswith("._"))

        if ace_tifs:
            ace_out_d = out_d / "ace_mosaic"
            ace_out_d.mkdir(parents=True, exist_ok=True)
            ace_res = assemble_probability_mosaic(
                prob_dir=ace_dir,
                out_dir=ace_out_d,
                pattern="*_ACE.tif",
                prefix="ace_mosaic",
                target_epsg=target_epsg,
                target_resolution=res,
            )
            if ace_threshold is not None:
                mask_zeros_to_nodata(
                    ace_res["mosaic_tif"],
                    ace_res["masked_tif"],
                    minimum_value=ace_threshold,
                )
            pdf_ace = ace_out_d / "ace_mosaic_map.pdf"
            from pbe_vae.utils.qgis_renderer import render_matplotlib_map_pdf
            render_matplotlib_map_pdf(
                mosaic_tif=ace_res["masked_tif"],
                out_pdf=pdf_ace,
                title="ACE (Adaptive Coherence Estimator) Detection Mosaic",
                cmap="viridis",
                cbar_label="ACE Detector Score",
            )
            ace_res["pdf_map"] = pdf_ace
            results["ace"] = ace_res
            print(f"[+] ACE Mosaic assembled: {ace_res['masked_tif']}")
            print(f"[+] ACE Mosaic PDF map: {pdf_ace}")

        # 3. SAM Detector Mosaic
        sam_dir = in_dir / "sam_detections" if (in_dir / "sam_detections").exists() else in_dir
        sam_tifs = sorted(p for p in sam_dir.glob("*_SAM.tif") if not p.name.startswith("._"))
        if not sam_tifs:
            sam_tifs = sorted(p for p in sam_dir.rglob("*_SAM.tif") if not p.name.startswith("._"))

        if sam_tifs:
            sam_out_d = out_d / "sam_mosaic"
            sam_out_d.mkdir(parents=True, exist_ok=True)
            sam_res = assemble_probability_mosaic(
                prob_dir=sam_dir,
                out_dir=sam_out_d,
                pattern="*_SAM.tif",
                prefix="sam_mosaic",
                target_epsg=target_epsg,
                target_resolution=res,
                zero_is_nodata=False,
            )
            if sam_threshold is not None:
                mask_zeros_to_nodata(
                    sam_res["mosaic_tif"],
                    sam_res["masked_tif"],
                    zero_is_nodata=False,
                    maximum_value=sam_threshold,
                )
            pdf_sam = sam_out_d / "sam_mosaic_map.pdf"
            from pbe_vae.utils.qgis_renderer import render_matplotlib_map_pdf
            render_matplotlib_map_pdf(
                mosaic_tif=sam_res["masked_tif"],
                out_pdf=pdf_sam,
                title="SAM (Spectral Angle Mapper) Detection Mosaic - lower is better",
                cmap="cividis",
                cbar_label="Spectral Angle (radians)",
                zero_is_nodata=False,
                invert_colorbar=True,
            )
            sam_res["pdf_map"] = pdf_sam
            results["sam"] = sam_res
            print(f"[+] SAM Mosaic assembled: {sam_res['masked_tif']}")
            print(f"[+] SAM Mosaic PDF map: {pdf_sam}")

        # 4. Three-way Comparative PDF Map
        if all(name in results for name in ("pbe_vae", "ace", "sam")):
            comp_pdf = out_d / "detection_comparison_pbe_vs_ace_vs_sam.pdf"
            from pbe_vae.utils.qgis_renderer import export_three_way_comparison_map_pdf
            export_three_way_comparison_map_pdf(
                pbe_mosaic_tif=results["pbe_vae"]["masked_tif"],
                ace_mosaic_tif=results["ace"]["masked_tif"],
                sam_mosaic_tif=results["sam"]["masked_tif"],
                out_pdf=comp_pdf,
            )
            results["comparison_pdf"] = comp_pdf
            print(f"[+] Three-way comparative PDF map exported: {comp_pdf}")
        elif all(name in results for name in ("ace", "sam")):
            comp_pdf = out_d / "detection_comparison_ace_vs_sam.pdf"
            from pbe_vae.utils.qgis_renderer import export_ace_sam_comparison_map_pdf
            export_ace_sam_comparison_map_pdf(
                ace_mosaic_tif=results["ace"]["masked_tif"],
                sam_mosaic_tif=results["sam"]["masked_tif"],
                out_pdf=comp_pdf,
            )
            results["comparison_pdf"] = comp_pdf
            print(f"[+] ACE/SAM comparative PDF map exported: {comp_pdf}")

        if plot_dem and "pbe_vae" in results:
            dem_value = self.cfg.get("paths", {}).get("dem_path", "")
            dem_path = Path(dem_value) if dem_value else None
            pbe_mosaic = results["pbe_vae"]["masked_tif"]
            if dem_path is not None and dem_path.is_file():
                from pbe_vae.utils.qgis_renderer import render_dem_elevation_map

                dem_figure = out_d / "dem_elevation_map.pdf"
                if render_dem_elevation_map(dem_path, pbe_mosaic, dem_figure):
                    results["dem_figure"] = dem_figure
                    print(f"[+] DEM elevation figure: {dem_figure}")
            else:
                print(f"[WARN] DEM figure skipped; DEM not found: {dem_path}")

        return results

    def annotate(
        self,
        mosaic: Optional[Union[str, Path]] = None,
        out_dir: Optional[Union[str, Path]] = None,
        top_n: int = 20,
        shp_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """
        Burns cartographic markers (Known ROIs: stars, New ROIs: bullseyes)
        into a transparent GeoTIFF overlay and exports high-DPI PDF map.
        """
        out_d = Path(out_dir or self.cfg.get("paths", {}).get("out_dir", "."))
        mos_tif = Path(mosaic or out_d / "prob_mosaic" / "prob_mosaic_masked.tif")
        if not mos_tif.exists():
            mos_tif = out_d / "prob_mosaic_masked.tif"

        if not mos_tif.exists():
            raise FileNotFoundError(f"Mosaic reference raster not found: {mos_tif}")

        annot_tif = mos_tif.parent / "roi_annotations.tif"
        pdf_out = mos_tif.parent / "roi_annotated_map.pdf"

        known_rois = self.cfg.get("rois", [])
        new_rois_txt = out_d / "new_roi_coordinates.txt"
        new_rois_df = read_roi_detections(new_rois_txt, top_n=top_n) if new_rois_txt.exists() else None
        new_rois_list = new_rois_df.to_dict(orient="records") if new_rois_df is not None else []

        burn_markers_to_raster(
            reference_mosaic_path=mos_tif,
            output_annot_path=annot_tif,
            known_rois=known_rois,
            new_rois=new_rois_list,
        )

        shapefile = Path(shp_path) if shp_path else Path(self.cfg.get("paths", {}).get("shp_path", ""))
        export_annotated_map_pdf(
            mosaic_tif=mos_tif,
            annot_tif=annot_tif,
            out_pdf=pdf_out,
            shp_path=shapefile if shapefile.exists() else None,
        )

        print(f"[+] Cartographic annotation overlay generated: {annot_tif}")
        print(f"[+] Annotated map PDF exported: {pdf_out}")
        return {"annot_tif": annot_tif, "pdf_map": pdf_out}

    def get_target_spectrum(self, method: str = "mean") -> np.ndarray:
        """
        Extracts the reference target spectral signature from harvested ROI pixels.
        Parameters:
            method: Aggregation metric ('mean' or 'median').
        Returns:
            1D numpy array of target reflectance across active spectral bands.
        """
        data_file = Path(self.cfg.get("paths", {}).get("out_dir", ".")) / "harvested_data.npz"
        if not data_file.exists():
            self.harvest()
        dat = np.load(data_file)
        fused_idxs = dat["idxs"]
        n_bands = len(fused_idxs)
        spectral_pixels = dat["X"][:, :n_bands]
        if method == "median":
            return np.median(spectral_pixels, axis=0).astype(np.float32)
        return np.mean(spectral_pixels, axis=0).astype(np.float32)

    def predict_ace(
        self,
        target_spectrum: Optional[Union[np.ndarray, str, Path]] = None,
        source: Optional[Union[str, Path, List[str]]] = None,
        conf: float = 0.5,
        output_dir: Optional[Union[str, Path]] = None,
        out_dir: Optional[Union[str, Path]] = None,
        save: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Runs ACE across flightlines, retaining scores at or above ``conf``.

        Parameters:
            target_spectrum: Target spectral signature array (1D), or path to spectrum file (.npy/.txt/.csv).
                             If None, extracts mean target spectrum from harvested ROIs via get_target_spectrum().
            source: Directory containing flightlines.
            conf: ACE detection threshold.
            output_dir: Destination folder.
            out_dir: Alias for output_dir.
            save: Whether to save output GeoTIFFs.
        """
        try:
            from spectral.algorithms.detectors import ACE, ace
            import spectral.algorithms as algo
        except ImportError as e:
            raise ImportError(
                "The official Spectral Python library (`spectral`) is required for ACE detection. "
                "Please install it using: pip install spectral"
            ) from e

        out_directory = Path(output_dir or out_dir or self.cfg.get("paths", {}).get("out_dir", ".")) / "ace_detections"
        ensure_dir(out_directory)

        # Retrieve target signature
        if target_spectrum is None:
            target_sig = self.get_target_spectrum(method="mean")
        elif isinstance(target_spectrum, (str, Path)):
            spec_path = Path(target_spectrum)
            if spec_path.suffix == ".npy":
                target_sig = np.load(spec_path)
            else:
                target_sig = np.loadtxt(spec_path)
        else:
            target_sig = np.asarray(target_spectrum, dtype=np.float32)

        base_dirs = source or self.cfg.get("paths", {}).get("base_dir", "")
        files = find_aviris_files(base_dirs)

        results = []
        for key, refl_dat, refl_hdr in tqdm(files, desc="Running ACE detector"):
            ds_hsi = open_georeferenced_dataset(refl_dat, refl_hdr)
            if ds_hsi is None:
                continue

            from pbe_vae.models.baselines import format_hsi_cube, format_target_spectrum

            h_raw = ds_hsi.ReadAsArray()
            data_file = Path(self.cfg.get("paths", {}).get("out_dir", ".")) / "harvested_data.npz"
            harvested = np.load(data_file) if data_file.exists() else None
            fused_idxs = harvested["idxs"] if harvested is not None else None
            active_wavelengths = harvested["active_wavelengths"] if harvested is not None and "active_wavelengths" in harvested else None

            # Adapt inputs to Spectral Python format: (rows, cols, bands)
            cube, valid_mask = format_hsi_cube(
                h_raw,
                active_indices=fused_idxs,
                source_wavelengths=get_wavelengths(refl_hdr) if active_wavelengths is not None else None,
                target_wavelengths=active_wavelengths,
            )
            target_sig = format_target_spectrum(target_sig, expected_bands=cube.shape[2])

            # Run official Spectral Python ACE detector
            ace_map = np.asarray(ace(cube, target_sig), dtype=np.float32)
            if ace_map.shape != (ds_hsi.RasterYSize, ds_hsi.RasterXSize) and ace_map.shape == (ds_hsi.RasterXSize, ds_hsi.RasterYSize):
                ace_map = ace_map.T
            detection_mask = valid_mask & np.isfinite(ace_map) & (ace_map >= float(conf))
            ace_map[~detection_mask] = -9999.0

            if save:
                gt = ds_hsi.GetGeoTransform()
                proj = ds_hsi.GetProjection()
                tif_path = out_directory / f"{key}_ACE.tif"
                drv = gdal.GetDriverByName("GTiff")
                out_ds = drv.Create(
                    str(tif_path),
                    ds_hsi.RasterXSize,
                    ds_hsi.RasterYSize,
                    1,
                    gdal.GDT_Float32,
                    options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"],
                )
                out_ds.SetGeoTransform(gt)
                out_ds.SetProjection(proj)
                b = out_ds.GetRasterBand(1)
                b.SetNoDataValue(-9999.0)
                b.WriteArray(ace_map)
                b.FlushCache()
                out_ds = None

            detected_scores = ace_map[detection_mask]
            results.append(
                {
                    "flight_line": key,
                    "mean_score": float(np.mean(detected_scores)) if detected_scores.size else 0.0,
                    "n_detected_pixels": int(np.sum(detection_mask)),
                    "threshold": float(conf),
                }
            )

        print(f"[+] ACE Target Detection finished -> {out_directory}")
        return results

    def predict_sam(
        self,
        target_spectrum: Optional[Union[np.ndarray, str, Path]] = None,
        source: Optional[Union[str, Path, List[str]]] = None,
        conf: float = 0.02,
        output_dir: Optional[Union[str, Path]] = None,
        out_dir: Optional[Union[str, Path]] = None,
        save: bool = True,
    ) -> List[Dict[str, Any]]:
        """Runs SAM across flightlines, retaining angles at or below ``conf`` radians."""
        try:
            from spectral.algorithms import spectral_angles
        except ImportError as e:
            raise ImportError(
                "The official Spectral Python library (`spectral`) with spectral_angles support "
                "is required for SAM detection. "
                "Please install it using: pip install spectral"
            ) from e

        out_directory = Path(output_dir or out_dir or self.cfg.get("paths", {}).get("out_dir", ".")) / "sam_detections"
        ensure_dir(out_directory)

        if target_spectrum is None:
            target_sig = self.get_target_spectrum(method="mean")
        elif isinstance(target_spectrum, (str, Path)):
            spec_path = Path(target_spectrum)
            target_sig = np.load(spec_path) if spec_path.suffix == ".npy" else np.loadtxt(spec_path)
        else:
            target_sig = np.asarray(target_spectrum, dtype=np.float32)

        base_dirs = source or self.cfg.get("paths", {}).get("base_dir", "")
        files = find_aviris_files(base_dirs)
        results = []

        for key, refl_dat, refl_hdr in tqdm(files, desc="Running SAM detector"):
            ds_hsi = open_georeferenced_dataset(refl_dat, refl_hdr)
            if ds_hsi is None:
                continue

            from pbe_vae.models.baselines import format_hsi_cube, format_target_spectrum

            h_raw = ds_hsi.ReadAsArray()
            data_file = Path(self.cfg.get("paths", {}).get("out_dir", ".")) / "harvested_data.npz"
            harvested = np.load(data_file) if data_file.exists() else None
            fused_idxs = harvested["idxs"] if harvested is not None else None
            active_wavelengths = harvested["active_wavelengths"] if harvested is not None and "active_wavelengths" in harvested else None
            cube, valid_mask = format_hsi_cube(
                h_raw,
                active_indices=fused_idxs,
                source_wavelengths=get_wavelengths(refl_hdr) if active_wavelengths is not None else None,
                target_wavelengths=active_wavelengths,
            )
            target_sig = format_target_spectrum(target_sig, expected_bands=cube.shape[2])

            # Spectral Python exposes SAM as spectral_angles(), which returns
            # one angle band per target member. Lower angles are closer matches.
            sam_map = np.asarray(
                spectral_angles(cube, target_sig[np.newaxis, :]),
                dtype=np.float32,
            )[..., 0]
            if sam_map.shape != (ds_hsi.RasterYSize, ds_hsi.RasterXSize) and sam_map.shape == (ds_hsi.RasterXSize, ds_hsi.RasterYSize):
                sam_map = sam_map.T
            # Unlike ACE, lower SAM angles are better.  Persist only accepted
            # detections so the mosaic and color bar cannot display rejected,
            # high-angle background pixels.
            detection_mask = valid_mask & (sam_map <= float(conf))
            sam_map[~detection_mask] = -9999.0

            if save:
                gt = ds_hsi.GetGeoTransform()
                proj = ds_hsi.GetProjection()
                tif_path = out_directory / f"{key}_SAM.tif"
                drv = gdal.GetDriverByName("GTiff")
                out_ds = drv.Create(
                    str(tif_path),
                    ds_hsi.RasterXSize,
                    ds_hsi.RasterYSize,
                    1,
                    gdal.GDT_Float32,
                    options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"],
                )
                out_ds.SetGeoTransform(gt)
                out_ds.SetProjection(proj)
                band = out_ds.GetRasterBand(1)
                band.SetNoDataValue(-9999.0)
                band.WriteArray(sam_map)
                band.FlushCache()
                out_ds = None

            detected_angles = sam_map[detection_mask]
            results.append(
                {
                    "flight_line": key,
                    "mean_score": float(np.mean(detected_angles)) if len(detected_angles) else float("nan"),
                    "n_detected_pixels": int(np.sum(detection_mask)),
                    "threshold_radians": float(conf),
                }
            )

        print(f"[+] SAM Target Detection finished -> {out_directory}")
        return results
