"""
pbe_vae.data.geo
================
Geospatial transformation, projection, and ENVI header parsing functions.
"""

import os
import re
import numpy as np
from pathlib import Path
from pyproj import Transformer
from osgeo import gdal, osr
from typing import List, Tuple, Dict, Any, Union, Optional


def get_wavelengths(hdr_path: Union[str, Path]) -> np.ndarray:
    """
    Extract physical wavelength array from ENVI header file.
    Automatically converts nm to micrometers if values > 10.
    """
    wavelengths = []
    in_wave = False
    with open(hdr_path, "r") as f:
        for line in f:
            if "wavelength =" in line or "wavelength=" in line:
                in_wave = True
                line = line.split("{")[-1]
            if in_wave:
                clean_line = line.replace("}", "").replace(",", " ").strip()
                if clean_line:
                    wavelengths.extend([float(v) for v in clean_line.split()])
                if "}" in line:
                    break
    if wavelengths and wavelengths[0] > 10:
        wavelengths = [w / 1000.0 for w in wavelengths]
    return np.array(wavelengths, dtype=np.float32)


def find_aviris_files(base_dirs: Union[str, List[str], Path]) -> List[Tuple[str, Path, Path]]:
    """
    Scans specified directories for AVIRIS reflectance headers (*_rfl.hdr).
    Returns list of (flightline_key, data_path, header_path).
    """
    if isinstance(base_dirs, (str, Path)):
        base_dirs = [base_dirs]

    files = []
    seen_keys = set()
    for base_dir in base_dirs:
        b_path = Path(base_dir)
        if not b_path.exists():
            continue
        # Search for AVIRIS (_rfl.hdr), SpecTIR/PROSPECTIR (_ref.hdr), and general ENVI headers
        candidate_hdrs = list(b_path.rglob("*_rfl.hdr")) + list(b_path.rglob("*_ref.hdr")) + list(b_path.rglob("*.hdr"))
        for refl_hdr in candidate_hdrs:
            if refl_hdr.name.startswith("._") or "_hytools" in refl_hdr.name:
                continue
            refl_dat = refl_hdr.with_suffix("")
            if not refl_dat.exists():
                refl_dat = refl_hdr.with_suffix(".dat")
                if not refl_dat.exists():
                    refl_dat = refl_hdr.with_suffix(".bin")
                    if not refl_dat.exists():
                        refl_dat = refl_hdr.with_suffix(".img")
                        if not refl_dat.exists():
                            refl_dat = refl_hdr.with_suffix(".raw")
                            if not refl_dat.exists():
                                continue
            key = refl_hdr.stem.replace("_rfl", "").replace("_pol_ref", "").replace("_ref", "")
            if key not in seen_keys:
                seen_keys.add(key)
                files.append((key, refl_dat, refl_hdr))
    return sorted(files, key=lambda x: x[0])


def parse_map_info(hdr_path: Union[str, Path]) -> Optional[Tuple[Tuple[float, float, float, float, float, float], str]]:
    """
    Parses ENVI 'map info' line to extract affine GeoTransform and WKT projection.
    Searches companion GLT/IGM headers if the raw header lacks georeferencing tags.
    """
    hdr_path = Path(hdr_path)
    map_info_str = None

    # 1. Check current header
    if hdr_path.exists():
        with open(hdr_path, "r", errors="ignore") as f:
            for line in f:
                if "map info" in line:
                    map_info_str = line
                    break

    # 2. Check companion GLT/geoloc headers
    if not map_info_str:
        key = hdr_path.stem.replace("_pol_ref", "").replace("_ref", "").replace("_rfl", "")
        candidates = [
            hdr_path.parent.parent / "IGM_GLT" / f"{key}_GLT.hdr",
            hdr_path.parent / f"{key}_GLT.hdr",
            hdr_path.parent.parent / "qgis_mosaic" / "georef_lines" / f"{key}_georef.tif",
        ]
        for cand in candidates:
            if cand.exists():
                if cand.suffix == ".tif":
                    ds_temp = gdal.Open(str(cand))
                    if ds_temp and ds_temp.GetProjection():
                        return ds_temp.GetGeoTransform(), ds_temp.GetProjection()
                elif cand.suffix == ".hdr":
                    with open(cand, "r", errors="ignore") as f:
                        for line in f:
                            if "map info" in line:
                                map_info_str = line
                                break
            if map_info_str:
                break

    if not map_info_str:
        return None

    m = re.search(r"\{([^}]+)\}", map_info_str)
    if not m:
        return None
    tokens = [t.strip() for t in m.group(1).split(",")]
    if len(tokens) < 7:
        return None

    proj_type = tokens[0].upper()
    ref_x = float(tokens[1])
    ref_y = float(tokens[2])
    tie_x = float(tokens[3])
    tie_y = float(tokens[4])
    psize_x = float(tokens[5])
    psize_y = float(tokens[6])

    ul_x = tie_x - (ref_x - 1.0) * psize_x
    ul_y = tie_y + (ref_y - 1.0) * psize_y
    gt = (ul_x, psize_x, 0.0, ul_y, 0.0, -psize_y)

    srs = osr.SpatialReference()
    if "UTM" in proj_type and len(tokens) >= 9:
        zone = int(tokens[7])
        is_north = "NORTH" in tokens[8].upper()
        srs.SetUTM(zone, is_north)
        srs.SetWellKnownGeogCS("WGS84")
    else:
        srs.SetWellKnownGeogCS("WGS84")

    return gt, srs.ExportToWkt()


def open_georeferenced_dataset(refl_dat: Union[str, Path], refl_hdr: Optional[Union[str, Path]] = None) -> Optional[gdal.Dataset]:
    """
    Opens an HSI dataset and ensures it possesses valid georeferencing (GeoTransform and Projection).
    If missing from the raw header, attaches georeferencing from companion GLT/IGM files.
    """
    dat_path = Path(refl_dat)
    hdr_path = Path(refl_hdr) if refl_hdr else dat_path.with_suffix(".hdr")

    key = dat_path.stem.replace("_pol_ref", "").replace("_ref", "").replace("_rfl", "")
    georef_tif = dat_path.parent.parent / "qgis_mosaic" / "georef_lines" / f"{key}_georef.tif"
    if georef_tif.exists():
        ds = gdal.Open(str(georef_tif))
        if ds is not None and ds.GetProjection():
            return ds

    ds = gdal.Open(str(dat_path))
    if ds is None:
        return None

    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    # If georeferencing is missing, attach from companion GLT
    if not proj or gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        info = parse_map_info(hdr_path)
        if info is not None:
            c_gt, c_wkt = info
            vrt_path = f"/vsimem/{key}_geo.vrt"
            vrt_ds = gdal.GetDriverByName("VRT").CreateCopy(vrt_path, ds)
            vrt_ds.SetGeoTransform(c_gt)
            vrt_ds.SetProjection(c_wkt)
            return vrt_ds

    return ds


def find_prob_tifs(directory: Union[str, Path]) -> List[Path]:
    """
    Finds all probability GeoTIFFs (*_PROB.tif) in directory,
    ignoring macOS resource fork files (._*).
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    return sorted(
        p for p in dir_path.glob("*_PROB.tif")
        if not p.name.startswith("._") and p.is_file()
    )


def pixel_to_latlon(
    row: int,
    col: int,
    geotransform: Tuple[float, float, float, float, float, float],
    projection_wkt: str
) -> Tuple[float, float]:
    """
    Convert (row, col) pixel coordinate to (lat, lon) using GDAL geotransform & SRS.
    """
    gt = geotransform
    x_proj = gt[0] + col * gt[1] + row * gt[2]
    y_proj = gt[3] + col * gt[4] + row * gt[5]

    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(projection_wkt)

    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    lat, lon, _ = transform.TransformPoint(x_proj, y_proj)
    return float(lat), float(lon)


def latlon_to_pixel(
    lat: float,
    lon: float,
    geotransform: Tuple[float, float, float, float, float, float],
    projection_wkt: str
) -> Tuple[int, int]:
    """
    Convert (lat, lon) in WGS84 to (col, row) pixel indices in a reference raster.
    """
    t = Transformer.from_crs("EPSG:4326", projection_wkt, always_xy=True)
    x, y = t.transform(lon, lat)
    gt = geotransform
    det = gt[1] * gt[5] - gt[2] * gt[4]
    col = int((gt[5] * (x - gt[0]) - gt[2] * (y - gt[3])) / det)
    row = int((gt[1] * (y - gt[3]) - gt[4] * (x - gt[0])) / det)
    return col, row


def roi_centres_in_image(
    cfg: Dict[str, Any],
    ds_hsi: gdal.Dataset
) -> List[Tuple[str, int, int]]:
    """
    Returns a list of (roi_name, col, row) for each configured ROI that falls
    inside the bounding box of the given GDAL dataset.
    Supports Lat/Lon, UTM (x, y), and (easting, northing).
    """
    gt = ds_hsi.GetGeoTransform()
    proj = ds_hsi.GetProjection()

    target_crs = proj if proj else cfg.get("processing", {}).get("target_crs", "EPSG:32611")
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    rows, cols = ds_hsi.RasterYSize, ds_hsi.RasterXSize
    results = []

    for i, roi in enumerate(cfg.get("rois", []), 1):
        try:
            if isinstance(roi, (list, tuple)):
                v1, v2 = float(roi[0]), float(roi[1])
                name = f"ROI_{i:02d}"
                if v1 > 1000 or v2 > 1000:
                    x, y = v1, v2
                else:
                    lon_val, lat_val = (v1, v2) if abs(v1) > abs(v2) else (v2, v1)
                    x, y = transformer.transform(lon_val, lat_val)
            elif isinstance(roi, dict):
                name = roi.get("name", f"ROI_{i:02d}")
                if "x" in roi and "y" in roi:
                    x, y = float(roi["x"]), float(roi["y"])
                elif "easting" in roi and "northing" in roi:
                    x, y = float(roi["easting"]), float(roi["northing"])
                elif "lat" in roi and "lon" in roi:
                    x, y = transformer.transform(float(roi["lon"]), float(roi["lat"]))
                elif "lat" in roi and "lng" in roi:
                    x, y = transformer.transform(float(roi["lng"]), float(roi["lat"]))
                else:
                    continue
            else:
                continue

            det = gt[1] * gt[5] - gt[2] * gt[4]
            if abs(det) < 1e-12:
                continue
            px_col = int((gt[5] * (x - gt[0]) - gt[2] * (y - gt[3])) / det)
            px_row = int((gt[1] * (y - gt[3]) - gt[4] * (x - gt[0])) / det)

            if 0 <= px_col < cols and 0 <= px_row < rows:
                results.append((name, px_col, px_row))
        except Exception:
            continue

    return results


def read_roi_detections(
    filepath: Union[str, Path],
    top_n: Optional[int] = None,
    min_prob: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Parses candidate detection coordinates from text file (new_roi_coordinates.txt).
    """
    path = Path(filepath)
    if not path.exists():
        return []

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("=") or line.startswith("-") or line.startswith("R"):
                continue
            m = re.match(
                r"^\s*(\d+)\s+(\S+)\s+([-\d.]+)\s+([-\d.]+)\s+(\d+)\s+([\d.]+)",
                line,
            )
            if m:
                rows.append({
                    "Rank": int(m.group(1)),
                    "Flight_Line": m.group(2),
                    "lat": float(m.group(3)),
                    "lon": float(m.group(4)),
                    "Pixels": int(m.group(5)),
                    "mean_prob": float(m.group(6)),
                })

    if min_prob is not None:
        rows = [r for r in rows if r["mean_prob"] >= min_prob]

    rows.sort(key=lambda r: r["mean_prob"], reverse=True)

    if top_n is not None and top_n > 0:
        rows = rows[:top_n]

    return rows
