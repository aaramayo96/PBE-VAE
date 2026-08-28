"""
pbe_vae.data.harvester
======================
Extracts spatial-spectral ROI cubes and aligns digital elevation models (DEM)
across configured coordinate extents. Uses spatial bounding-box indexing to
rapidly identify and process ONLY the flightlines that physically intersect the ROIs.
"""

import os
import sys
import numpy as np
from pathlib import Path
from pyproj import Transformer
from osgeo import gdal
from tqdm import tqdm
from typing import Dict, Any, Union, List, Optional, Tuple

# Clean QGIS path if present to prevent C-extension conflicts
sys.path = [p for p in sys.path if "QGIS" not in p]

from pbe_vae.data.geo import get_wavelengths, find_aviris_files, open_georeferenced_dataset
from pbe_vae.data.spectral import align_spectral_cube, selected_band_indices

_DEFAULT_REFL_CLIP = (0.0, 1.5)


def normalize_reflectance(arr: np.ndarray, clip_bounds: Tuple[float, float] = (0.0, 1.5)) -> np.ndarray:
    """
    Normalizes raw hyperspectral reflectance arrays.
    Automatically detects integer-scaled reflectance (e.g. 0-10000 or %*100)
    and rescales down to standard physical reflectance [0.0, 1.0].
    """
    arr_f = np.nan_to_num(arr.astype(np.float32))
    pos_mask = (arr_f > 0.0) & (arr_f != -9999.0)
    if np.any(pos_mask):
        p99 = float(np.percentile(arr_f[pos_mask], 99))
        if p99 > 50.0:
            arr_f = arr_f / 10000.0
        elif p99 > 5.0:
            arr_f = arr_f / 100.0
    return np.clip(arr_f, *clip_bounds)


def get_dataset_bounds(ds: gdal.Dataset) -> Tuple[float, float, float, float]:
    """Returns (min_x, min_y, max_x, max_y) bounding box of a GDAL dataset."""
    gt = ds.GetGeoTransform()
    cols, rows = ds.RasterXSize, ds.RasterYSize
    x1, y1 = gt[0], gt[3]
    x2 = gt[0] + cols * gt[1] + rows * gt[2]
    y2 = gt[3] + cols * gt[4] + rows * gt[5]
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def boxes_intersect(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> bool:
    """Checks if two 2D bounding boxes (min_x, min_y, max_x, max_y) overlap."""
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])


def get_roi_extents(
    cfg: Dict[str, Any],
    roi_radius_pixels: Optional[int] = None,
    pixel_size_meters: float = 5.0,
    roi_radius_meters: Optional[float] = None,
    target_crs: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int, float]:
    """
    Converts Lat/Lon or Projected ROIs into bounding boxes based on pixel radius.
    """
    crs = target_crs or cfg.get("processing", {}).get("target_crs", "EPSG:32611")

    # Determine radius in pixels & calculate meter conversion
    if roi_radius_pixels is not None:
        radius_px = int(roi_radius_pixels)
    elif "roi_radius_pixels" in cfg.get("processing", {}):
        radius_px = int(cfg["processing"]["roi_radius_pixels"])
    elif roi_radius_meters is not None:
        radius_px = int(round(roi_radius_meters / pixel_size_meters))
    else:
        radius_m_cfg = float(cfg.get("processing", {}).get("roi_radius_meters", 200.0))
        radius_px = int(round(radius_m_cfg / pixel_size_meters))

    radius_m = float(radius_px * pixel_size_meters)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    roi_extents = []
    for i, r in enumerate(cfg.get("rois", []), 1):
        if isinstance(r, (list, tuple)):
            v1, v2 = float(r[0]), float(r[1])
            name = f"ROI_{i:02d}"
            if v1 > 1000 or v2 > 1000:
                cx, cy = v1, v2
            else:
                lon_val, lat_val = (v1, v2) if abs(v1) > abs(v2) else (v2, v1)
                cx, cy = transformer.transform(lon_val, lat_val)
        elif isinstance(r, dict):
            name = r.get("name", f"Site_{i:02d}")
            if "x" in r and "y" in r:
                cx, cy = float(r["x"]), float(r["y"])
            elif "easting" in r and "northing" in r:
                cx, cy = float(r["easting"]), float(r["northing"])
            elif "lon" in r and "lat" in r:
                cx, cy = transformer.transform(float(r["lon"]), float(r["lat"]))
            else:
                continue
        else:
            continue

        extent = (cx - radius_m, cy - radius_m, cx + radius_m, cy + radius_m)
        roi_extents.append(
            {"id": i, "extent": extent, "name": name, "cx": cx, "cy": cy}
        )
    return roi_extents, radius_px, radius_m


def harvest_rois(
    cfg: Dict[str, Any],
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
    """
    Harvests spectral-DEM pixels across all configured ROI extents and flight lines.
    Uses spatial bounding-box indexing to process ONLY the flight lines that intersect the ROIs.
    """
    out_dir = Path(output_dir or cfg.get("paths", {}).get("out_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)

    cubes_dir = out_dir / "roi_cubes"
    if save_cubes:
        cubes_dir.mkdir(parents=True, exist_ok=True)

    crs = target_crs or cfg.get("processing", {}).get("target_crs", "EPSG:32611")
    u_dem = use_dem if use_dem is not None else cfg.get("processing", {}).get("use_dem", True)
    d_path = dem_path or cfg.get("paths", {}).get("dem_path", "")
    clip_bounds = refl_clip or _DEFAULT_REFL_CLIP

    v_range = vnir_range or tuple(cfg.get("processing", {}).get("vnir_range", [0.38, 1.2]))
    s_range = swir_range or tuple(cfg.get("processing", {}).get("swir_range", [2.0, 2.5]))

    has_dem = bool(u_dem and d_path and Path(d_path).exists())

    files = find_aviris_files(cfg["paths"]["base_dir"])
    if not files:
        raise FileNotFoundError(
            f"No AVIRIS flight lines found in {cfg['paths']['base_dir']}"
        )

    # 1. Determine pixel ground resolution from the first flightline raster
    ds_sample = open_georeferenced_dataset(files[0][1], files[0][2])
    if ds_sample:
        gt_sample = ds_sample.GetGeoTransform()
        pixel_size_x = abs(gt_sample[1]) if gt_sample[1] != 0 else 5.0
        pixel_size_y = abs(gt_sample[5]) if gt_sample[5] != 0 else 5.0
        pixel_size_m = float((pixel_size_x + pixel_size_y) / 2.0)
        ds_sample = None
    else:
        pixel_size_m = 5.0

    # 2. Convert ROI radius from pixels to meters
    rois, radius_px, radius_m = get_roi_extents(
        cfg=cfg,
        roi_radius_pixels=roi_radius_pixels,
        pixel_size_meters=pixel_size_m,
        roi_radius_meters=roi_radius_meters,
        target_crs=crs,
    )

    full_wavelength_array = get_wavelengths(files[0][2])
    fused_idxs = selected_band_indices(full_wavelength_array, cfg.get("processing", {}))
    active_wavelengths = full_wavelength_array[fused_idxs]
    v_count = int(np.sum((active_wavelengths >= v_range[0]) & (active_wavelengths <= v_range[1])))
    s_count = int(np.sum((active_wavelengths >= s_range[0]) & (active_wavelengths <= s_range[1])))

    # 3. Print Pixel-to-Meter Conversion Summary
    window_px = 2 * radius_px
    window_m = 2.0 * radius_m
    print(
        f"[+] ROI Extent: {radius_px} px radius | Pixel Size: {pixel_size_m:.2f} m/px -> "
        f"Ground Radius: {radius_m:.1f} m ({window_px}×{window_px} px window = {window_m:.1f} m × {window_m:.1f} m ground footprint)"
    )
    print(
        f"[+] Active Bands: {v_count} VNIR ({v_range[0]}-{v_range[1]}µm) + "
        f"{s_count} SWIR ({s_range[0]}-{s_range[1]}µm) = {len(fused_idxs)} channels | "
        f"CRS: {crs} | Use DEM: {has_dem}"
    )

    # 4. Spatial Indexing: Map which flightlines physically intersect the target ROIs
    active_flightline_jobs = []
    for key, refl_dat, refl_hdr in files:
        src_ds = open_georeferenced_dataset(refl_dat, refl_hdr)
        if src_ds is None:
            continue
        fl_bounds = get_dataset_bounds(src_ds)
        matching_rois = [roi for roi in rois if boxes_intersect(fl_bounds, roi["extent"])]
        if matching_rois:
            active_flightline_jobs.append((key, refl_dat, refl_hdr, src_ds, matching_rois))

    skipped_count = len(files) - len(active_flightline_jobs)
    print(
        f"[+] Spatial Mapping: {len(active_flightline_jobs)} flightline(s) intersect the {len(rois)} ROI(s) "
        f"(Filtered out {skipped_count} non-overlapping flightlines)."
    )

    X_target, id_target = [], []

    # 5. Process ONLY intersecting flightlines
    for key, refl_dat, refl_hdr, src_ds, matching_rois in tqdm(
        active_flightline_jobs, desc="Harvesting intersecting flightlines"
    ):
        source_wavelengths = get_wavelengths(refl_hdr)
        for roi in matching_rois:
            try:
                ds_hsi = gdal.Warp(
                    "/vsimem/h.tif",
                    src_ds,
                    options=gdal.WarpOptions(
                        dstSRS=crs,
                        outputBounds=roi["extent"],
                        srcNodata=0,
                        dstNodata=-9999,
                    ),
                )
                if not ds_hsi:
                    continue

                h_array = ds_hsi.ReadAsArray().astype(np.float32)
                h_cube = np.transpose(h_array, (1, 2, 0))
                h_aligned = align_spectral_cube(
                    h_cube, source_wavelengths, active_wavelengths
                )
                h_flat = h_aligned.reshape(-1, h_aligned.shape[-1])
                valid_mask = (
                    h_array[h_array.shape[0] // 2].ravel() != -9999.0
                )

                if has_dem:
                    ds_dem = gdal.Warp(
                        "/vsimem/d.tif",
                        str(d_path),
                        options=gdal.WarpOptions(
                            width=ds_hsi.RasterXSize,
                            height=ds_hsi.RasterYSize,
                            outputBounds=roi["extent"],
                            dstSRS=crs,
                        ),
                    )
                    d_array = ds_dem.ReadAsArray() if ds_dem else np.zeros((h_array.shape[1], h_array.shape[2]), dtype=np.float32)
                    d_flat = d_array.flatten().reshape(-1, 1)
                else:
                    d_array = None
                    d_flat = None

                if np.any(valid_mask):
                    spectral_subset = normalize_reflectance(
                        h_flat[valid_mask], clip_bounds
                    )

                    if has_dem and d_flat is not None:
                        dem_subset = d_flat[valid_mask]
                        X_target.append(np.hstack((spectral_subset, dem_subset)))
                    else:
                        X_target.append(spectral_subset)

                    id_target.extend([roi["id"]] * int(np.sum(valid_mask)))

                    if save_cubes:
                        h_3d = np.transpose(
                            normalize_reflectance(h_aligned, clip_bounds), (2, 0, 1)
                        )

                        if has_dem and d_array is not None:
                            d_3d = d_array[np.newaxis, :, :].astype(np.float32)
                            cube_3d = np.concatenate([h_3d, d_3d], axis=0)
                        else:
                            cube_3d = h_3d

                        cube_3d = np.transpose(cube_3d, (1, 2, 0))
                        valid_mask_2d = valid_mask.reshape(
                            h_array.shape[1], h_array.shape[2]
                        )

                        np.save(cubes_dir / f"roi_{roi['id']}_{key}_cube.npy", cube_3d)
                        np.save(cubes_dir / f"roi_{roi['id']}_{key}_mask.npy", valid_mask_2d)

                gdal.Unlink("/vsimem/h.tif")
                if has_dem:
                    gdal.Unlink("/vsimem/d.tif")

            except Exception as e:
                print(f"  [!] Skipping ROI {roi['id']} in {key}: {e}")
                continue

    if not X_target:
        raise ValueError(
            "No valid pixels harvested. Check ROI coordinates and flight line files."
        )

    X_all = np.vstack(X_target)
    save_path = out_dir / "harvested_data.npz"
    np.savez(
        save_path,
        X=X_all,
        ids=np.array(id_target),
        idxs=fused_idxs,
        active_wavelengths=active_wavelengths,
        has_dem=has_dem,
        roi_radius_pixels=radius_px,
        roi_radius_meters=radius_m,
        pixel_size_meters=pixel_size_m,
        target_crs=crs,
    )

    print(
        f"[+] Successfully harvested {len(X_all):,} pixels from {len(np.unique(id_target))} ROI(s)."
    )
    print(f"[+] Saved dataset to: {save_path}")

    return {
        "harvested_path": str(save_path),
        "n_pixels": len(X_all),
        "fused_idxs": fused_idxs,
        "active_wavelengths": active_wavelengths,
        "n_rois": len(np.unique(id_target)),
        "has_dem": has_dem,
        "roi_radius_pixels": radius_px,
        "roi_radius_meters": radius_m,
        "pixel_size_meters": pixel_size_m,
        "target_crs": crs,
    }
