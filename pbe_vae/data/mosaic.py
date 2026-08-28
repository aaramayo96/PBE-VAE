"""
pbe_vae.data.mosaic
===================
Functions for reprojection, nodata masking, GDAL VRT building, and mosaic assembly.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict, Any
import numpy as np
from osgeo import gdal


def warp_probability_tile(
    src_tif: Union[str, Path],
    out_tif: Union[str, Path],
    target_epsg: str = "EPSG:32611",
    target_resolution: float = 100.0,
    nodata_val: float = -9999.0,
    gdal_warp_bin: Optional[Union[str, Path]] = None,
    overwrite: bool = True,
    zero_is_nodata: bool = True,
) -> bool:
    """
    Reprojects a single probability GeoTIFF to a standard target EPSG CRS (North-up)
    and uniform resolution. Corrects 0.0 margin padding to nodata.
    """
    src_path = Path(src_tif)
    out_path = Path(out_tif)

    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        return True

    if out_path.exists():
        out_path.unlink()

    # Sanity check if GDAL can open
    ds = gdal.Open(str(src_path))
    if ds is None:
        return False
    ds = None

    warp_exe = str(gdal_warp_bin) if gdal_warp_bin else "gdalwarp"

    src_nodata = "0 -9999" if zero_is_nodata else "-9999"
    cmd = [
        warp_exe,
        "-overwrite",
        "-t_srs", target_epsg,
        "-tr", str(target_resolution), str(target_resolution),
        "-r", "bilinear",
        "-srcnodata", src_nodata,
        "-dstnodata", str(nodata_val),
        "-multi",
        "-wo", "NUM_THREADS=ALL_CPUS",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        str(src_path),
        str(out_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not out_path.exists() or out_path.stat().st_size == 0:
            return False
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to GDAL Python API if binary is not in PATH
        try:
            ds_src = gdal.Open(str(src_path))
            out_ds = gdal.Warp(
                str(out_path),
                ds_src,
                dstSRS=target_epsg,
                xRes=target_resolution,
                yRes=target_resolution,
                resampleAlg=gdal.GRA_Bilinear,
                srcNodata=src_nodata,
                dstNodata=nodata_val,
                creationOptions=["COMPRESS=LZW", "TILED=YES"]
            )
            out_ds = None
            ds_src = None
            return out_path.exists() and out_path.stat().st_size > 0
        except Exception:
            if out_path.exists():
                out_path.unlink()
            return False


def batch_warp_tiles(
    src_tifs: List[Path],
    georef_dir: Union[str, Path],
    target_epsg: str = "EPSG:32611",
    target_resolution: float = 100.0,
    nodata_val: float = -9999.0,
    gdal_warp_bin: Optional[Union[str, Path]] = None,
    zero_is_nodata: bool = True,
) -> List[Path]:
    """
    Warp a collection of probability rasters into a georeferenced output directory.
    Returns list of successfully warped tile paths.
    """
    out_dir = Path(georef_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warped_tiles = []
    for src in src_tifs:
        out_tif = out_dir / f"{src.stem}_warped.tif"
        success = warp_probability_tile(
            src, out_tif,
            target_epsg=target_epsg,
            target_resolution=target_resolution,
            nodata_val=nodata_val,
            gdal_warp_bin=gdal_warp_bin,
            zero_is_nodata=zero_is_nodata,
        )
        if success:
            warped_tiles.append(out_tif)
    return warped_tiles


def build_vrt_mosaic(
    tile_paths: List[Path],
    vrt_path: Union[str, Path],
    nodata_val: float = -9999.0,
    target_resolution: float = 100.0,
) -> Path:
    """
    Builds a GDAL Virtual Raster (VRT) from multiple uniform tiles.
    """
    vrt_out = Path(vrt_path)
    vrt_out.parent.mkdir(parents=True, exist_ok=True)
    if vrt_out.exists():
        vrt_out.unlink()

    str_paths = [str(p) for p in tile_paths]
    vrt_ds = gdal.BuildVRT(
        str(vrt_out),
        str_paths,
        resolution="user",
        xRes=target_resolution,
        yRes=target_resolution,
        srcNodata=nodata_val,
        VRTNodata=nodata_val,
    )
    vrt_ds.FlushCache()
    vrt_ds = None
    return vrt_out


def translate_vrt_to_geotiff(
    vrt_path: Union[str, Path],
    out_tif: Union[str, Path],
    nodata_val: float = -9999.0
) -> Path:
    """
    Translates a GDAL VRT into a compressed monolithic GeoTIFF.
    """
    vrt_in = Path(vrt_path)
    out_path = Path(out_tif)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds_vrt = gdal.Open(str(vrt_in))
    if ds_vrt is None:
        raise RuntimeError(f"Cannot open VRT: {vrt_in}")

    translate_options = gdal.TranslateOptions(
        format="GTiff",
        noData=nodata_val,
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"]
    )
    out_ds = gdal.Translate(str(out_path), ds_vrt, options=translate_options)
    out_ds.FlushCache()
    out_ds = None
    ds_vrt = None
    return out_path


def mask_zeros_to_nodata(
    src_tif: Union[str, Path],
    dst_tif: Union[str, Path],
    nodata: float = -9999.0,
    zero_is_nodata: bool = True,
    minimum_value: Optional[float] = None,
    maximum_value: Optional[float] = None,
) -> Path:
    """
    Masks NoData/background values and optional detector threshold failures.
    """
    src_path = Path(src_tif)
    dst_path = Path(dst_tif)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    ds = gdal.Open(str(src_path))
    if ds is None:
        raise RuntimeError(f"Cannot open source raster: {src_path}")

    drv = gdal.GetDriverByName("GTiff")
    out = drv.CreateCopy(
        str(dst_path), ds, 0,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"]
    )
    for b_idx in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b_idx)
        arr = band.ReadAsArray().astype(np.float32)
        # Warping can turn the -9999 sentinel into values such as -9998.995
        # at tile edges; treat the whole sentinel neighborhood as NoData.
        invalid = np.isclose(arr, nodata, atol=1.0) | (arr < 0.0)
        if zero_is_nodata:
            invalid |= arr == 0.0
        if minimum_value is not None:
            invalid |= arr < float(minimum_value)
        if maximum_value is not None:
            invalid |= arr > float(maximum_value)
        arr[invalid] = nodata
        ob = out.GetRasterBand(b_idx)
        ob.SetNoDataValue(nodata)
        ob.WriteArray(arr)
        ob.FlushCache()
    out = None
    ds = None
    return dst_path


def assemble_probability_mosaic(
    prob_dir: Union[str, Path],
    out_dir: Union[str, Path],
    pattern: str = "*_PROB.tif",
    prefix: str = "prob_mosaic",
    target_epsg: str = "EPSG:32611",
    target_resolution: Optional[Union[float, str]] = None,
    nodata_val: float = -9999.0,
    gdal_warp_bin: Optional[Union[str, Path]] = None,
    zero_is_nodata: bool = True,
) -> Dict[str, Path]:
    """
    High-level orchestrator function to warp tiles, assemble VRT,
    translate to GeoTIFF, and generate masked GeoTIFF mosaic.
    Defaults to preserving the exact original native pixel resolution.
    """
    prob_path = Path(prob_dir)
    out_path = Path(out_dir)
    georef_dir = out_path / "georef_tiles"

    tifs = sorted(
        p for p in prob_path.glob(pattern)
        if not p.name.startswith("._") and p.is_file()
    )
    if not tifs:
        tifs = sorted(
            p for p in prob_path.rglob(pattern)
            if not p.name.startswith("._") and p.is_file()
        )
    if not tifs:
        raise FileNotFoundError(f"No {pattern} rasters found in {prob_path}")

    # Determine resolution: default to original native resolution from first raster
    final_res: float = 5.0
    if target_resolution is None or str(target_resolution).lower() in ["original", "native", "auto"]:
        try:
            ds_first = gdal.Open(str(tifs[0]))
            if ds_first is not None:
                gt_first = ds_first.GetGeoTransform()
                if abs(gt_first[1]) > 0:
                    final_res = float(abs(gt_first[1]))
                ds_first = None
        except Exception:
            final_res = 5.0
    else:
        final_res = float(target_resolution)

    warped_tiles = batch_warp_tiles(
        tifs, georef_dir,
        target_epsg=target_epsg,
        target_resolution=final_res,
        nodata_val=nodata_val,
        gdal_warp_bin=gdal_warp_bin,
        zero_is_nodata=zero_is_nodata,
    )

    if not warped_tiles:
        raise RuntimeError(f"No {pattern} tiles were successfully warped.")

    vrt_path = out_path / f"{prefix}.vrt"
    mosaic_tif = out_path / f"{prefix}.tif"
    masked_tif = out_path / f"{prefix}_masked.tif"

    build_vrt_mosaic(warped_tiles, vrt_path, nodata_val=nodata_val, target_resolution=final_res)
    translate_vrt_to_geotiff(vrt_path, mosaic_tif, nodata_val=nodata_val)
    mask_zeros_to_nodata(
        mosaic_tif,
        masked_tif,
        nodata=nodata_val,
        zero_is_nodata=zero_is_nodata,
    )

    return {
        "vrt": vrt_path,
        "mosaic_tif": mosaic_tif,
        "masked_tif": masked_tif,
        "n_tiles": len(warped_tiles)
    }
