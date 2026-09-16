"""Render the probability mosaic map PDF using the exact HSC Turbo-Black colormap from GitHub."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pbe_vae_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/pbe_vae_cache")

import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

from pbe_vae.utils.qgis_renderer import (
    hsc_mosaic_cmap,
    add_hsc_scale_bar,
    add_hsc_north_arrow,
)


def render_pdf(
    mosaic_tif: Path,
    out_pdf: Path,
    max_dim: int = 4000,
    dpi: int = 300,
):
    print(f"[+] Loading mosaic: {mosaic_tif}")
    ds = gdal.Open(str(mosaic_tif))
    if ds is None:
        raise FileNotFoundError(f"Cannot open raster: {mosaic_tif}")

    cols, rows = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    res_x = abs(gt[1])
    extent = [gt[0], gt[0] + cols * gt[1], gt[3] + rows * gt[5], gt[3]]

    # Scale for overview decimation to fit in memory
    scale = min(1.0, max_dim / max(cols, rows))
    buf_w = max(1, int(round(cols * scale)))
    buf_h = max(1, int(round(rows * scale)))

    print(f"[+] Reading raster overview: {buf_w}x{buf_h} (from {cols}x{rows})...")
    arr = ds.GetRasterBand(1).ReadAsArray(buf_xsize=buf_w, buf_ysize=buf_h).astype(np.float32)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    nodata = -9999.0 if nodata is None else nodata

    mask = ~np.isclose(arr, nodata, atol=1.0) & (arr > 0.0) & np.isfinite(arr)
    arr_disp = np.ma.masked_where(~mask, arr)

    fig, ax = plt.subplots(figsize=(12, 10))

    # EXACT HSC Turbo-Black Colormap from pbe_vae.utils.qgis_renderer
    cmap = hsc_mosaic_cmap(invert=False)

    im = ax.imshow(
        arr_disp,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=extent,
        origin="upper",
    )

    full_title = f"Target Detection Probability Mosaic Map\n(Spatial Resolution: {res_x:.1f} m/px)"
    ax.set_title(full_title, fontsize=13, pad=10, weight="bold")
    ax.set_xlabel("UTM Easting (m)", fontsize=10, weight="bold")
    ax.set_ylabel("UTM Northing (m)", fontsize=10, weight="bold")
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    add_hsc_scale_bar(ax, extent)
    add_hsc_north_arrow(ax, extent)

    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("Detection Probability [0, 1]", rotation=270, labelpad=15, fontsize=10, weight="bold")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_pdf), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Successfully exported PDF with original HSC colormap: {out_pdf}")


if __name__ == "__main__":
    out_dir = Path("/Users/hsc/Documents/ARIZONA_HS/pbe_vae_aml_experiment_2")
    mos_tif = out_dir / "prob_mosaic_masked.tif"
    if not mos_tif.exists():
        mos_tif = out_dir / "prob_mosaic.tif"

    pdf_main = out_dir / "prob_mosaic_map.pdf"
    pdf_sub = out_dir / "prob_mosaic" / "prob_mosaic_map.pdf"

    render_pdf(mos_tif, pdf_main)

    import shutil
    pdf_sub.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_main, pdf_sub)
    print(f"[+] Copied to subfolder: {pdf_sub}")

