"""
pbe_vae.utils.qgis_renderer
===========================
Functions for headless QGIS map composition, layer styling, and PDF publishing.
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any


_HSC_TURBO_BLACK_STOPS = [
    (0.00, (0, 0, 0)),
    (0.02, (48, 18, 59)),
    (0.13, (65, 90, 200)),
    (0.25, (33, 145, 235)),
    (0.38, (30, 190, 200)),
    (0.50, (140, 220, 80)),
    (0.63, (230, 210, 40)),
    (0.75, (250, 150, 30)),
    (0.87, (220, 65, 20)),
    (1.00, (122, 4, 3)),
]


def hsc_mosaic_cmap(invert: bool = False):
    """Return the HSC mosaic ramp, optionally reversed without changing its colors."""
    from matplotlib.colors import LinearSegmentedColormap

    positions = [position for position, _ in _HSC_TURBO_BLACK_STOPS]
    colors = [tuple(channel / 255.0 for channel in color) for _, color in _HSC_TURBO_BLACK_STOPS]
    cmap = LinearSegmentedColormap.from_list("hsc_turbo_black", list(zip(positions, colors)))
    if invert:
        # SAM lower angle means a closer match.  Reverse the exact ACE/PBE
        # ramp rather than using a separate approximate palette.
        cmap = cmap.reversed(name="hsc_turbo_black_reversed")
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    return cmap


def add_hsc_scale_bar(ax, extent):
    """Add the scale bar used by the HSC mosaic previews."""
    import matplotlib.pyplot as plt

    xmin, xmax, ymin, ymax = extent
    width_m = xmax - xmin
    height_m = ymax - ymin
    length_m = next((value for value in [100000, 50000, 20000, 10000, 5000, 2000, 1000, 500, 200, 100] if width_m / value >= 3), width_m / 4)
    x_pad, y_pad = width_m * 0.04, height_m * 0.04
    sb_xmin, sb_ymin = xmin + x_pad, ymin + y_pad
    sb_ymax = sb_ymin + height_m * 0.015
    background = plt.Rectangle(
        (xmin + x_pad * 0.5, ymin + y_pad * 0.5),
        length_m + x_pad,
        y_pad * 1.2 + (sb_ymax - sb_ymin),
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
        zorder=10,
    )
    ax.add_patch(background)
    midpoint = sb_xmin + length_m / 2
    ax.add_patch(plt.Rectangle((sb_xmin, sb_ymin), length_m / 2, sb_ymax - sb_ymin, facecolor="black", edgecolor="black", zorder=11))
    ax.add_patch(plt.Rectangle((midpoint, sb_ymin), length_m / 2, sb_ymax - sb_ymin, facecolor="white", edgecolor="black", zorder=11))
    label = f"{length_m / 1000:.0f} km" if length_m >= 1000 else f"{length_m:.0f} m"
    ax.text(sb_xmin + length_m / 2, sb_ymax + height_m * 0.008, label, color="black", weight="bold", fontsize=8, ha="center", va="bottom", zorder=12)


def add_hsc_north_arrow(ax, extent):
    """Add the north arrow used by the HSC mosaic previews."""
    import matplotlib.pyplot as plt

    xmin, xmax, ymin, ymax = extent
    width_m, height_m = xmax - xmin, ymax - ymin
    arrow_w, arrow_h = width_m * 0.02, height_m * 0.05
    x_center, y_center = xmax - width_m * 0.06, ymax - height_m * 0.08
    top, bottom = (x_center, y_center + arrow_h / 2), (x_center, y_center - arrow_h / 2)
    left, right, center = (x_center - arrow_w / 2, y_center), (x_center + arrow_w / 2, y_center), (x_center, y_center)
    for points, facecolor in [([top, left, center], "black"), ([top, right, center], "white"), ([bottom, left, center], "white"), ([bottom, right, center], "black")]:
        ax.add_patch(plt.Polygon(points, facecolor=facecolor, edgecolor="black", zorder=11))
    ax.text(x_center, y_center + arrow_h / 2 + height_m * 0.008, "N", color="black", weight="bold", fontsize=11, ha="center", va="bottom", zorder=12)


def init_qgis(qgis_prefix: Optional[Union[str, Path]] = None):
    """
    Initializes QGIS standalone runtime application with environment safeguards.
    """
    try:
        from qgis.core import QgsApplication
    except ImportError:
        # If running in standard Python where QGIS isn't installed
        return None

    prefix = str(qgis_prefix or "/Applications/QGIS-LTR.app/Contents/MacOS")
    qgis_res = Path("/Applications/QGIS-LTR.app/Contents/Resources")
    if qgis_res.exists():
        os.environ.setdefault("PROJ_LIB", str(qgis_res / "proj"))
        os.environ.setdefault("GDAL_DATA", str(qgis_res / "gdal/share/gdal"))

    QgsApplication.setPrefixPath(prefix, True)
    qgs = QgsApplication([], False)
    qgs.initQgis()
    return qgs


def apply_pseudocolor_ramp(
    raster_layer,
    ramp_name: str = "Magma",
    fallback_ramp: str = "Reds",
    vmin: float = 0.0,
    vmax: float = 1.0
) -> bool:
    """
    Applies a continuous color ramp shader to a QGIS single-band raster layer.
    """
    try:
        from qgis.core import (
            QgsStyle, QgsColorRampShader, QgsRasterShader, QgsSingleBandPseudoColorRenderer
        )
        color_ramp = QgsStyle.defaultStyle().colorRamp(ramp_name)
        if color_ramp is None:
            color_ramp = QgsStyle.defaultStyle().colorRamp(fallback_ramp)

        fcn = QgsColorRampShader(vmin, vmax, color_ramp)
        fcn.setColorRampType(QgsColorRampShader.Interpolated)
        fcn.classifyColorRamp(classes=256)

        shader = QgsRasterShader()
        shader.setRasterShaderFunction(fcn)

        renderer = QgsSingleBandPseudoColorRenderer(
            raster_layer.dataProvider(), 1, shader
        )
        renderer.setClassificationMin(vmin)
        renderer.setClassificationMax(vmax)

        raster_layer.setRenderer(renderer)
        raster_layer.triggerRepaint()
        return True
    except Exception:
        return False


def apply_vector_outline_style(
    vector_layer,
    outline_color: str = "#8c8c8c",
    outline_width: str = "0.15"
) -> bool:
    """
    Applies transparent fill with clean thin outline to a QGIS polygon or line layer.
    """
    try:
        from qgis.core import QgsSimpleFillSymbolLayer, QgsSimpleLineSymbolLayer
        sym = vector_layer.renderer().symbol()
        if vector_layer.geometryType() == 2:  # Polygon
            sl = QgsSimpleFillSymbolLayer.create({
                "color": "0,0,0,0",
                "outline_color": outline_color,
                "outline_width": outline_width
            })
            if sl:
                sym.changeSymbolLayer(0, sl)
        elif vector_layer.geometryType() == 1:  # Line
            sl = QgsSimpleLineSymbolLayer.create({
                "color": outline_color,
                "line_width": outline_width
            })
            if sl:
                sym.changeSymbolLayer(0, sl)
        vector_layer.triggerRepaint()
        return True
    except Exception:
        return False


def render_map_layout_to_pdf(
    layers: List[Any],
    output_pdf: Union[str, Path],
    dpi: int = 300,
    page_width_mm: float = 287.0,
    page_height_mm: float = 200.0
) -> bool:
    """
    Renders map layers to a publication-quality PDF using QgsPrintLayout.
    """
    try:
        from qgis.core import (
            QgsProject, QgsPrintLayout, QgsLayoutItemMap,
            QgsLayoutPoint, QgsLayoutSize, QgsLayoutExporter, QgsUnitTypes
        )
        import PyQt5.QtGui as QtGui

        project = QgsProject.instance()
        layout = QgsPrintLayout(project)
        layout.initializeDefaults()
        layout.setName("PBEVAEMapLayout")

        map_item = QgsLayoutItemMap(layout)
        map_item.setRect(20, 20, 20, 20)
        # Reverse layers so top of stack draws on top
        map_item.setLayers(list(reversed(layers)))

        primary_extent = layers[0].extent()
        primary_extent.scale(1.05)
        map_item.setExtent(primary_extent)
        layout.addLayoutItem(map_item)

        map_item.attemptMove(QgsLayoutPoint(5.0, 5.0, QgsUnitTypes.LayoutMillimeters))
        map_item.attemptResize(QgsLayoutSize(page_width_mm, page_height_mm, QgsUnitTypes.LayoutMillimeters))
        map_item.setBackgroundColor(QtGui.QColor("white"))

        exporter = QgsLayoutExporter(layout)
        settings = QgsLayoutExporter.PdfExportSettings()
        settings.dpi = dpi

        out_path = Path(output_pdf)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        res = exporter.exportToPdf(str(out_path), settings)
        return res == QgsLayoutExporter.Success
    except Exception:
        return False


def render_matplotlib_map_pdf(
    mosaic_tif: Union[str, Path],
    out_pdf: Union[str, Path],
    title: str = "Target Detection Probability Mosaic Map",
    cmap: str = "magma",
    cbar_label: str = "Probability [0, 1]",
    dpi: int = 300,
    zero_is_nodata: bool = True,
    invert_colorbar: bool = False,
) -> bool:
    """High-resolution PDF map generator for mosaic rasters as cartographic images."""
    import matplotlib.pyplot as plt
    import numpy as np
    from osgeo import gdal

    ds = gdal.Open(str(mosaic_tif))
    if ds is None:
        return False
    arr = ds.ReadAsArray().astype(np.float32)
    nodata = ds.GetRasterBand(1).GetNoDataValue() or -9999.0
    gt = ds.GetGeoTransform()
    rows, cols = ds.RasterYSize, ds.RasterXSize

    res_x = abs(gt[1])
    extent = [gt[0], gt[0] + cols * gt[1], gt[3] + rows * gt[5], gt[3]]

    mask = ~np.isclose(arr, nodata, atol=1.0) & (arr >= 0.0) & ~np.isnan(arr)
    if zero_is_nodata:
        mask &= arr != 0.0
    arr_disp = np.ma.masked_where(~mask, arr)

    fig, ax = plt.subplots(figsize=(10, 8))

    vmax = 1.0 if ("prob" in cmap.lower() or "prob" in title.lower()) else (np.percentile(arr[mask], 99.5) if np.any(mask) else 1.0)
    im = ax.imshow(
        arr_disp,
        cmap=hsc_mosaic_cmap(invert=invert_colorbar),
        vmin=0.0,
        vmax=max(float(vmax), 1e-6),
        extent=extent,
        origin="upper",
    )

    full_title = f"{title}\n(Spatial Resolution: {res_x:.1f} m/px)"
    ax.set_title(full_title, fontsize=12, pad=8)
    ax.set_xlabel("UTM Easting (m)", fontsize=9, weight="bold")
    ax.set_ylabel("UTM Northing (m)", fontsize=9, weight="bold")
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    add_hsc_scale_bar(ax, extent)
    add_hsc_north_arrow(ax, extent)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, rotation=270, labelpad=15, fontsize=9, weight="bold")

    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_pdf), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def render_dem_elevation_map(
    dem_tif: Union[str, Path],
    reference_tif: Union[str, Path],
    out_pdf: Union[str, Path],
    dpi: int = 300,
) -> bool:
    """Render a 300 dpi terrain PDF clipped to a detection mosaic footprint."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from osgeo import gdal

    reference = gdal.Open(str(reference_tif))
    if reference is None:
        return False

    gt = reference.GetGeoTransform()
    projection = reference.GetProjection()
    width, height = reference.RasterXSize, reference.RasterYSize
    bounds = (gt[0], gt[3] + height * gt[5], gt[0] + width * gt[1], gt[3])
    reference = None

    warped_path = "/vsimem/pbe_vae_dem_plot.tif"
    dem = gdal.Warp(
        warped_path,
        str(dem_tif),
        options=gdal.WarpOptions(
            format="GTiff",
            dstSRS=projection,
            outputBounds=bounds,
            width=width,
            height=height,
            resampleAlg="average",
            dstNodata=-9999.0,
        ),
    )
    if dem is None:
        return False

    band = dem.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    values = band.ReadAsArray().astype(np.float32)
    dem = None
    gdal.Unlink(warped_path)

    invalid = ~np.isfinite(values)
    if nodata is not None:
        invalid |= values == nodata
    terrain = np.ma.masked_where(invalid, values)
    if terrain.count() == 0:
        return False

    vmin, vmax = np.percentile(terrain.compressed(), [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1.0

    xmin, ymin, xmax, ymax = bounds
    extent = [xmin, xmax, ymin, ymax]
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = mpl.colormaps["terrain"].copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    image = ax.imshow(
        terrain,
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        extent=extent,
        origin="upper",
        zorder=1,
    )

    elevation_range = float(vmax - vmin)
    contour_interval = next(
        (value for value in [5, 10, 20, 25, 50, 100, 200] if elevation_range / value <= 15),
        500,
    )
    first_level = np.floor(vmin / contour_interval) * contour_interval
    levels = np.arange(first_level, vmax + contour_interval, contour_interval)
    if len(levels) > 1:
        xs = xmin + (np.arange(width) + 0.5) * gt[1]
        ys = ymax + (np.arange(height) + 0.5) * gt[5]
        contours = ax.contour(
            xs,
            ys,
            terrain,
            levels=levels,
            colors="#3b2a1a",
            linewidths=0.4,
            alpha=0.6,
            zorder=2,
        )
        ax.clabel(contours, inline=True, fontsize=6, fmt="%d m")

    pad_x = (xmax - xmin) * 0.03
    pad_y = (ymax - ymin) * 0.03
    plot_extent = [xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y]
    ax.set_xlim(plot_extent[0], plot_extent[1])
    ax.set_ylim(plot_extent[2], plot_extent[3])
    ax.set_aspect("equal")
    ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("UTM Easting (m)", fontsize=9, weight="bold")
    ax.set_ylabel("UTM Northing (m)", fontsize=9, weight="bold")
    ax.tick_params(axis="both", labelsize=8)
    add_hsc_scale_bar(ax, plot_extent)
    add_hsc_north_arrow(ax, plot_extent)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("Elevation (m)", rotation=270, labelpad=15, fontsize=9, weight="bold")

    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_pdf), dpi=dpi, bbox_inches="tight", format="pdf")
    plt.close(fig)
    return True


def export_comparison_map_pdf(
    pbe_mosaic_tif: Union[str, Path],
    ace_mosaic_tif: Union[str, Path],
    out_pdf: Union[str, Path],
    dpi: int = 300,
) -> bool:
    """
    Renders a high-resolution side-by-side comparative PDF map between
    PBE-VAE Detection Probability and classical ACE Detector response mosaics.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from osgeo import gdal

    ds_pbe = gdal.Open(str(pbe_mosaic_tif))
    ds_ace = gdal.Open(str(ace_mosaic_tif))
    if ds_pbe is None or ds_ace is None:
        return False

    arr_pbe = ds_pbe.ReadAsArray().astype(np.float32)
    arr_ace = ds_ace.ReadAsArray().astype(np.float32)

    nodata_pbe = ds_pbe.GetRasterBand(1).GetNoDataValue() or -9999.0
    nodata_ace = ds_ace.GetRasterBand(1).GetNoDataValue() or -9999.0

    gt = ds_pbe.GetGeoTransform()
    rows, cols = ds_pbe.RasterYSize, ds_pbe.RasterXSize
    res_x = abs(gt[1])
    extent = [gt[0], gt[0] + cols * gt[1], gt[3] + rows * gt[5], gt[3]]

    mask_pbe = ~np.isclose(arr_pbe, nodata_pbe, atol=1.0) & (arr_pbe >= 0.0) & (arr_pbe != 0.0) & ~np.isnan(arr_pbe)
    mask_ace = ~np.isclose(arr_ace, nodata_ace, atol=1.0) & (arr_ace >= 0.0) & (arr_ace != 0.0) & ~np.isnan(arr_ace)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)
    fig.patch.set_facecolor("#181818")

    # 1. PBE-VAE Detection Map
    axes[0].set_facecolor("#0f0f0f")
    arr_pbe_disp = np.ma.masked_where(~mask_pbe, arr_pbe)
    im1 = axes[0].imshow(arr_pbe_disp, cmap="magma", vmin=0.0, vmax=1.0, extent=extent, origin="upper")
    axes[0].set_title(f"PBE-VAE: Calibrated Target Probability [0, 1]\n(Resolution: {res_x:.1f} m/px)", fontsize=13, fontweight="bold", pad=12, color="white")
    axes[0].set_xlabel("Easting / X (m)", fontsize=11, color="white")
    axes[0].set_ylabel("Northing / Y (m)", fontsize=11, color="white")
    axes[0].tick_params(colors="white")
    axes[0].grid(True, linestyle="--", alpha=0.25, color="gray")
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Target Detection Probability", fontsize=11, color="white")
    cbar1.ax.tick_params(colors="white")

    # 2. ACE Detection Map
    axes[1].set_facecolor("#0f0f0f")
    arr_ace_disp = np.ma.masked_where(~mask_ace, arr_ace)
    ace_max = np.percentile(arr_ace[mask_ace], 99.5) if np.any(mask_ace) else 1.0
    im2 = axes[1].imshow(arr_ace_disp, cmap="viridis", vmin=0.0, vmax=max(float(ace_max), 0.5), extent=extent, origin="upper")
    axes[1].set_title(f"ACE: Adaptive Coherence Estimator Score\n(Resolution: {res_x:.1f} m/px)", fontsize=13, fontweight="bold", pad=12, color="white")
    axes[1].set_xlabel("Easting / X (m)", fontsize=11, color="white")
    axes[1].tick_params(colors="white")
    axes[1].grid(True, linestyle="--", alpha=0.25, color="gray")
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("ACE Cosine / Coherence Value", fontsize=11, color="white")
    cbar2.ax.tick_params(colors="white")

    fig.suptitle(f"Hyperspectral Target Detection Mosaic Comparison: PBE-VAE vs. ACE | Spatial Resolution: {res_x:.1f} m/px", fontsize=16, fontweight="bold", color="white")
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_pdf), dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def export_three_way_comparison_map_pdf(
    pbe_mosaic_tif: Union[str, Path],
    ace_mosaic_tif: Union[str, Path],
    sam_mosaic_tif: Union[str, Path],
    out_pdf: Union[str, Path],
    dpi: int = 300,
) -> bool:
    """Render PBE-VAE, ACE, and SAM mosaics in one comparison PDF."""
    import matplotlib.pyplot as plt
    import numpy as np
    from osgeo import gdal

    paths = [pbe_mosaic_tif, ace_mosaic_tif, sam_mosaic_tif]
    datasets = [gdal.Open(str(path)) for path in paths]
    if any(ds is None for ds in datasets):
        return False

    arrays = [ds.ReadAsArray().astype(np.float32) for ds in datasets]
    masks = []
    for index, (ds, arr) in enumerate(zip(datasets, arrays)):
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        nodata = -9999.0 if nodata is None else nodata
        mask = ~np.isclose(arr, nodata, atol=1.0) & (arr >= 0.0) & ~np.isnan(arr)
        if index != 2:
            mask &= arr != 0.0
        masks.append(mask)

    ds_ref = datasets[0]
    gt = ds_ref.GetGeoTransform()
    rows, cols = ds_ref.RasterYSize, ds_ref.RasterXSize
    extent = [gt[0], gt[0] + cols * gt[1], gt[3] + rows * gt[5], gt[3]]
    res_x = abs(gt[1])

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    panels = [
        ("PBE-VAE", "Calibrated Target Probability", 0.0, 1.0, "Target Probability", False),
        ("ACE", "Adaptive Coherence Estimator Score", 0.0, None, "ACE Score", False),
        ("SAM", "Spectral Angle Mapper (lower is better)", 0.0, None, "Spectral Angle (radians)", True),
    ]

    for ax, arr, mask, (name, title, vmin, vmax, cbar_label, invert) in zip(axes, arrays, masks, panels):
        display = np.ma.masked_where(~mask, arr)
        if vmax is None:
            vmax = float(np.percentile(arr[mask], 99.5)) if np.any(mask) else 1.0
            vmax = max(vmax, 1e-6)
        image = ax.imshow(display, cmap=hsc_mosaic_cmap(invert=invert), vmin=vmin, vmax=vmax, extent=extent, origin="upper")
        ax.set_title(f"{name}: {title}", fontsize=12, pad=8)
        ax.set_xlabel("UTM Easting (m)", fontsize=9, weight="bold")
        ax.set_ylabel("UTM Northing (m)", fontsize=9, weight="bold")
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        add_hsc_scale_bar(ax, extent)
        add_hsc_north_arrow(ax, extent)
        cbar = fig.colorbar(image, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label, rotation=270, labelpad=15, fontsize=9, weight="bold")

    fig.suptitle(
        f"Target-Detection Comparison - PBE-VAE, ACE, and SAM ({res_x:.1f} m/px)",
        fontsize=14,
    )
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_pdf), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def export_ace_sam_comparison_map_pdf(
    ace_mosaic_tif: Union[str, Path],
    sam_mosaic_tif: Union[str, Path],
    out_pdf: Union[str, Path],
    dpi: int = 300,
) -> bool:
    """Render the independent ACE and SAM mineral detectors in HSC report style."""
    import matplotlib.pyplot as plt
    import numpy as np
    from osgeo import gdal

    datasets = [gdal.Open(str(ace_mosaic_tif)), gdal.Open(str(sam_mosaic_tif))]
    if any(ds is None for ds in datasets):
        return False

    arrays = [ds.ReadAsArray().astype(np.float32) for ds in datasets]
    masks = []
    for index, (ds, arr) in enumerate(zip(datasets, arrays)):
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        nodata = -9999.0 if nodata is None else nodata
        mask = ~np.isclose(arr, nodata, atol=1.0) & (arr >= 0.0) & ~np.isnan(arr)
        if index == 0:
            mask &= arr != 0.0
        masks.append(mask)

    reference = datasets[0]
    gt = reference.GetGeoTransform()
    rows, cols = reference.RasterYSize, reference.RasterXSize
    extent = [gt[0], gt[0] + cols * gt[1], gt[3] + rows * gt[5], gt[3]]
    res_x = abs(gt[1])

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    panels = [
        ("ACE", "Adaptive Coherence Estimator Score", "ACE Score", False),
        ("SAM", "Spectral Angle Mapper (lower is better)", "Spectral Angle (radians)", True),
    ]
    for ax, arr, mask, (name, title, cbar_label, invert) in zip(axes, arrays, masks, panels):
        display = np.ma.masked_where(~mask, arr)
        vmax = float(np.percentile(arr[mask], 99.5)) if np.any(mask) else 1.0
        image = ax.imshow(display, cmap=hsc_mosaic_cmap(invert=invert), vmin=0.0, vmax=max(vmax, 1e-6), extent=extent, origin="upper")
        ax.set_title(f"{name}: {title}", fontsize=12, pad=8)
        ax.set_xlabel("UTM Easting (m)", fontsize=9, weight="bold")
        ax.set_ylabel("UTM Northing (m)", fontsize=9, weight="bold")
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        add_hsc_scale_bar(ax, extent)
        add_hsc_north_arrow(ax, extent)
        cbar = fig.colorbar(image, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label, rotation=270, labelpad=15, fontsize=9, weight="bold")

    fig.suptitle(
        f"Mineral Detection Comparison - ACE and SAM ({res_x:.1f} m/px)",
        fontsize=14,
    )
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_pdf), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def export_annotated_map_pdf(
    mosaic_tif: Union[str, Path],
    annot_tif: Optional[Union[str, Path]] = None,
    out_pdf: Optional[Union[str, Path]] = None,
    shp_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> bool:
    """
    High-level map renderer for probability mosaic with marker annotations and shapefile.
    Attempts headless QGIS first, falling back to Matplotlib PDF rendering.
    """
    try:
        from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer
        project = QgsProject.instance()
        project.removeAllMapLayers()
        layers = []

        rlayer = QgsRasterLayer(str(mosaic_tif), "Target Probability")
        if rlayer.isValid():
            project.addMapLayer(rlayer)
            apply_pseudocolor_ramp(rlayer, ramp_name="Magma", fallback_ramp="Reds")
            layers.append(rlayer)

        if shp_path and Path(shp_path).exists():
            vlayer = QgsVectorLayer(str(shp_path), "Flight Lines", "ogr")
            if vlayer.isValid():
                project.addMapLayer(vlayer)
                apply_vector_outline_style(vlayer)
                layers.append(vlayer)

        if annot_tif and Path(annot_tif).exists():
            alayer = QgsRasterLayer(str(annot_tif), "ROI Markers")
            if alayer.isValid():
                project.addMapLayer(alayer)
                layers.append(alayer)

        if layers and render_map_layout_to_pdf(layers, out_pdf, dpi=dpi):
            return True
    except Exception:
        pass

    # Matplotlib High-DPI Fallback
    return render_matplotlib_map_pdf(
        mosaic_tif=mosaic_tif,
        out_pdf=out_pdf,
        title="Hyperspectral Target Probability Map",
        cmap="magma",
        cbar_label="Detection Probability",
        dpi=dpi,
    )
