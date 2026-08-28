"""
pbe_vae.utils.cartography
=========================
Cartographic raster marker drawing and annotation overlay generation functions.
"""

import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional
import numpy as np
from osgeo import gdal
from pbe_vae.data.geo import latlon_to_pixel

# Default Bullseye radii in pixels (outer -> inner)
# At 100 m/px: 60 px ~ 6 km, visible on large 200 km maps
DEFAULT_MARKER_RADII = [60, 44, 30, 18, 7]

# Default (R, G, B, A) uint8 color rings
DEFAULT_KNOWN_COLORS = [
    (0, 0, 0, 255),        # outer black ring
    (0, 230, 230, 255),    # cyan
    (0, 0, 0, 255),        # black separator
    (100, 255, 255, 255),  # bright aqua
    (255, 255, 255, 255),  # white center
]

DEFAULT_NEW_COLORS = [
    (0, 0, 0, 255),        # outer black ring
    (50, 255, 0, 255),     # neon green
    (0, 0, 0, 255),        # black separator
    (180, 255, 80, 255),   # yellow-green
    (255, 255, 255, 255),  # white center
]


def point_in_polygon(x: float, y: float, polygon_vertices: List[Tuple[float, float]]) -> bool:
    """
    Ray-casting test to check whether (x, y) is inside a 2D polygon.
    """
    n = len(polygon_vertices)
    inside = False
    px, py = polygon_vertices[-1]
    for qx, qy in polygon_vertices:
        if ((py > y) != (qy > y)) and (x < (qx - px) * (y - py) / (qy - py) + px):
            inside = not inside
        px, py = qx, qy
    return inside


def draw_bullseye_marker(
    rgba: np.ndarray,
    row: int,
    col: int,
    radii: Optional[List[int]] = None,
    colors: Optional[List[Tuple[int, int, int, int]]] = None
) -> None:
    """
    Draws concentric filled circular rings (outer -> inner) at (row, col).
    Modifies (H, W, 4) uint8 RGBA array in place.
    """
    if radii is None:
        radii = DEFAULT_MARKER_RADII
    if colors is None:
        colors = DEFAULT_NEW_COLORS

    h, w = rgba.shape[:2]
    for r, c in zip(radii, colors):
        rmin = max(0, row - r)
        rmax = min(h, row + r + 1)
        cmin = max(0, col - r)
        cmax = min(w, col + r + 1)
        for rr in range(rmin, rmax):
            for cc in range(cmin, cmax):
                if (rr - row) ** 2 + (cc - col) ** 2 <= r * r:
                    rgba[rr, cc] = c


def draw_star_marker(
    rgba: np.ndarray,
    row: int,
    col: int,
    outer_radius: int = 60,
    inner_radius: Optional[int] = None,
    colors: Optional[List[Tuple[int, int, int, int]]] = None,
    n_points: int = 5
) -> None:
    """
    Draws a filled n-pointed star marker with dark outline and center dot at (row, col).
    Modifies (H, W, 4) uint8 RGBA array in place.
    """
    if inner_radius is None:
        inner_radius = max(6, outer_radius // 3)
    if colors is None:
        colors = DEFAULT_KNOWN_COLORS

    h, w = rgba.shape[:2]

    # Precompute star polygon vertices
    verts = []
    for i in range(2 * n_points):
        angle = math.pi / n_points * i - math.pi / 2
        r = outer_radius if i % 2 == 0 else inner_radius
        verts.append((r * math.cos(angle), r * math.sin(angle)))

    # Star fill
    for rr in range(max(0, row - outer_radius - 1), min(h, row + outer_radius + 2)):
        for cc in range(max(0, col - outer_radius - 1), min(w, col + outer_radius + 2)):
            dy = rr - row
            dx = cc - col
            if point_in_polygon(dx, dy, verts):
                rgba[rr, cc] = colors[1]

    # Dark border
    big_verts = [(x * 1.25, y * 1.25) for x, y in verts]
    for rr in range(max(0, row - outer_radius - 2), min(h, row + outer_radius + 3)):
        for cc in range(max(0, col - outer_radius - 2), min(w, col + outer_radius + 3)):
            dy = rr - row
            dx = cc - col
            if point_in_polygon(dx, dy, big_verts) and not point_in_polygon(dx, dy, verts):
                rgba[rr, cc] = colors[0]

    # White center dot
    center_r = max(1, inner_radius // 2)
    for rr in range(max(0, row - center_r), min(h, row + center_r + 1)):
        for cc in range(max(0, col - center_r), min(w, col + center_r + 1)):
            if (rr - row) ** 2 + (cc - col) ** 2 <= center_r * center_r:
                rgba[rr, cc] = colors[-1]


def burn_markers_to_raster(
    reference_mosaic_path: Union[str, Path],
    output_annot_path: Union[str, Path],
    known_rois: Optional[List[Dict[str, Any]]] = None,
    new_rois: Optional[List[Dict[str, Any]]] = None,
    marker_radii: Optional[List[int]] = None,
    known_colors: Optional[List[Tuple[int, int, int, int]]] = None,
    new_colors: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Path:
    """
    Creates a transparent 4-band RGBA GeoTIFF overlay with burned cartographic markers.
    """
    ref_ds = gdal.Open(str(reference_mosaic_path))
    if ref_ds is None:
        raise RuntimeError(f"Cannot open reference raster: {reference_mosaic_path}")

    gt = ref_ds.GetGeoTransform()
    proj_wkt = ref_ds.GetProjection()
    h = ref_ds.RasterYSize
    w = ref_ds.RasterXSize
    ref_ds = None

    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # 1. Burn known ROI stars
    if known_rois:
        outer_r = (marker_radii or DEFAULT_MARKER_RADII)[0]
        for r in known_rois:
            lat, lon = r["lat"], r["lon"]
            col, row = latlon_to_pixel(lat, lon, gt, proj_wkt)
            if 0 <= row < h and 0 <= col < w:
                draw_star_marker(
                    rgba, row, col,
                    outer_radius=outer_r,
                    colors=known_colors or DEFAULT_KNOWN_COLORS
                )

    # 2. Burn new ROI bullseyes
    if new_rois:
        for r in new_rois:
            lat, lon = r["lat"], r["lon"]
            col, row = latlon_to_pixel(lat, lon, gt, proj_wkt)
            if 0 <= row < h and 0 <= col < w:
                draw_bullseye_marker(
                    rgba, row, col,
                    radii=marker_radii or DEFAULT_MARKER_RADII,
                    colors=new_colors or DEFAULT_NEW_COLORS
                )

    out_path = Path(output_annot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(out_path), w, h, 4, gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER", "ALPHA=YES"]
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj_wkt)
    for b in range(4):
        band = ds.GetRasterBand(b + 1)
        band.WriteArray(rgba[:, :, b])
        band.FlushCache()
    ds = None

    return out_path

