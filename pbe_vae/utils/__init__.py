"""
pbe_vae.utils
=============
Utility modules for metrics, configuration, cartography, and visualization.
"""

from pbe_vae.utils.config import load_config, save_config, ensure_dir
from pbe_vae.utils.metrics import (
    haversine_dist,
    compute_separability_metrics,
    compute_classification_metrics,
    compute_bbox_iou,
)
from pbe_vae.utils.cartography import (
    point_in_polygon,
    draw_bullseye_marker,
    draw_star_marker,
    burn_markers_to_raster,
    DEFAULT_MARKER_RADII,
    DEFAULT_KNOWN_COLORS,
    DEFAULT_NEW_COLORS,
)
from pbe_vae.utils.qgis_renderer import (
    init_qgis,
    apply_pseudocolor_ramp,
    apply_vector_outline_style,
    render_map_layout_to_pdf,
    export_annotated_map_pdf,
)
from pbe_vae.utils.visualization import (
    plot_diagnostic_dashboard,
    build_pdf_report,
)

__all__ = [
    "load_config",
    "save_config",
    "ensure_dir",
    "haversine_dist",
    "compute_separability_metrics",
    "compute_classification_metrics",
    "compute_bbox_iou",
    "point_in_polygon",
    "draw_bullseye_marker",
    "draw_star_marker",
    "burn_markers_to_raster",
    "DEFAULT_MARKER_RADII",
    "DEFAULT_KNOWN_COLORS",
    "DEFAULT_NEW_COLORS",
    "init_qgis",
    "apply_pseudocolor_ramp",
    "apply_vector_outline_style",
    "render_map_layout_to_pdf",
    "export_annotated_map_pdf",
    "plot_diagnostic_dashboard",
    "build_pdf_report",
]
