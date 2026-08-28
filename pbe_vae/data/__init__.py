"""
pbe_vae.data
============
Data loading, geospatial manipulation, dataset sampling, and mosaic utilities.
"""

from pbe_vae.data.geo import (
    get_wavelengths,
    find_aviris_files,
    find_prob_tifs,
    pixel_to_latlon,
    latlon_to_pixel,
    roi_centres_in_image,
    read_roi_detections,
)
from pbe_vae.data.mosaic import (
    warp_probability_tile,
    batch_warp_tiles,
    build_vrt_mosaic,
    translate_vrt_to_geotiff,
    mask_zeros_to_nodata,
    assemble_probability_mosaic,
)
from pbe_vae.data.harvester import get_roi_extents, harvest_rois
from pbe_vae.data.dataset import (
    HyperspectralDataset,
    SpatialFlightlineSampler,
    harvest_background_data,
)

__all__ = [
    "get_wavelengths",
    "find_aviris_files",
    "find_prob_tifs",
    "pixel_to_latlon",
    "latlon_to_pixel",
    "roi_centres_in_image",
    "read_roi_detections",
    "warp_probability_tile",
    "batch_warp_tiles",
    "build_vrt_mosaic",
    "translate_vrt_to_geotiff",
    "mask_zeros_to_nodata",
    "assemble_probability_mosaic",
    "get_roi_extents",
    "harvest_rois",
    "HyperspectralDataset",
    "SpatialFlightlineSampler",
    "harvest_background_data",
]
