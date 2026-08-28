"""
pbe_vae.models.baselines
========================
Input adapters and integration for official Spectral Python (`spectral.algorithms`) detectors:
  - algo.ACE: Adaptive Coherence / Cosine Estimator
  - algo.CEM: Constrained Energy Minimization
  - algo.MatchedFilter: Matched Filter Detector
  - algo.SAM: Spectral Angle Mapper

Adapts raw hyperspectral imagery (GDAL arrays, band order, nodata masks, wavelength subsetting)
to the input shapes and formats expected by the library algorithms.
"""

import numpy as np
from typing import Union, Optional, Tuple, Dict, Any

try:
    import spectral.algorithms as algo
    from spectral.algorithms import ACE, CEM, MatchedFilter, SAM
except ImportError:
    algo = None
    ACE = None
    CEM = None
    MatchedFilter = None
    SAM = None


def format_hsi_cube(
    data: np.ndarray,
    active_indices: Optional[np.ndarray] = None,
    source_wavelengths: Optional[np.ndarray] = None,
    target_wavelengths: Optional[np.ndarray] = None,
    refl_clip: Tuple[float, float] = (0.0, 1.5),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adapts raw hyperspectral image arrays to Spectral Python (SPy) format: (rows, cols, bands).

    Parameters:
        data: Input hyperspectral array in shape (bands, rows, cols) or (rows, cols, bands).
        active_indices: 1D array of band indices to slice (legacy fixed-layout mode).
        source_wavelengths: Physical wavelength centers for the source flightline.
        target_wavelengths: Trained model wavelengths. When supplied with
            source_wavelengths, every flightline is aligned to this grid.
        refl_clip: Min and max reflectance clipping bounds.

    Returns:
        cube: 3D float32 array in shape (rows, cols, bands).
        valid_mask: 2D boolean mask indicating non-nodata pixels.
    """
    arr = np.asarray(data, dtype=np.float32)

    # 1. Transpose from GDAL (bands, rows, cols) to SPy (rows, cols, bands) if needed
    if arr.ndim == 3:
        if arr.shape[0] < arr.shape[1] or (active_indices is not None and arr.shape[0] >= len(active_indices) and arr.shape[2] != len(active_indices)):
            arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    rows, cols, bands = arr.shape

    # 2. Extract valid data mask
    mid_band = bands // 2
    valid_mask = (arr[..., mid_band] != 0.0) & (arr[..., mid_band] != -9999.0) & ~np.isnan(arr[..., mid_band])

    # 3. Channel slicing to match active wavelength bands
    if source_wavelengths is not None and target_wavelengths is not None:
        from pbe_vae.data.spectral import align_spectral_cube

        arr = align_spectral_cube(arr, source_wavelengths, target_wavelengths)
    elif active_indices is not None:
        arr = arr[:, :, active_indices]

    # 4. Numerical cleanup and integer reflectance normalization
    from pbe_vae.data.harvester import normalize_reflectance
    cube = normalize_reflectance(arr, refl_clip)

    return cube, valid_mask


def format_target_spectrum(
    target: Union[np.ndarray, list],
    expected_bands: Optional[int] = None,
) -> np.ndarray:
    """
    Adapts target signature array to 1D float array of shape (bands,).
    """
    tgt = np.asarray(target, dtype=np.float32)
    if tgt.ndim > 1:
        tgt = np.mean(tgt, axis=0)
    tgt = tgt.ravel()

    if expected_bands is not None and len(tgt) != expected_bands:
        raise ValueError(
            f"Target signature dimension ({len(tgt)} bands) does not match expected image bands ({expected_bands} bands)."
        )
    return tgt


def run_ace_detector(
    image: np.ndarray,
    target: np.ndarray,
    active_indices: Optional[np.ndarray] = None,
    nodata_value: float = -9999.0,
) -> np.ndarray:
    """
    Runs the official Spectral Python ACE algorithm on an adapted HSI cube.

    Usage:
        detection_map = run_ace_detector(image_cube, target_signature)
    """
    try:
        from spectral.algorithms.detectors import ACE, ace
    except ImportError as e:
        raise ImportError(
            "Spectral Python (`spectral`) is required. Please install it using: pip install spectral"
        ) from e

    # 1. Format inputs for Spectral Python: (rows, cols, bands)
    cube, valid_mask = format_hsi_cube(image, active_indices=active_indices)
    target_sig = format_target_spectrum(target, expected_bands=cube.shape[2])

    # 2. Run official Spectral Python ACE detector
    detection_map = np.asarray(ace(cube, target_sig), dtype=np.float32)

    # 3. Apply nodata mask to background margins
    detection_map[~valid_mask] = nodata_value

    return detection_map


def run_cem_detector(
    image: np.ndarray,
    target: np.ndarray,
    active_indices: Optional[np.ndarray] = None,
    nodata_value: float = -9999.0,
) -> np.ndarray:
    """
    Runs the official Spectral Python CEM algorithm on an adapted HSI cube.
    """
    if algo is None:
        raise ImportError(
            "Spectral Python (`spectral`) is required. Please install it using: pip install spectral"
        )

    cube, valid_mask = format_hsi_cube(image, active_indices=active_indices)
    target_sig = format_target_spectrum(target, expected_bands=cube.shape[2])

    cem_detector = algo.CEM(target=target_sig, background=cube)
    detection_map = cem_detector(cube).astype(np.float32)
    detection_map[~valid_mask] = nodata_value

    return detection_map


ACEDetector = ACE
CEMDetector = CEM
SAMDetector = SAM

__all__ = [
    "algo",
    "ACE",
    "CEM",
    "MatchedFilter",
    "SAM",
    "ACEDetector",
    "CEMDetector",
    "SAMDetector",
    "format_hsi_cube",
    "format_target_spectrum",
    "run_ace_detector",
    "run_cem_detector",
]
