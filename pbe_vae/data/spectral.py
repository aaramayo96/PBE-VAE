"""Header-driven spectral-band selection and cross-flightline alignment."""

from __future__ import annotations

from typing import Any, Iterable, Tuple

import numpy as np


def _ranges(value: Any, default: Iterable[Tuple[float, float]]) -> list[Tuple[float, float]]:
    """Convert YAML wavelength ranges into validated micrometer intervals."""
    ranges = value if value is not None else default
    if len(ranges) == 2 and np.isscalar(ranges[0]) and np.isscalar(ranges[1]):
        ranges = [ranges]
    parsed = []
    for interval in ranges:
        if len(interval) != 2:
            raise ValueError("Each wavelength interval must contain exactly two values.")
        low, high = sorted((float(interval[0]), float(interval[1])))
        parsed.append((low, high))
    return parsed


def selected_band_indices(wavelengths: np.ndarray, processing: dict[str, Any]) -> np.ndarray:
    """Select model bands and exclude configured atmospheric absorption windows."""
    wvl = np.asarray(wavelengths, dtype=np.float32).ravel()
    if not len(wvl):
        raise ValueError("Flightline header does not contain wavelength metadata.")

    vnir = _ranges(processing.get("vnir_range"), [(0.38, 1.20)])[0]
    swir = _ranges(processing.get("swir_range"), [(2.00, 2.50)])[0]
    excluded = _ranges(
        processing.get("excluded_wavelength_ranges"),
        [(1.30, 1.50), (1.75, 2.00)],
    )

    keep = (
        ((wvl >= vnir[0]) & (wvl <= vnir[1]))
        | ((wvl >= swir[0]) & (wvl <= swir[1]))
    )
    for low, high in excluded:
        keep &= ~((wvl >= low) & (wvl <= high))

    indices = np.flatnonzero(keep & np.isfinite(wvl))
    if len(indices) < 2:
        raise ValueError(
            "Band selection left fewer than two channels. Check wavelength units and processing ranges."
        )
    return indices.astype(np.int32)


def align_spectral_cube(
    cube: np.ndarray,
    source_wavelengths: np.ndarray,
    target_wavelengths: np.ndarray,
) -> np.ndarray:
    """Linearly align an ``(rows, cols, bands)`` cube to model wavelengths."""
    data = np.asarray(cube, dtype=np.float32)
    source = np.asarray(source_wavelengths, dtype=np.float32).ravel()
    target = np.asarray(target_wavelengths, dtype=np.float32).ravel()
    if data.ndim != 3 or data.shape[-1] != len(source):
        raise ValueError(
            f"Spectral cube shape {data.shape} does not match {len(source)} source wavelengths."
        )
    if not len(target):
        raise ValueError("No model wavelengths were supplied for spectral alignment.")

    order = np.argsort(source)
    source = source[order]
    data = data[..., order]
    source, unique_idx = np.unique(source, return_index=True)
    data = data[..., unique_idx]
    if target[0] < source[0] or target[-1] > source[-1]:
        raise ValueError(
            "Flightline wavelength coverage does not span the trained model wavelengths: "
            f"source={source[0]:.4f}-{source[-1]:.4f} um, "
            f"model={target[0]:.4f}-{target[-1]:.4f} um."
        )
    if len(source) == len(target) and np.allclose(source, target, rtol=0.0, atol=1e-5):
        return data

    upper = np.searchsorted(source, target, side="left")
    upper = np.clip(upper, 1, len(source) - 1)
    lower = upper - 1
    span = source[upper] - source[lower]
    weight = (target - source[lower]) / np.maximum(span, 1e-12)
    aligned = data[..., lower] * (1.0 - weight) + data[..., upper] * weight
    return aligned.astype(np.float32, copy=False)


__all__ = ["align_spectral_cube", "selected_band_indices"]
