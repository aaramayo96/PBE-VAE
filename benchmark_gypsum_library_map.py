"""Map the averaged gypsum spectrum with ACE and SAM.

This benchmark demonstrates a direct spectral-library use of ``pbe_vae``:
load the bundled gypsum reference spectra, average them, apply the configured
AVIRIS band selection, then map that target with ACE and SAM.  It does not run
the VAE, classifier training, or the PBE-VAE probability detector.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / "configs" / "default.yaml"
LIBRARY_DIR = SCRIPT_DIR / "library"
OUTPUT_DIR = SCRIPT_DIR / "benchmark_results" / "gypsum_library_map"

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pbe_vae_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/pbe_vae_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from pbe_vae import PBEVAE
from pbe_vae.data.geo import find_aviris_files
from pbe_vae.data.spectral import selected_band_indices

# Set SOURCE_DIRS to a list of AVIRIS reflectance folders to override
# auto-discovery. Leave as None to prefer the mounted consolidated Arizona set,
# then fall back to configs/default.yaml paths.base_dir.
SOURCE_DIRS = None
SOURCE_DIR_CANDIDATES = [
    "/Volumes/HSC/ARIZONA_FULL/flightlines",
    "/Volumes/HSC/ADEQ",
]

GYPSUM_SPECTRUM_PATTERN = "s07_AV14_Gypsum_HS333.*B_(Selenite)_ASDFRa_AREF.txt"
WAVELENGTH_FILE = LIBRARY_DIR / "s07_AV14_Wavelengths_in_microns_224_ch_AVIRIS14a.txt"

ACE_CONFIDENCE = 0.02
SAM_CONFIDENCE = 0.06
METHODS = ("ace", "sam")
NUM_WORKERS = int(os.environ.get("GYPSUM_WORKERS", "1"))
SKIP_EXISTING = True


def resolve_source_dirs(model: PBEVAE) -> list[str]:
    """Choose source folders with discoverable AVIRIS flightlines."""
    if SOURCE_DIRS is None:
        candidates = []
        for path in SOURCE_DIR_CANDIDATES:
            if Path(path).exists() and find_aviris_files([path]):
                candidates = [path]
                break
        if not candidates:
            configured = model.cfg.get("paths", {}).get("base_dir", [])
            candidates = configured if isinstance(configured, list) else [configured]
    elif isinstance(SOURCE_DIRS, (str, Path)):
        candidates = [str(SOURCE_DIRS)]
    else:
        candidates = list(SOURCE_DIRS)

    files = find_aviris_files(candidates)
    if not files:
        raise FileNotFoundError(
            "No AVIRIS flightlines found for gypsum ACE/SAM mapping. "
            f"Checked source dirs: {candidates}"
        )

    print(f"[+] Source dirs: {candidates}")
    print(f"[+] Found {len(files)} AVIRIS flightlines")
    return [str(path) for path in candidates]


def load_usgs_ascii_spectrum(path: Path) -> np.ndarray:
    """Load a USGS-style ASCII spectrum with a one-line text header."""
    values = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                values.append(float(stripped))
            except ValueError:
                continue

    if not values:
        raise ValueError(f"No numeric spectral values found in {path}")
    return np.asarray(values, dtype=np.float32)


def average_spectra(paths: Iterable[Path]) -> np.ndarray:
    """Load same-band spectra and return their per-band average."""
    spectra = [load_usgs_ascii_spectrum(path) for path in paths]
    if not spectra:
        raise FileNotFoundError(
            f"No gypsum spectra matched {GYPSUM_SPECTRUM_PATTERN} in {LIBRARY_DIR}"
        )

    band_counts = {spectrum.shape[0] for spectrum in spectra}
    if len(band_counts) != 1:
        raise ValueError(f"Gypsum spectra have inconsistent band counts: {sorted(band_counts)}")

    return np.mean(np.vstack(spectra), axis=0).astype(np.float32)


def build_gypsum_target(model: PBEVAE, output_dir: Path) -> Path:
    """Average the bundled gypsum spectra and save the selected-band target."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gypsum_files = sorted(LIBRARY_DIR.glob(GYPSUM_SPECTRUM_PATTERN))
    target_signature = average_spectra(gypsum_files)
    wavelengths = load_usgs_ascii_spectrum(WAVELENGTH_FILE)

    if wavelengths.shape[0] != target_signature.shape[0]:
        raise ValueError(
            "Gypsum target and wavelength table have different band counts: "
            f"{target_signature.shape[0]} vs {wavelengths.shape[0]}"
        )

    fused_idxs = selected_band_indices(wavelengths, model.cfg.get("processing", {}))
    active_wavelengths = wavelengths[fused_idxs]
    target_selected = target_signature[fused_idxs]

    full_target_path = output_dir / "gypsum_average_full_224_band_signature.npy"
    selected_target_path = output_dir / "gypsum_average_selected_band_signature.npy"
    np.save(full_target_path, target_signature)
    np.save(selected_target_path, target_selected)
    np.save(output_dir / "gypsum_aviris14a_wavelengths_microns.npy", wavelengths)
    np.save(output_dir / "gypsum_selected_wavelengths_microns.npy", active_wavelengths)

    # ACE/SAM use this lightweight metadata to align flightlines to the same
    # selected wavelength grid as the gypsum target. No VAE training data is used.
    np.savez(
        output_dir / "harvested_data.npz",
        X=target_selected[np.newaxis, :],
        ids=np.array([1], dtype=np.int32),
        idxs=fused_idxs,
        active_wavelengths=active_wavelengths,
        has_dem=False,
        source="bundled_gypsum_library_average",
        source_spectra=np.array([path.name for path in gypsum_files]),
    )

    print(f"[+] Averaged {len(gypsum_files)} gypsum spectra")
    print(f"[+] Selected bands: {len(fused_idxs)} of {target_signature.shape[0]}")
    print(f"[+] Target signature: {selected_target_path}")
    return selected_target_path


def output_path_for_method(method: str, key: str) -> Path:
    """Return the expected detector GeoTIFF path for a flightline key."""
    method_l = method.lower()
    if method_l == "ace":
        return OUTPUT_DIR / "ace_detections" / f"{key}_ACE.tif"
    if method_l == "sam":
        return OUTPUT_DIR / "sam_detections" / f"{key}_SAM.tif"
    raise ValueError(f"Unsupported detector method: {method}")


def chunk_items(items: list[str], n_chunks: int) -> list[list[str]]:
    """Round-robin items into balanced non-empty chunks."""
    chunks = [[] for _ in range(max(1, n_chunks))]
    for i, item in enumerate(items):
        chunks[i % len(chunks)].append(item)
    return [chunk for chunk in chunks if chunk]


def run_detector_worker(method: str, source_dirs: list[str], target_path: str) -> list[dict]:
    """Run one ACE or SAM worker over a subset of flightline folders."""
    worker_model = PBEVAE(CONFIG_FILE)
    worker_model.cfg["paths"]["out_dir"] = str(OUTPUT_DIR)
    worker_model.cfg["paths"]["base_dir"] = source_dirs
    return worker_model.run_single_spectrum(
        target_spectrum=target_path,
        source=source_dirs,
        methods=(method,),
        conf={"ace": ACE_CONFIDENCE, "sam": SAM_CONFIDENCE},
        output_dir=OUTPUT_DIR,
        save=True,
    ).get(method, [])


def run_method_parallel(method: str, files: list[tuple], target_path: Path) -> list[dict]:
    """Run one detector with optional skip/resume and worker parallelism."""
    pending_dirs = []
    skipped = 0
    for key, _refl_dat, refl_hdr in files:
        if SKIP_EXISTING and output_path_for_method(method, key).exists():
            skipped += 1
            continue
        pending_dirs.append(str(Path(refl_hdr).parent))

    print(
        f"[+] {method.upper()}: {len(pending_dirs)} pending, "
        f"{skipped} skipped existing, workers={NUM_WORKERS}"
    )
    if not pending_dirs:
        return []

    if NUM_WORKERS <= 1:
        return run_detector_worker(method, pending_dirs, str(target_path))

    results = []
    chunks = chunk_items(pending_dirs, NUM_WORKERS)
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(run_detector_worker, method, chunk, str(target_path))
            for chunk in chunks
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = PBEVAE(CONFIG_FILE)
    model.cfg["paths"]["out_dir"] = str(OUTPUT_DIR)
    source_dirs = resolve_source_dirs(model)
    model.cfg["paths"]["base_dir"] = source_dirs
    files = find_aviris_files(source_dirs)

    target_path = build_gypsum_target(model, OUTPUT_DIR)

    results = {
        method: run_method_parallel(method, files, target_path)
        for method in METHODS
    }

    print("\n[+] Gypsum ACE/SAM spectral-library mapping complete")
    print(f"[+] Outputs: {OUTPUT_DIR}")
    for method, rows in results.items():
        detected_pixels = sum(row.get("n_detected_pixels", 0) for row in rows)
        print(f"[+] {method.upper()}: {len(rows)} flightlines, {detected_pixels} detected pixels")

    return results


if __name__ == "__main__":
    main()
