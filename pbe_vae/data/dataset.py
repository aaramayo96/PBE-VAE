"""
pbe_vae.data.dataset
====================
Dataset definitions and background sampling routines for hyperspectral target detection.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional
from osgeo import gdal
from tqdm import tqdm

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, TensorDataset
except ImportError:
    torch = None
    class _DummyDataset:
        pass
    Dataset = _DummyDataset
    DataLoader = None
    TensorDataset = None

_REFL_CLIP = (0.0, 1.5)


class HyperspectralDataset(Dataset):
    """
    PyTorch Dataset wrapper for hyperspectral pixel tensors.
    """
    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None, transform=None):
        if torch is not None:
            self.data = torch.from_numpy(data.astype(np.float32))
            self.labels = torch.from_numpy(labels.astype(np.int64)) if labels is not None else None
        else:
            self.data = data.astype(np.float32)
            self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        if self.labels is not None:
            return sample, self.labels[idx]
        return sample


class SpatialFlightlineSampler:
    """
    Sampler utility for spatially contiguous chunks across flightlines.
    """
    def __init__(self, n_samples: int, chunk_size: int = 512):
        self.n_samples = n_samples
        self.chunk_size = chunk_size


def harvest_background_data(
    files: List[Tuple[str, Path, Path]], 
    cfg: Dict[str, Any], 
    fused_idxs: np.ndarray,
    active_wavelengths: Optional[np.ndarray],
    vae_model, vae_X_mean, vae_X_std,
    cluster_bundle, target_size: int, device: str = 'cpu',
    use_dem: Optional[bool] = None,
    dem_weight: Optional[float] = None
) -> np.ndarray:
    """
    Randomly sample flight lines for unlabeled background pixels.
    Uses the SAME VAE -> cluster pipeline so features are consistent with targets.
    """
    from pbe_vae.models.classifier import make_features
    
    if use_dem is None:
        use_dem = cfg.get("processing", {}).get("use_dem", True)
    if dem_weight is None:
        dem_weight = cfg.get("processing", {}).get("dem_weight", cfg.get("processing", {}).get("dem_weight_ratio", 1.0))
    
    dem_path = cfg.get("paths", {}).get("dem_path", "")
    has_dem = bool(use_dem and dem_path and Path(dem_path).exists())

    print(f"\n[+] Sampling Background pixels (Target count: {target_size:,}) | Use DEM: {has_dem}...")
    X_bg = []
    processing = cfg.get("processing", {})
    background_ratio = float(processing.get("background_sample_ratio", 1.0))
    background_max = int(processing.get("background_max_pixels", 50_000))
    max_flightlines = int(processing.get("background_max_flightlines", 0))
    if max_flightlines > 0 and len(files) > max_flightlines:
        selected = np.linspace(0, len(files) - 1, max_flightlines, dtype=int)
        files = [files[index] for index in np.unique(selected)]

    target_total = max(1, min(int(target_size * background_ratio), background_max))
    samples_per = max(1, int(np.ceil(target_total / max(len(files), 1))))

    from pbe_vae.data.geo import get_wavelengths, open_georeferenced_dataset
    from pbe_vae.data.spectral import align_spectral_cube

    for key, refl_dat, refl_hdr in tqdm(files, desc="Background sampling"):
        try:
            ds_hsi = open_georeferenced_dataset(refl_dat, refl_hdr)
            if ds_hsi is None:
                continue
            cols, rows = ds_hsi.RasterXSize, ds_hsi.RasterYSize

            x_off, y_off = cols // 4, rows // 4
            x_sz, y_sz = cols // 2, min(2000, rows // 2)

            h_chunk = ds_hsi.ReadAsArray(x_off, y_off, x_sz, y_sz).astype(np.float32).transpose(1, 2, 0)
            gt = ds_hsi.GetGeoTransform()

            if has_dem:
                ds_dem = gdal.Warp(
                    f"/vsimem/bg_{key}.tif", dem_path,
                    options=gdal.WarpOptions(
                        format="GTiff", dstSRS=ds_hsi.GetProjection(),
                        width=x_sz, height=y_sz,
                        outputBounds=(
                            gt[0] + x_off * gt[1],
                            gt[3] + (y_off + y_sz) * gt[5],
                            gt[0] + (x_off + x_sz) * gt[1],
                            gt[3] + y_off * gt[5]
                        )
                    )
                )
                d_chunk = ds_dem.ReadAsArray() if ds_dem else None
                if d_chunk is not None and d_chunk.ndim == 3:
                    d_chunk = d_chunk[0]
            else:
                d_chunk = None

            from pbe_vae.data.geo import roi_centres_in_image
            known_rois = roi_centres_in_image(cfg, ds_hsi)

            mid = h_chunk.shape[2] // 2
            mask = (h_chunk[..., mid] != 0.0) & (h_chunk[..., mid] != -9999.0)

            # Strict exclusion of known target ROIs from background
            r_excl = int(cfg.get("discovery", {}).get("dedup_radius_pixels", 80))
            for _, r_col, r_row in known_rois:
                c_c = r_col - x_off
                c_r = r_row - y_off
                if -r_excl <= c_c < x_sz + r_excl and -r_excl <= c_r < y_sz + r_excl:
                    rr0, rr1 = max(0, c_r - r_excl), min(y_sz, c_r + r_excl + 1)
                    cc0, cc1 = max(0, c_c - r_excl), min(x_sz, c_c + r_excl + 1)
                    mask[rr0:rr1, cc0:cc1] = False

            if np.any(mask):
                from pbe_vae.data.harvester import normalize_reflectance

                if active_wavelengths is None:
                    h_sub = normalize_reflectance(h_chunk[:, :, fused_idxs], _REFL_CLIP)
                else:
                    source_wavelengths = get_wavelengths(refl_hdr)
                    h_sub = normalize_reflectance(
                        align_spectral_cube(h_chunk, source_wavelengths, active_wavelengths),
                        _REFL_CLIP,
                    )

                if has_dem and d_chunk is not None:
                    d_sub = np.nan_to_num(d_chunk)[..., np.newaxis]
                    raw_flat = np.concatenate(
                        [h_sub.reshape(-1, h_sub.shape[2]), d_sub.reshape(-1, 1)], axis=1
                    )
                else:
                    raw_flat = h_sub.reshape(-1, h_sub.shape[2])

                feats = make_features(
                    raw_pixels=raw_flat,
                    vae_model=vae_model,
                    vae_X_mean=vae_X_mean,
                    vae_X_std=vae_X_std,
                    cluster_bundle=cluster_bundle,
                    device=device,
                    use_dem=has_dem,
                    dem_weight=dem_weight,
                )
                valid = feats[mask.ravel()]
                np.random.shuffle(valid)
                X_bg.append(valid[:samples_per])

            if has_dem:
                gdal.Unlink(f"/vsimem/bg_{key}.tif")
        except Exception:
            continue

    if not X_bg:
        raise ValueError("Could not harvest any background data from flight lines.")
    out = np.vstack(X_bg)
    if len(out) > target_total:
        keep = np.random.choice(len(out), size=target_total, replace=False)
        out = out[keep]
    print(f"  -> {len(out):,} background pixels harvested.")
    return out
