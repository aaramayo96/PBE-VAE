"""Build only the probability mosaic from existing *_PROB.tif tiles.

This script intentionally skips harvest, training, classifier fitting,
prediction, and candidate export. It reuses the existing mosaic utilities.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pbe_vae_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/pbe_vae_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from pbe_vae.data.mosaic import assemble_probability_mosaic
from pbe_vae.utils.config import load_config
from pbe_vae.utils.qgis_renderer import export_annotated_map_pdf


DEFAULT_CONFIG = REPO_ROOT / "configs" / "aml_standard_detected_roi.yaml"


def _config_value(cfg: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    value = cfg.get(section, {}).get(key, default)
    return default if value is None else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a probability mosaic from already-created *_PROB.tif files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Config file used for default output path and CRS.",
    )
    parser.add_argument(
        "--prob-dir",
        type=Path,
        default=None,
        help="Directory containing *_PROB.tif files. Defaults to paths.out_dir from config.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for final mosaic output. Defaults to <prob-dir>.",
    )
    parser.add_argument(
        "--target-epsg",
        default=None,
        help="Target CRS, for example EPSG:32611. Defaults to processing.target_crs from config.",
    )
    parser.add_argument(
        "--target-resolution",
        default=None,
        help="Output resolution in CRS units, or native/original/auto. Defaults to config value or 100.",
    )
    parser.add_argument(
        "--pattern",
        default="*_PROB.tif",
        help="Input raster glob pattern.",
    )
    parser.add_argument(
        "--prefix",
        default="prob_mosaic_100m",
        help="Output basename prefix.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip the PDF map preview.",
    )
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Render the PDF from an existing final GeoTIFF without rebuilding the mosaic.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep warped tiles, VRT, and unmasked mosaic in the output directory.",
    )
    return parser.parse_args()


def main() -> dict[str, Path]:
    args = parse_args()
    cfg = load_config(args.config)

    prob_dir = args.prob_dir or Path(cfg["paths"]["out_dir"])
    out_dir = args.out_dir or prob_dir
    target_epsg = args.target_epsg or _config_value(
        cfg,
        "processing",
        "target_crs",
        "EPSG:32611",
    )
    target_resolution = args.target_resolution
    if target_resolution is None:
        target_resolution = _config_value(cfg, "processing", "target_resolution", 100.0)
    final_tif = out_dir / f"{args.prefix}_masked.tif"
    pdf_path = out_dir / f"{args.prefix}_map.pdf"

    if args.pdf_only:
        if not final_tif.exists():
            raise FileNotFoundError(f"Cannot render PDF; missing final mosaic: {final_tif}")
        if export_annotated_map_pdf(mosaic_tif=final_tif, out_pdf=pdf_path):
            print(f"[+] Existing masked GeoTIFF: {final_tif}")
            print(f"[+] PDF map: {pdf_path}")
            print("[+] PDF-only render completed.")
            return {"masked_tif": final_tif, "pdf_map": pdf_path}
        raise RuntimeError(f"PDF map rendering failed for {final_tif}")

    prob_tifs = sorted(
        p for p in prob_dir.glob(args.pattern)
        if p.is_file() and not p.name.startswith("._")
    )
    if not prob_tifs:
        raise FileNotFoundError(f"No {args.pattern} files found in {prob_dir}")

    print(f"[+] Probability tiles: {len(prob_tifs)}")
    print(f"[+] Source: {prob_dir}")
    print(f"[+] Final mosaic output: {out_dir}")
    print(f"[+] Target CRS: {target_epsg}")
    print(f"[+] Target resolution: {target_resolution}")

    temp_context = None
    work_dir = out_dir
    if not args.keep_intermediates:
        temp_context = tempfile.TemporaryDirectory(prefix="pbe_vae_mosaic_")
        work_dir = Path(temp_context.name)

    try:
        result = assemble_probability_mosaic(
            prob_dir=prob_dir,
            out_dir=work_dir,
            pattern=args.pattern,
            prefix=args.prefix,
            target_epsg=target_epsg,
            target_resolution=target_resolution,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        if args.keep_intermediates:
            final_tif = result["masked_tif"]
        else:
            shutil.copy2(result["masked_tif"], final_tif)
            result = {"masked_tif": final_tif, "n_tiles": result["n_tiles"]}

        print(f"[+] Warped tiles: {result['n_tiles']}")
        print(f"[+] Final masked GeoTIFF: {final_tif}")

        if not args.no_pdf:
            if export_annotated_map_pdf(mosaic_tif=final_tif, out_pdf=pdf_path):
                result["pdf_map"] = pdf_path
                print(f"[+] PDF map: {pdf_path}")
            else:
                print("[WARN] PDF map rendering failed; GeoTIFF mosaic was still created.")
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    print("[+] Mosaic-only build completed.")
    return result


if __name__ == "__main__":
    main()
