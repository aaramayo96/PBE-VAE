"""
pbe_vae.cli.main
================
Command-Line Interface entrypoint for PBE-VAE target detector framework.
"""

import argparse
import sys
from pathlib import Path
from pbe_vae.detector import PBEVAE


def main():
    parser = argparse.ArgumentParser(
        prog="pbe-vae",
        description="PBE-VAE: Probability Based Expert Variational Autoencoder Detector CLI"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML file")
    
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # 1. harvest command
    harvest_parser = subparsers.add_parser("harvest", help="Harvest ROI cubes and build dataset")
    harvest_parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    harvest_parser.add_argument("--roi-radius-pixels", type=int, default=None, help="Input ROI radius in pixels")
    harvest_parser.add_argument("--roi-radius-meters", type=float, default=None, help="Input ROI radius in meters")
    
    # 2. cluster command
    cluster_parser = subparsers.add_parser("cluster", help="Train PGE-VAE and latent space clustering models")
    cluster_parser.add_argument("--n-clusters", type=int, default=10, help="Number of clusters")
    cluster_parser.add_argument("--out-dir", type=str, default=None, help="Output directory")

    # 3. train command
    train_parser = subparsers.add_parser("train", help="Train GBDT target detection model with spatial LOGO-CV")
    train_parser.add_argument("--out-dir", type=str, default=None, help="Output directory")

    # 4. predict command
    predict_parser = subparsers.add_parser("predict", help="Run pixel-level inference across flight lines")
    predict_parser.add_argument("--weights", type=str, default=None, help="Path to trained_model.pkl")
    predict_parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    predict_parser.add_argument(
        "--detected-roi-size-pixels",
        type=int,
        nargs="+",
        default=None,
        metavar="PX",
        help="Standard detected ROI proposal size in pixels; pass one value for square or two for H W",
    )
    predict_parser.add_argument("--roi-window-max-multiple", type=int, default=None, help="Maximum multiple of the detected ROI proposal size")
    predict_parser.add_argument("--min-roi-coverage", type=float, default=None, help="Minimum detected-pixel coverage inside the proposal ROI")

    # 5. mosaic command
    mosaic_parser = subparsers.add_parser("mosaic", help="Reproject and mosaic probability GeoTIFFs")
    mosaic_parser.add_argument("--prob-dir", type=str, default=None, help="Directory containing *_PROB.tif files")
    mosaic_parser.add_argument("--out-dir", type=str, default=None, help="Output mosaic directory")
    mosaic_parser.add_argument("--shp", type=str, default=None, help="Path to flight line shapefile")

    # 6. annotate command
    annot_parser = subparsers.add_parser("annotate", help="Burn cartographic markers and export annotated PDF")
    annot_parser.add_argument("--mosaic-tif", type=str, default=None, help="Path to mosaic GeoTIFF")
    annot_parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    annot_parser.add_argument("--top-n", type=int, default=20, help="Top N new ROIs to burn")
    annot_parser.add_argument("--shp", type=str, default=None, help="Path to flight line shapefile")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    detector = PBEVAE(model=args.config, weights=getattr(args, 'weights', None))

    if args.command == "harvest":
        detector.harvest(
            output_dir=args.out_dir,
            roi_radius_pixels=args.roi_radius_pixels,
            roi_radius_meters=args.roi_radius_meters,
        )
    elif args.command == "cluster":
        detector.cluster(n_clusters=args.n_clusters, output_dir=args.out_dir)
    elif args.command == "train":
        detector.train(out_dir=args.out_dir)
    elif args.command == "predict":
        detected_roi_size = args.detected_roi_size_pixels
        if detected_roi_size is not None and len(detected_roi_size) == 1:
            detected_roi_size = detected_roi_size[0]
        detector.predict(
            out_dir=args.out_dir,
            detected_roi_size_pixels=detected_roi_size,
            roi_window_max_multiple=args.roi_window_max_multiple,
            min_roi_coverage=args.min_roi_coverage,
        )
    elif args.command == "mosaic":
        detector.mosaic(source=args.prob_dir, out_dir=args.out_dir, shp_path=args.shp)
    elif args.command == "annotate":
        detector.annotate(mosaic=args.mosaic_tif, out_dir=args.out_dir, top_n=args.top_n, shp_path=args.shp)


if __name__ == "__main__":
    main()
