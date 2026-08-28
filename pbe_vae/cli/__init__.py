import argparse
import sys
from pathlib import Path
from pbe_vae.detector import PBEVAE

def main():
    parser = argparse.ArgumentParser(
        prog="pbe-vae",
        description="PBE-VAE: Physics-Guided Embedding Variational Autoencoder Detector CLI"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML file")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # harvest command
    harvest_parser = subparsers.add_parser("harvest", help="Harvest ROI cubes and build dataset")
    harvest_parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    
    # cluster command
    cluster_parser = subparsers.add_parser("cluster", help="Train PGE-VAE and latent space clustering models")
    cluster_parser.add_argument("--n-clusters", type=int, default=10, help="Number of GMM clusters")
    cluster_parser.add_argument("--out-dir", type=str, default=None, help="Output directory")

    # train command
    train_parser = subparsers.add_parser("train", help="Train GBDT target detection model with spatial CV")
    train_parser.add_argument("--out-dir", type=str, default=None, help="Output directory")

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Run pixel-level inference on flight lines")
    predict_parser.add_argument("--weights", type=str, default=None, help="Path to trained_model.pkl")
    predict_parser.add_argument("--out-dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    detector = PBEVAE(config_path=args.config, weights=getattr(args, 'weights', None))

    if args.command == "harvest":
        detector.harvest(output_dir=args.out_dir)
    elif args.command == "cluster":
        detector.cluster(n_clusters=args.n_clusters, output_dir=args.out_dir)
    elif args.command == "train":
        detector.train(output_dir=args.out_dir)
    elif args.command == "predict":
        detector.predict(output_dir=args.out_dir)

if __name__ == "__main__":
    main()
