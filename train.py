#!/usr/bin/env python3
"""
Top-level training entrypoint script for PBE-VAE.

Usage:
    # Python API:
    from pbe_vae import PBEVAE
    model = PBEVAE("configs/default.yaml")
    results = model.train(data="configs/default.yaml", epochs=100, n_clusters=10)

    # CLI:
    python train.py --config configs/default.yaml --epochs 100 --n-clusters 10
"""

import argparse
from pbe_vae import PBEVAE


def main():
    parser = argparse.ArgumentParser(description="PBE-VAE Detector Training Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=100, help="Number of VAE epochs")
    parser.add_argument("--n-clusters", type=int, default=10, help="Number of GMM clusters")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Load model from YAML
    model = PBEVAE(model=args.config)

    # Train the model wrapped in a single function (Harvest -> VAE Cluster -> GBDT Fit)
    results = model.train(
        data=args.config,
        epochs=args.epochs,
        n_clusters=args.n_clusters,
        out_dir=args.out_dir
    )

    print("\n✅ Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
