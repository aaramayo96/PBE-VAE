#!/usr/bin/env python3
"""
Top-level inference entrypoint script for PBE-VAE.

Usage:
    # Python API:
    from pbe_vae import PBEVAE
    model = PBEVAE("trained_model.pkl")
    detections = model.predict(conf=0.85)

    # CLI:
    python predict.py --weights trained_model.pkl --conf 0.85
"""

import argparse
from pbe_vae import PBEVAE


def main():
    parser = argparse.ArgumentParser(description="PBE-VAE Detector Inference Runner")
    parser.add_argument("--weights", type=str, default="trained_model.pkl", help="Path to trained model bundle")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--source", type=str, default=None, help="Directory of AVIRIS flightlines")
    parser.add_argument("--conf", type=float, default=None, help="Probability threshold (0.0 to 1.0)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Load model from pretrained weights
    model = PBEVAE(model=args.weights)

    print("=== Running PBE-VAE Target Detection Inference ===")
    detections = model.predict(source=args.source, conf=args.conf, out_dir=args.out_dir)

    print(f"\n✅ Inference completed successfully! Detected {len(detections)} candidate sites.")


if __name__ == "__main__":
    main()
