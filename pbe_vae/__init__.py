"""
PBE-VAE: Probability Based Expert Variational Autoencoder Package
=================================================================
A modular Python detector framework for hyperspectral environmental target detection.
"""

__version__ = "1.0.0"

from pbe_vae.detector import PBEVAE
from pbe_vae.models.baselines import ACEDetector, CEMDetector, SAMDetector

__all__ = ["PBEVAE", "ACEDetector", "CEMDetector", "SAMDetector", "__version__"]
