"""
pbe_vae.models
==============
Physics-guided neural network models (PGE-VAE) and GBDT classifier wrappers.
"""

from pbe_vae.models.vae import (
    ExpertGate,
    LightweightVAE,
    train_and_extract_vae,
    apply_vae,
)
from pbe_vae.models.classifier import (
    PBEVAEClassifier,
    cluster_soft_features,
    make_features,
)

__all__ = [
    "ExpertGate",
    "LightweightVAE",
    "train_and_extract_vae",
    "apply_vae",
    "PBEVAEClassifier",
    "cluster_soft_features",
    "make_features",
]
