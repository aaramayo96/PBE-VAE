"""
pbe_vae.models.vae
==================
Variational Autoencoder architectures with Physics-Guided Expert Prior Gates.
Supports dynamic layer-by-layer YAML model definitions in [from, number, module, args] format.
"""

from __future__ import annotations
import copy
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Type, Union, List

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    class _DummyTensor:
        def float(self):
            return self
        def __mul__(self, other):
            return self
        def __add__(self, other):
            return self

    class _DummyModule:
        def __init__(self, *args, **kwargs):
            pass
        def to(self, *args, **kwargs):
            return self
        def eval(self):
            return self
        def train(self):
            return self
        def state_dict(self):
            return {}
        def load_state_dict(self, *args, **kwargs):
            pass

    torch = type(
        'torch',
        (),
        {
            'from_numpy': lambda *args, **kwargs: _DummyTensor(),
            'Tensor': lambda *args, **kwargs: _DummyTensor(),
            'rand_like': lambda *args, **kwargs: _DummyTensor(),
            'sigmoid': lambda *args, **kwargs: _DummyTensor(),
            'where': lambda *args, **kwargs: _DummyTensor(),
            'log': lambda *args, **kwargs: _DummyTensor(),
            'exp': lambda *args, **kwargs: _DummyTensor(),
            'ones': lambda *args, **kwargs: _DummyTensor(),
            'randn_like': lambda *args, **kwargs: _DummyTensor(),
            'mean': lambda *args, **kwargs: 0.0,
            'save': lambda *args, **kwargs: None,
            'load': lambda *args, **kwargs: {},
            'cuda': type('cuda', (), {'is_available': lambda: False})(),
            'backends': type('backends', (), {'mps': type('mps', (), {'is_available': lambda: False})()})(),
        },
    )()
    nn = type(
        'nn',
        (),
        {
            'Module': _DummyModule,
            'Parameter': lambda *args, **kwargs: _DummyTensor(),
            'Sequential': lambda *args, **kwargs: _DummyModule(),
            'Linear': lambda *args, **kwargs: _DummyModule(),
            'ReLU': lambda *args, **kwargs: _DummyModule(),
            'LeakyReLU': lambda *args, **kwargs: _DummyModule(),
            'GELU': lambda *args, **kwargs: _DummyModule(),
            'ELU': lambda *args, **kwargs: _DummyModule(),
            'Tanh': lambda *args, **kwargs: _DummyModule(),
            'Sigmoid': lambda *args, **kwargs: _DummyModule(),
            'BatchNorm1d': lambda *args, **kwargs: _DummyModule(),
            'LayerNorm': lambda *args, **kwargs: _DummyModule(),
            'Dropout': lambda *args, **kwargs: _DummyModule(),
            'MSELoss': lambda *args, **kwargs: _DummyModule(),
            'BCEWithLogitsLoss': lambda *args, **kwargs: _DummyModule(),
        },
    )()
    DataLoader = None
    TensorDataset = None

from pbe_vae.utils.config import load_config


# =============================================================================
# 1. Expert Gate Architecture
# =============================================================================

class ExpertGate(nn.Module):
    """
    Expert Prior Gate:
    Unsupervised gate that dynamically masks and weights spectral bands based
    on 2nd-derivative spectral curvature and variance priors.
    """
    def __init__(
        self,
        n_features: int,
        expert_prior: np.ndarray,
        temp: float = 0.5,
        reduction_ratio: int = 4,
        scale_init: float = 3.0,
        min_gate_threshold: float = 0.05,
    ):
        super().__init__()
        self.temp = temp
        self.min_gate_threshold = min_gate_threshold

        expert_prior = np.clip(expert_prior, 1e-4, 1.0 - 1e-4)
        p_logits = np.log(expert_prior / (1.0 - expert_prior))
        self.static_logits = nn.Parameter(torch.from_numpy(p_logits).float())

        r = max(4, n_features // max(1, int(reduction_ratio)))
        self.context = nn.Sequential(
            nn.Linear(n_features, max(8, n_features // r)),
            nn.LeakyReLU(0.2),
            nn.Linear(max(8, n_features // r), n_features),
            nn.Tanh(),
        )
        self.scale = nn.Parameter(torch.ones(1) * scale_init)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        adj = self.context(x)
        logits = (self.static_logits + adj) * self.scale

        if self.training:
            u = torch.rand_like(logits)
            noise = torch.log(u + 1e-10) - torch.log(1.0 - u + 1e-10)
            gate = torch.sigmoid((logits + noise) / self.temp)
        else:
            gate = torch.sigmoid(logits)
            gate = torch.where(gate < self.min_gate_threshold, torch.zeros_like(gate), gate)

        return x * gate, gate


# Aliases for backwards compatibility
ExpertGateV1 = ExpertGate


def build_gate(
    n_features: int = 100,
    expert_prior: Optional[np.ndarray] = None,
    temp: float = 0.5,
    reduction_ratio: int = 4,
    scale_init: float = 3.0,
    min_gate_threshold: float = 0.05,
    **kwargs,
) -> nn.Module:
    """Factory creating an Expert Gate module."""
    if expert_prior is None:
        expert_prior = np.ones(n_features, dtype=np.float32) * 0.5
    return ExpertGate(
        n_features=n_features,
        expert_prior=expert_prior,
        temp=temp,
        reduction_ratio=reduction_ratio,
        scale_init=scale_init,
        min_gate_threshold=min_gate_threshold,
    )


# =============================================================================
# 2. Dynamic Model Parser & PyTorch VAE Builder
# =============================================================================

MODULE_LOOKUP: Dict[str, Any] = {
    "ExpertGate": ExpertGate,
    "Linear": nn.Linear if torch is not None else None,
    "ReLU": nn.ReLU if torch is not None else None,
    "LeakyReLU": nn.LeakyReLU if torch is not None else None,
    "GELU": nn.GELU if torch is not None else None,
    "ELU": nn.ELU if torch is not None else None,
    "Tanh": nn.Tanh if torch is not None else None,
    "Sigmoid": nn.Sigmoid if torch is not None else None,
    "BatchNorm1d": nn.BatchNorm1d if torch is not None else None,
    "LayerNorm": nn.LayerNorm if torch is not None else None,
    "Dropout": nn.Dropout if torch is not None else None,
}


def build_sequential_block(
    layer_defs: List[List[Any]],
    in_dim: int,
    substitutions: Dict[str, Any],
) -> Tuple[nn.Sequential, int]:
    """
    Parses a list of [from, number, module_name, args] definitions
    and builds an nn.Sequential container.
    """
    layers = []
    current_dim = in_dim

    for layer_def in layer_defs:
        _, repeats, mod_name, raw_args = layer_def
        raw_args = copy.deepcopy(raw_args)

        # Substitute variable names (e.g. latent_dim, input_dim)
        args = []
        for a in raw_args:
            if isinstance(a, str) and a in substitutions:
                args.append(substitutions[a])
            else:
                args.append(a)

        mod_cls = MODULE_LOOKUP.get(mod_name)
        if mod_cls is None:
            # Fallback to getattr from torch.nn
            mod_cls = getattr(nn, mod_name, None)
        if mod_cls is None:
            raise ValueError(f"Unknown module '{mod_name}' in model configuration.")

        for _ in range(repeats):
            if mod_name == "Linear":
                out_dim = int(args[0])
                layers.append(nn.Linear(current_dim, out_dim))
                current_dim = out_dim
            elif mod_name == "BatchNorm1d":
                num_features = int(args[0]) if args else current_dim
                layers.append(nn.BatchNorm1d(num_features))
            elif mod_name == "LayerNorm":
                num_features = int(args[0]) if args else current_dim
                layers.append(nn.LayerNorm(num_features))
            elif mod_name in ["LeakyReLU", "ReLU", "GELU", "ELU", "Tanh", "Sigmoid", "Dropout"]:
                layers.append(mod_cls(*args))
            else:
                layers.append(mod_cls(*args))

    return nn.Sequential(*layers), current_dim


class DynamicPBEVAE(nn.Module):
    """
    Layer-by-layer dynamically parsed PBE-VAE network constructed from YAML configuration.
    Structure:
      1. gate: [from, number, ExpertGate, [temp, reduction, scale, threshold]]
      2. encoder: [[from, number, module, args], ...]
      3. latent: [[from, number, module, args], ...] (fc_mu, fc_logvar)
      4. decoder: [[from, number, module, args], ...]
    """
    def __init__(
        self,
        config: Union[str, Path, Dict[str, Any]],
        input_dim: int,
        expert_prior: np.ndarray,
        latent_dim: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(config, (str, Path)):
            self.yaml_cfg = load_config(config)
        else:
            self.yaml_cfg = config

        self.input_dim = input_dim
        self.latent_dim = int(
            latent_dim
            if latent_dim is not None
            else self.yaml_cfg.get("latent_dim", 5)
        )

        substitutions = {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
        }

        # 1. Gate Parsing
        gate_defs = self.yaml_cfg.get("gate", [])
        if gate_defs:
            g_def = gate_defs[0]
            g_args = g_def[3] if len(g_def) > 3 else [0.5, 4, 3.0, 0.05]
            temp = float(g_args[0]) if len(g_args) > 0 else 0.5
            reduction = int(g_args[1]) if len(g_args) > 1 else 4
            scale_init = float(g_args[2]) if len(g_args) > 2 else 3.0
            min_thresh = float(g_args[3]) if len(g_args) > 3 else 0.05
            self.gate = ExpertGate(
                n_features=input_dim,
                expert_prior=expert_prior,
                temp=temp,
                reduction_ratio=reduction,
                scale_init=scale_init,
                min_gate_threshold=min_thresh,
            )
        else:
            self.gate = ExpertGate(
                n_features=input_dim,
                expert_prior=expert_prior,
            )

        # 2. Encoder Parsing
        enc_defs = self.yaml_cfg.get("encoder", [])
        if not enc_defs:
            # Default encoder if not specified
            enc_defs = [
                [-1, 1, "Linear", [64]],
                [-1, 1, "LeakyReLU", [0.2]],
                [-1, 1, "BatchNorm1d", [64]],
                [-1, 1, "Linear", [32]],
                [-1, 1, "LeakyReLU", [0.2]],
            ]
        self.encoder, enc_out_dim = build_sequential_block(enc_defs, input_dim, substitutions)

        # 3. Latent Layers (fc_mu and fc_logvar)
        self.fc_mu = nn.Linear(enc_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, self.latent_dim)

        # 4. Decoder Parsing
        dec_defs = self.yaml_cfg.get("decoder", [])
        if not dec_defs:
            dec_defs = [
                [-1, 1, "Linear", [32]],
                [-1, 1, "LeakyReLU", [0.2]],
                [-1, 1, "BatchNorm1d", [32]],
                [-1, 1, "Linear", [64]],
                [-1, 1, "LeakyReLU", [0.2]],
                [-1, 1, "Linear", ["input_dim"]],
            ]
        self.decoder, _ = build_sequential_block(dec_defs, self.latent_dim, substitutions)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_gated, gate = self.gate(x)
        h = self.encoder(x_gated)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        rec = self.decoder(z)
        return rec, mu, logvar, gate


# Aliases for backwards compatibility
LightweightVAE = DynamicPBEVAE
LightweightVAEV1 = DynamicPBEVAE


def build_vae(
    config: Optional[Union[str, Path, Dict[str, Any]]] = None,
    input_dim: int = 100,
    expert_prior: Optional[np.ndarray] = None,
    latent_dim: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Factory creating a dynamic VAE module from a YAML configuration file or dict."""
    if config is None:
        config = "configs/models/pbe_vae_standard.yaml"

    if expert_prior is None:
        expert_prior = np.ones(input_dim, dtype=np.float32) * 0.5

    return DynamicPBEVAE(
        config=config,
        input_dim=input_dim,
        expert_prior=expert_prior,
        latent_dim=latent_dim,
    )


# =============================================================================
# 3. Prior Computation & Training Orchestrator
# =============================================================================

def compute_expert_prior(
    X_spectra: np.ndarray,
    active_wvls: np.ndarray,
    has_dem: bool = False,
    curvature_threshold: float = 3.37,
) -> np.ndarray:
    """Computes physics-guided spectral curvature & variance prior vector."""
    X_clean = np.nan_to_num(np.asarray(X_spectra, dtype=np.float64), nan=0.0, posinf=1.5, neginf=0.0)
    wvl_clean = np.asarray(active_wvls, dtype=np.float64)

    vnir_mask = wvl_clean < 1.0
    swir_mask = wvl_clean >= 1.0

    grad_comp = np.nan_to_num(np.nanmean(np.abs(np.gradient(X_clean, axis=1)), axis=0))
    var_comp = np.nan_to_num(np.nanvar(X_clean, axis=0))

    if grad_comp.max() > grad_comp.min():
        grad_comp = (grad_comp - grad_comp.min()) / (grad_comp.max() - grad_comp.min())
    if var_comp.max() > var_comp.min():
        var_comp = (var_comp - var_comp.min()) / (var_comp.max() - var_comp.min())

    curvatures = np.nan_to_num(np.abs(np.gradient(np.gradient(X_clean, axis=1), axis=1)))
    swir_c = float(np.nanmean(curvatures[:, swir_mask])) if np.any(swir_mask) else 0.0
    vnir_c = float(np.nanmean(curvatures[:, vnir_mask])) if np.any(vnir_mask) else 1e-6

    if np.isnan(vnir_c) or vnir_c <= 1e-12:
        vnir_c = 1e-6
    if np.isnan(swir_c):
        swir_c = 0.0

    curvature_ratio = float(swir_c / vnir_c)
    if np.isnan(curvature_ratio) or np.isinf(curvature_ratio):
        curvature_ratio = 1.0

    if curvature_ratio >= curvature_threshold:
        print(f"  -> [Prior] SWIR Dominance detected (Curvature Ratio: {curvature_ratio:.2f} >= {curvature_threshold:.2f})")
        prior = (0.8 * grad_comp) + (0.2 * var_comp)
        if np.any(swir_mask):
            prior[swir_mask] *= 2.0
        if np.any(vnir_mask):
            prior[vnir_mask] *= 0.5
    else:
        print(f"  -> [Prior] VNIR Dominance detected (Curvature Ratio: {curvature_ratio:.2f} < {curvature_threshold:.2f})")
        prior = (0.8 * var_comp) + (0.2 * grad_comp)
        if np.any(vnir_mask):
            prior[vnir_mask] *= 2.0
        if np.any(swir_mask):
            prior[swir_mask] *= 0.5

    if prior.max() == prior.min():
        prior = np.ones_like(prior) * 0.5
    else:
        prior = (prior - prior.min()) / (prior.max() - prior.min() + 1e-12)

    prior = np.nan_to_num(prior, nan=0.5)
    prior = np.clip(prior, 0.15, 0.95)

    if has_dem:
        prior = np.append(prior, 0.95)

    return prior.astype(np.float32)


def train_and_extract_vae(
    X_np: np.ndarray,
    active_wvls: Optional[np.ndarray] = None,
    latent_dim: int = 5,
    epochs: int = 100,
    batch_size: int = 4096,
    device: str = "cpu",
    model_cfg: Optional[Union[str, Path, Dict[str, Any]]] = None,
    **kwargs,
) -> Tuple[np.ndarray, nn.Module, np.ndarray, np.ndarray]:
    """
    Initializes, computes physics-guided prior, and trains VAE to compress spectra down to latent space.
    Fully configurable from layer-by-layer YAML definitions.
    """
    cfg: Dict[str, Any] = {}
    if model_cfg is not None:
        if isinstance(model_cfg, (str, Path)):
            cfg = load_config(model_cfg)
        elif isinstance(model_cfg, dict):
            cfg = model_cfg
    else:
        cfg = load_config("configs/models/pbe_vae_standard.yaml")

    l_dim = int(cfg.get("latent_dim", latent_dim))
    rec_weight = float(cfg.get("reconstruction_loss_weight", 10.0))
    kld_weight = float(cfg.get("kld_loss_weight", 0.0001))
    learning_rate = float(cfg.get("lr", 1e-3))
    clip_norm = float(cfg.get("clip_grad_norm", 1.0))
    prior_weight = float(cfg.get("prior_loss_weight", 5.0))
    sparsity_weight = float(cfg.get("sparsity_loss_weight", 1.0))
    min_bands_weight = float(cfg.get("min_bands_loss_weight", 50.0))
    min_bands_thresh = float(cfg.get("min_bands_threshold", 19.0))
    curvature_thresh = float(cfg.get("curvature_threshold", cfg.get("processing", {}).get("curvature_threshold", 3.37)))

    input_dim = X_np.shape[1]
    has_dem = (active_wvls is not None) and (len(active_wvls) == input_dim - 1)

    if active_wvls is None:
        active_wvls = np.linspace(0.4, 2.5, input_dim if not has_dem else input_dim - 1)

    X_spectra = X_np[:, :-1] if has_dem else X_np
    expert_prior = compute_expert_prior(X_spectra, active_wvls, has_dem=has_dem, curvature_threshold=curvature_thresh)

    model_name = cfg.get("name", "pbe_vae")
    print(
        f"[+] Initializing Model '{model_name}' ({l_dim}-dim latent space for {len(X_np):,} pixels)..."
    )
    print(f"  -> Training on device: {device}")
    model = build_vae(
        config=cfg,
        input_dim=input_dim,
        expert_prior=expert_prior,
        latent_dim=l_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_mean = np.mean(X_np, axis=0)
    X_std = np.std(X_np, axis=0) + 1e-8
    X_scaled = (X_np - X_mean) / X_std

    tensor_x = torch.Tensor(X_scaled)
    dataset = TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    t_prior = torch.from_numpy(expert_prior).float().to(device)
    target_retention = input_dim // 2

    for ep in range(epochs):
        ep_loss = 0.0
        gate_active = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            rec, mu, logvar, gate = model(batch_x)

            mse = nn.functional.mse_loss(rec, batch_x, reduction="mean")
            prior_loss = nn.functional.binary_cross_entropy(
                gate, t_prior.unsqueeze(0).expand_as(gate), reduction="mean"
            )
            active_bands = gate.sum(dim=1).mean()
            sparsity = torch.abs(active_bands - target_retention)
            min_bands_loss = torch.relu(min_bands_thresh - active_bands) * min_bands_weight
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = (
                rec_weight * mse
                + prior_weight * prior_loss
                + sparsity_weight * sparsity
                + min_bands_loss
                + kld_weight * kld
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

            ep_loss += loss.item()
            gate_active += gate.sum(dim=1).mean().item()

        if (ep + 1) % 25 == 0 or ep == epochs - 1:
            avg_loss = ep_loss / len(loader)
            avg_bands = gate_active / len(loader)
            print(
                f"  Epoch {ep+1}/{epochs} - Loss: {avg_loss:.4f} | Avg Active Bands: {avg_bands:.1f}/{input_dim}"
            )

    model.eval()
    extract_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    with torch.no_grad():
        for (batch_x,) in extract_loader:
            batch_x = batch_x.to(device)
            _, mu, _, _ = model(batch_x)
            latents.append(mu.cpu().numpy())

    return np.vstack(latents), model, X_mean, X_std


def apply_vae(
    X_np: np.ndarray,
    model: nn.Module,
    X_mean: np.ndarray,
    X_std: np.ndarray,
    batch_size: int = 4096,
    device: str = "cpu",
) -> np.ndarray:
    """Apply a trained VAE to new data and return latent representation means."""
    model.eval()
    X_scaled = (X_np - X_mean) / (X_std + 1e-8)
    tensor_x = torch.Tensor(X_scaled)
    dataset = TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            _, mu, _, _ = model(batch_x)
            latents.append(mu.cpu().numpy())
    return np.vstack(latents).astype(np.float32)
