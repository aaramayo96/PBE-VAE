"""
pbe_vae.models.classifier
=========================
Supervised target detection classifiers trained on VAE latent cluster features & DEM.
Supports selectable detector models (GBDT, Hist-GBDT, Random Forest, XGBoost, LightGBM)
with spatial LeaveOneGroupOut cross-validation and probability calibration.
"""

from __future__ import annotations
import os
import numpy as np
from typing import Tuple, Dict, Any, Union, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
        ExtraTreesClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator
    except ImportError:
        FrozenEstimator = None
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.model_selection import LeaveOneGroupOut
except ImportError:
    GradientBoostingClassifier = None
    HistGradientBoostingClassifier = None
    RandomForestClassifier = None
    ExtraTreesClassifier = None
    LogisticRegression = None
    MLPClassifier = None
    CalibratedClassifierCV = None
    FrozenEstimator = None
    RobustScaler = None
    StandardScaler = None
    LeaveOneGroupOut = None

from pbe_vae.models.vae import LightweightVAE, apply_vae


def cluster_soft_features(
    latents_std: np.ndarray,
    cluster_bundle: Dict[str, Any],
    prob_method: Optional[str] = None,
) -> np.ndarray:
    """
    Return soft cluster-membership probabilities for each pixel from standardised VAE latents.
    Supports selectable probability conversion methods:
      1. 'gmm': Gaussian Mixture Model posterior probabilities.
      2. 'softmax' / 'euclidean_softmax': Exponential negative distance softmax.
      3. 'rbf': Radial Basis Function (Gaussian kernel) soft assignment.
      4. 'student_t' / 'dec': Student-t kernel (as in Deep Embedded Clustering / t-SNE).
      5. 'cosine': Directional cosine similarity softmax.
    """
    model = cluster_bundle["model"]
    method = str(prob_method or cluster_bundle.get("prob_method", cluster_bundle.get("method", "gmm"))).lower().strip()
    n_clusters = int(cluster_bundle.get("n_clusters", 1))

    # Compute cluster centers if available
    centers = getattr(model, "cluster_centers_", None)
    if centers is None and hasattr(model, "means_"):
        centers = model.means_

    # Special handling for single-material target detection (n_clusters == 1)
    if n_clusters == 1:
        if hasattr(model, "score_samples") and "gmm" in method:
            log_dens = model.score_samples(latents_std)  # (N,)
            density_mean = cluster_bundle.get("density_mean")
            density_std = cluster_bundle.get("density_std")
            activation = str(cluster_bundle.get("single_cluster_activation", "sigmoid")).lower().strip()
            temperature = max(float(cluster_bundle.get("single_cluster_temperature", 1.0)), 1e-8)
            offset = float(cluster_bundle.get("single_cluster_offset", 0.0))
            if density_mean is not None and density_std is not None:
                z_score = (log_dens - float(density_mean)) / (float(density_std) + 1e-8)
                if activation == "sigmoid":
                    # Extreme scores are already saturated probabilities; clipping
                    # keeps the exponential numerically stable.
                    scaled = (z_score - offset) / temperature
                    proximity = 1.0 / (1.0 + np.exp(-np.clip(scaled, -60.0, 60.0)))
                elif activation in {"exp_density", "exponential"}:
                    density_max = float(cluster_bundle.get("density_max", np.max(log_dens)))
                    proximity = np.exp(
                        np.clip((log_dens - density_max - offset) / temperature, -60.0, 0.0)
                    )
                else:
                    raise ValueError(
                        f"Unsupported single_cluster_activation '{activation}'. "
                        "Choose 'sigmoid' or 'exp_density'."
                    )
            else:
                density_max = cluster_bundle.get("density_max")
                if density_max is None and centers is not None:
                    density_max = float(np.max(model.score_samples(centers)))
                if density_max is None:
                    density_max = float(np.max(log_dens))
                proximity = np.exp(np.clip(log_dens - float(density_max), -60.0, 0.0))
            return proximity[:, np.newaxis].astype(np.float32)
        elif centers is not None:
            sq_dists = np.sum((latents_std - centers[0]) ** 2, axis=-1)
            sigma = float(cluster_bundle.get("rbf_sigma", 1.0))
            proximity = np.exp(-sq_dists / (2.0 * sigma ** 2 + 1e-8))
            return proximity[:, np.newaxis].astype(np.float32)
        else:
            mean_z = np.mean(latents_std, axis=0, keepdims=True)
            sq_dists = np.sum((latents_std - mean_z) ** 2, axis=-1)
            proximity = np.exp(-sq_dists / (2.0 + 1e-8))
            return proximity[:, np.newaxis].astype(np.float32)

    # 1. GMM Posterior Probabilities for Multi-Cluster
    if hasattr(model, "predict_proba") and "gmm" in method:
        return model.predict_proba(latents_std).astype(np.float32)

    if centers is not None:
        # (N, K) Euclidean distance matrix
        diffs = latents_std[:, np.newaxis, :] - centers[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=-1)  # (N, K)
        dists = np.sqrt(np.maximum(sq_dists, 1e-12))

        # 2. Euclidean Distance Softmax
        if method in ["softmax", "euclidean_softmax", "dist_softmax"]:
            neg_d = -dists
            neg_d -= neg_d.max(axis=1, keepdims=True)
            exp_d = np.exp(neg_d)
            return (exp_d / exp_d.sum(axis=1, keepdims=True)).astype(np.float32)

        # 3. Radial Basis Function (Gaussian Kernel Softmax)
        elif method in ["rbf", "gaussian_kernel"]:
            sigma = float(cluster_bundle.get("rbf_sigma", 1.0))
            gamma = 1.0 / (2.0 * sigma ** 2)
            neg_sq = -gamma * sq_dists
            neg_sq -= neg_sq.max(axis=1, keepdims=True)
            exp_sq = np.exp(neg_sq)
            return (exp_sq / exp_sq.sum(axis=1, keepdims=True)).astype(np.float32)

        # 4. Student-t Kernel (DEC / t-SNE Soft Assignment)
        elif method in ["student_t", "dec", "t_distribution"]:
            alpha = float(cluster_bundle.get("t_alpha", 1.0))
            q = (1.0 + sq_dists / alpha) ** (- (alpha + 1.0) / 2.0)
            return (q / q.sum(axis=1, keepdims=True)).astype(np.float32)

        # 5. Cosine Similarity Softmax
        elif method in ["cosine", "cosine_softmax"]:
            z_norm = np.linalg.norm(latents_std, axis=1, keepdims=True) + 1e-8
            c_norm = np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
            cos_sim = np.dot(latents_std / z_norm, (centers / c_norm).T)  # (N, K)
            cos_scaled = cos_sim * 5.0  # Temperature scaling
            cos_scaled -= cos_scaled.max(axis=1, keepdims=True)
            exp_cos = np.exp(cos_scaled)
            return (exp_cos / exp_cos.sum(axis=1, keepdims=True)).astype(np.float32)

    if hasattr(model, "transform"):
        dists = model.transform(latents_std)
        neg_d = -dists
        neg_d -= neg_d.max(axis=1, keepdims=True)
        exp_d = np.exp(neg_d)
        return (exp_d / exp_d.sum(axis=1, keepdims=True)).astype(np.float32)

    labels = model.predict(latents_std)
    soft = np.zeros((len(labels), n_clusters), dtype=np.float32)
    for i, lbl in enumerate(labels):
        col = int(lbl) % n_clusters
        soft[i, col] = 1.0
    return soft


def make_features(
    raw_pixels: np.ndarray,
    vae_model: LightweightVAE,
    vae_X_mean: np.ndarray,
    vae_X_std: np.ndarray,
    cluster_bundle: Dict[str, Any],
    device: str = "cpu",
    batch_size: int = 4096,
    use_dem: bool = False,
    dem_weight: float = 1.0,
) -> np.ndarray:
    """
    Build the supervised detector feature vector for a 2-D array of raw pixels:
    Step 1: VAE encode -> (N, latent_dim)
    Step 2: Standardise latents using scaler
    Step 3: Cluster soft probabilities -> (N, n_clusters)
    Step 4: (Optional) If use_dem=True and DEM channel is present -> (N, n_clusters + 1)
    """
    has_dem = bool(use_dem and cluster_bundle.get("has_dem", False))
    if cluster_bundle.get("vae_input") == "hsi_only":
        vae_input = raw_pixels[:, : len(vae_X_mean)]
    else:
        vae_input = raw_pixels
    latents = apply_vae(
        vae_input, vae_model, vae_X_mean, vae_X_std,
        batch_size=batch_size, device=str(device)
    )

    dem_integration = str(cluster_bundle.get("dem_integration", "legacy_direct")).lower()
    if has_dem and dem_integration == "legacy_direct" and "dem_scaler" in cluster_bundle:
        dem_raw = raw_pixels[:, -1:]
        cluster_dem_weight = float(cluster_bundle.get("dem_weight_ratio", dem_weight))
        dem_feature = cluster_bundle["dem_scaler"].transform(dem_raw) * cluster_dem_weight
        cluster_input = np.hstack([latents, dem_feature])
    else:
        dem_feature = None
        cluster_input = latents

    latents_std = cluster_bundle["scaler"].transform(cluster_input)
    soft = cluster_soft_features(latents_std, cluster_bundle)

    if has_dem and dem_integration == "spectral_gate":
        dem_raw = raw_pixels[:, -1:]
        dem_z = cluster_bundle["dem_scaler"].transform(dem_raw).ravel()
        width = max(float(cluster_bundle.get("dem_similarity_width", 1.0)), 1e-8)
        dem_similarity = np.exp(-0.5 * np.square(np.clip(dem_z / width, -8.0, 8.0)))
        dem_fraction = float(np.clip(cluster_bundle.get("dem_weight_ratio", dem_weight), 0.0, 1.0))
        # DEM cannot produce a positive detection on its own.  At a weight of
        # 0.10 it can lower a spectral score by at most 10 percent.
        gate = (1.0 - dem_fraction) + dem_fraction * dem_similarity
        return (soft * gate[:, np.newaxis]).astype(np.float32)

    if has_dem:
        if dem_feature is None:
            dem_feature = raw_pixels[:, -1:] * float(dem_weight)
        return np.hstack([soft, dem_feature]).astype(np.float32)
    return soft.astype(np.float32)


def get_auto_device(preferred_device: Optional[str] = None) -> str:
    """Returns 'mps' on Apple Silicon, 'cuda' on NVIDIA GPUs, or 'cpu'."""
    if preferred_device is not None:
        return str(preferred_device)
    if torch is not None:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


class PyTorchMLPClassifier:
    """
    PyTorch-accelerated Neural Network classifier with automatic MPS / CUDA / CPU device acceleration.
    Fully compatible with scikit-learn Cross-Validation and CalibratedClassifierCV.
    """
    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 32),
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 4096,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = get_auto_device(device)
        self.random_state = random_state
        self.classes_ = np.array([0, 1])
        self.net_ = None

    def _build_net(self, input_dim: int) -> torch.nn.Sequential:
        import torch.nn as nn
        layers = []
        in_d = input_dim
        for h in self.hidden_dims:
            layers.extend([
                nn.Linear(in_d, h),
                nn.BatchNorm1d(h),
                nn.SiLU(),
                nn.Dropout(0.1),
            ])
            in_d = h
        layers.append(nn.Linear(in_d, 2))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(self.random_state)
        dev = torch.device(self.device)
        self.classes_ = np.unique(y)

        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net_ = self._build_net(X.shape[1]).to(dev)
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        self.net_.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(dev), by.to(dev)
                optimizer.zero_grad()
                out = self.net_(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        self.net_.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        dev = torch.device(self.device)
        if self.net_ is None:
            raise RuntimeError("PyTorchMLPClassifier is not fitted.")
        self.net_.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32).to(dev)
            logits = self.net_(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def get_params(self, deep=True):
        return {
            "hidden_dims": self.hidden_dims,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "device": self.device,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def build_base_classifier(
    classifier_type: str = "gradient_boosting",
    cfg: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Factory creating a base classifier model (GBDT, Hist-GBDT, PyTorch GPU/MPS Neural Net, Random Forest, XGBoost, LightGBM).
    """
    c_type = str(classifier_type).lower().strip()
    cfg = cfg or {}

    if c_type in ["neural_net", "mlp", "mlp_gpu", "pytorch_mlp", "nn"]:
        return PyTorchMLPClassifier(
            hidden_dims=cfg.get("hidden_dims", (64, 32)),
            lr=cfg.get("lr", 1e-3),
            epochs=cfg.get("epochs", 100),
            batch_size=cfg.get("batch_size", 4096),
            weight_decay=cfg.get("weight_decay", 1e-4),
            device=cfg.get("device", None),
            random_state=cfg.get("random_state", 42),
        )

    elif c_type in ["gbdt", "hist_gradient_boosting", "hist_gbdt", "hgb", "fast_gbdt"]:
        if HistGradientBoostingClassifier is None:
            raise ImportError("scikit-learn is required for HistGradientBoostingClassifier.")
        return HistGradientBoostingClassifier(
            max_iter=cfg.get("n_estimators", cfg.get("max_iter", 300)),
            learning_rate=cfg.get("learning_rate", 0.05),
            max_depth=cfg.get("max_depth", 6),
            min_samples_leaf=cfg.get("min_samples_leaf", 10),
            random_state=cfg.get("random_state", 42),
        )

    elif c_type in ["gradient_boosting", "exact_gbdt", "gbc"]:
        if GradientBoostingClassifier is None:
            raise ImportError("scikit-learn is required for GradientBoostingClassifier.")
        return GradientBoostingClassifier(
            n_estimators=cfg.get("n_estimators", 300),
            learning_rate=cfg.get("learning_rate", 0.05),
            max_depth=cfg.get("max_depth", 4),
            subsample=cfg.get("subsample", 0.8),
            min_samples_leaf=cfg.get("min_samples_leaf", 5),
            random_state=cfg.get("random_state", 42),
        )

    elif c_type in ["random_forest", "rf"]:
        if RandomForestClassifier is None:
            raise ImportError("scikit-learn is required for RandomForestClassifier.")
        return RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", 12),
            min_samples_leaf=cfg.get("min_samples_leaf", 3),
            n_jobs=cfg.get("n_jobs", -1),
            random_state=cfg.get("random_state", 42),
        )

    elif c_type in ["extra_trees", "et"]:
        if ExtraTreesClassifier is None:
            raise ImportError("scikit-learn is required for ExtraTreesClassifier.")
        return ExtraTreesClassifier(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", 12),
            n_jobs=cfg.get("n_jobs", -1),
            random_state=cfg.get("random_state", 42),
        )

    elif c_type in ["xgboost", "xgb"]:
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=cfg.get("n_estimators", 300),
                learning_rate=cfg.get("learning_rate", 0.05),
                max_depth=cfg.get("max_depth", 4),
                subsample=cfg.get("subsample", 0.8),
                random_state=cfg.get("random_state", 42),
                eval_metric="logloss",
                tree_method="hist",
                device=cfg.get("device", "cuda" if torch is not None and torch.cuda.is_available() else "cpu"),
            )
        except ImportError:
            raise ImportError("xgboost package is required for XGBClassifier. Install via 'pip install xgboost'.")

    elif c_type in ["lightgbm", "lgbm", "lgb"]:
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=cfg.get("n_estimators", 300),
                learning_rate=cfg.get("learning_rate", 0.05),
                max_depth=cfg.get("max_depth", 4),
                subsample=cfg.get("subsample", 0.8),
                random_state=cfg.get("random_state", 42),
                device_type=cfg.get("device", "cuda" if torch is not None and torch.cuda.is_available() else "cpu"),
            )
        except ImportError:
            raise ImportError("lightgbm package is required for LGBMClassifier. Install via 'pip install lightgbm'.")

    elif c_type in ["logistic_regression", "lr"]:
        if LogisticRegression is None:
            raise ImportError("scikit-learn is required for LogisticRegression.")
        return LogisticRegression(
            C=cfg.get("C", 1.0),
            max_iter=cfg.get("max_iter", 1000),
            random_state=cfg.get("random_state", 42),
        )

    else:
        raise ValueError(
            f"Unsupported classifier_type '{classifier_type}'. Available: "
            "['neural_net', 'hist_gradient_boosting', 'gradient_boosting', 'random_forest', 'extra_trees', 'xgboost', 'lightgbm', 'logistic_regression']"
        )


class PBEVAEClassifier:
    """
    Supervised target detector wrapping feature scaling, base classifier selection,
    spatial LeaveOneGroupOut cross-validation, and isotonic probability calibration.
    """
    def __init__(
        self,
        model=None,
        scaler=None,
        idxs=None,
        best_threshold: float = 0.85,
        classifier_type: str = "gradient_boosting",
        classifier_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.scaler = scaler
        self.idxs = idxs
        self.best_threshold = best_threshold
        self.classifier_type = classifier_type
        self.classifier_cfg = classifier_cfg or {}

    def fit(
        self,
        X_all: np.ndarray,
        target_flag: np.ndarray,
        groups: np.ndarray,
        classifier_type: Optional[str] = None,
        classifier_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Fits robust scaler, performs LeaveOneGroupOut spatial cross-validation,
        trains chosen detector model, and applies isotonic calibration.
        """
        c_type = classifier_type or self.classifier_type
        c_cfg = classifier_cfg or self.classifier_cfg

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_all)

        clf = build_base_classifier(c_type, c_cfg)

        bg_mask = (groups == 0)
        bg_indices = np.where(bg_mask)[0]
        unique_rois = np.unique(groups[~bg_mask]) if np.any(~bg_mask) else np.array([])

        splits = []
        if len(unique_rois) >= 2:
            if len(bg_indices) > 0:
                np.random.seed(42)
                np.random.shuffle(bg_indices)
                bg_chunks = np.array_split(bg_indices, len(unique_rois))
                for i, rid in enumerate(unique_rois):
                    groups[bg_chunks[i]] = rid
            logo = LeaveOneGroupOut()
            splits = list(logo.split(X_scaled, target_flag, groups))
            print(f"[+] Performing spatial Leave-One-Group-Out CV ({len(unique_rois)} spatial groups) with {c_type}...")
        else:
            from sklearn.model_selection import StratifiedKFold
            n_pos = int(target_flag.sum())
            n_splits = min(5, max(2, n_pos)) if n_pos >= 2 else 2
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(skf.split(X_scaled, target_flag))
            print(f"[+] Performing Stratified K-Fold CV ({n_splits} folds) with {c_type}...")

        y_true_bin, y_prob_all = [], []
        for train_idx, test_idx in splits:
            fold_clf = build_base_classifier(c_type, c_cfg)
            y_train = target_flag[train_idx]
            y_test = target_flag[test_idx]

            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
            negative_weight = float(c_cfg.get("background_negative_weight", 1.0))
            sw = np.where(y_train == 1, pos_weight, negative_weight)

            try:
                fold_clf.fit(X_scaled[train_idx], y_train, sample_weight=sw)
            except TypeError:
                fold_clf.fit(X_scaled[train_idx], y_train)

            probs = fold_clf.predict_proba(X_scaled[test_idx])
            prob_target = probs[:, 1] if probs.shape[1] == 2 else probs.ravel()

            y_true_bin.extend(y_test)
            y_prob_all.extend(prob_target)

        from pbe_vae.utils.metrics import compute_classification_metrics
        eval_stats = compute_classification_metrics(np.array(y_true_bin), np.array(y_prob_all))
        self.best_threshold = eval_stats["best_threshold"]

        print(f"[+] Fitting final {c_type} model on all data ({len(X_all):,} pixels) + probability calibration...")
        n_pos = int(target_flag.sum())
        n_neg = len(target_flag) - n_pos
        pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
        negative_weight = float(c_cfg.get("background_negative_weight", 1.0))
        sw_all = np.where(target_flag == 1, pos_weight, negative_weight)

        try:
            clf.fit(X_scaled, target_flag, sample_weight=sw_all)
        except TypeError:
            clf.fit(X_scaled, target_flag)

        cal_method = c_cfg.get("calibration", "isotonic")
        if cal_method:
            if FrozenEstimator is not None:
                calibrated = CalibratedClassifierCV(FrozenEstimator(clf), method=cal_method)
            else:
                calibrated = CalibratedClassifierCV(clf, cv="prefit", method=cal_method)
            calibrated.fit(X_scaled, target_flag)
            self.model = calibrated
        else:
            self.model = clf

        self.classifier_type = c_type
        return eval_stats

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts probability of target class (class 1) for scaled or unscaled features."""
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        probs = self.model.predict_proba(X_scaled)
        if hasattr(probs, "ndim") and probs.ndim == 2:
            if hasattr(self.model, "classes_"):
                classes = list(self.model.classes_)
                if 1 in classes:
                    return probs[:, classes.index(1)]
            if probs.shape[1] == 2:
                return probs[:, 1]
        return probs.ravel()
