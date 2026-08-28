import numpy as np
try:
    from sklearn.metrics import silhouette_score, davies_bouldin_score, precision_recall_curve, f1_score, cohen_kappa_score, auc
except ImportError:
    silhouette_score = None
    davies_bouldin_score = None
    precision_recall_curve = None
    f1_score = None
    cohen_kappa_score = None
    auc = None

def haversine_dist(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in meters.
    """
    R = 6371000.0  # Radius of earth in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0)**2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def compute_separability_metrics(X, labels, sample_size=20000, random_state=42):
    """
    Compute Silhouette Score and Davies-Bouldin Index for latent clusters.
    """
    try:
        n_samples = len(X)
        ssize = min(sample_size, n_samples)
        sil = float(silhouette_score(X, labels, sample_size=ssize, random_state=random_state))
        db = float(davies_bouldin_score(X, labels))
    except Exception:
        sil, db = 0.0, 0.0
    return sil, db

def compute_classification_metrics(y_true, y_prob):
    """
    Compute optimal threshold, F1 score, Cohen's Kappa, and PR-AUC.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    
    if len(thresholds) > 0:
        best_idx = int(np.argmax(f1_scores))
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])
    else:
        best_threshold = 0.5
        best_f1 = 0.0

    y_pred = (y_prob >= best_threshold).astype(int)
    kappa = float(cohen_kappa_score(y_true, y_pred))
    pr_auc = float(auc(recall, precision))

    return {
        'best_threshold': best_threshold,
        'f1': best_f1,
        'kappa': kappa,
        'pr_auc': pr_auc,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    }

def compute_bbox_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) of two bounding boxes (r0, c0, r1, c1).
    """
    g_r0, g_c0, g_r1, g_c1 = boxA
    p_r0, p_c0, p_r1, p_c1 = boxB

    int_r0 = max(g_r0, p_r0)
    int_r1 = min(g_r1, p_r1)
    int_c0 = max(g_c0, p_c0)
    int_c1 = min(g_c1, p_c1)

    inter = max(0, int_r1 - int_r0) * max(0, int_c1 - int_c0)
    areaA = (g_r1 - g_r0) * (g_c1 - g_c0)
    areaB = (p_r1 - p_r0) * (p_c1 - p_c0)
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0.0
