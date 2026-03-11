"""
src/eval/zeroshot_eval.py — Phase 5: Zero-Shot Cross-Domain Evaluation

Scientific rule: PRIMARY zero-shot score = energy-only, NO target-domain labels.
All label-using analyses are clearly marked as calibrated or exploratory.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.mixture import GaussianMixture

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — ZERO-SHOT TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────

def compute_trajectory_zeroshot(model, dataset, device, batch_size=256):
    """
    Runs the FROZEN ECG model on mapped network windows.

    Returns dict:
        z_star_seq: (T, d) np.ndarray
        energy_seq: (T,) np.ndarray
        labels_seq: (T,) np.ndarray
        iter_seq:   (T,) np.ndarray
        index_seq:  (T,) np.ndarray
    """
    from torch.utils.data import DataLoader

    # Enforce model immutability
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    z_list, energy_list, label_list, iter_list, idx_list = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            X, labels, beat_idx = batch
            X = X.to(device)
            z_star, _, _, info = model(X)

            z_list.append(z_star.cpu().numpy())
            energy_list.append(model.energy(z_star, X).cpu().numpy())
            label_list.append(labels.numpy())
            iter_list.append(np.full(len(X), info['n_iters']))
            idx_list.append(beat_idx.numpy() if hasattr(beat_idx, 'numpy')
                            else np.array(beat_idx))

    # Use actual concatenated count T — never n_batches * batch_size
    return {
        'z_star_seq': np.concatenate(z_list, axis=0),
        'energy_seq': np.concatenate(energy_list, axis=0),
        'labels_seq': np.concatenate(label_list, axis=0),
        'iter_seq':   np.concatenate(iter_list, axis=0),
        'index_seq':  np.concatenate(idx_list, axis=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — ANOMALY SCORES
# ─────────────────────────────────────────────────────────────────────────────

def _minmax(arr):
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def energy_only_score(energy_seq):
    """
    PRIMARY ZERO-SHOT SCORE. Uses NO label information of any kind.

    Min-max normalized energy within the evaluated stream.
    Returns scores in [0, 1].
    """
    return _minmax(energy_seq)


def nearest_centroid_unsupervised_score(z_star_seq, centroids=None):
    """
    SECONDARY unsupervised score. NO target-domain labels allowed.

    If centroids is None, fits a GMM on z_star_seq itself (self-fitted).
    Label this result "self_fitted_unsupervised_latent" in reports.

    Returns:
        scores: (T,) in [0, 1]
        label:  "self_fitted_unsupervised_latent" if self-fitted
    """
    if centroids is None:
        # Fit GMM unsupervised on the evaluated stream itself
        best_bic, best_gmm = np.inf, None
        for K in range(2, min(8, len(z_star_seq) // 50 + 1)):
            try:
                gmm = GaussianMixture(n_components=K, covariance_type='diag',
                                      n_init=2, random_state=42)
                gmm.fit(z_star_seq)
                bic = gmm.bic(z_star_seq)
                if bic < best_bic:
                    best_bic, best_gmm = bic, gmm
            except Exception:
                continue
        if best_gmm is None:
            best_gmm = GaussianMixture(n_components=2, n_init=2, random_state=42)
            best_gmm.fit(z_star_seq)
        centroids = best_gmm.means_
        score_label = "self_fitted_unsupervised_latent"
    else:
        score_label = "external_centroids_unsupervised"

    dists = np.stack(
        [np.linalg.norm(z_star_seq - c, axis=1) for c in centroids], axis=1
    ).min(axis=1)
    return _minmax(dists), score_label


def normal_centroid_calibrated_score(z_star_seq, normal_centroid):
    """
    CALIBRATED / EXPLORATORY ONLY.
    Uses a label-informed centroid — NOT pure zero-shot.
    Must be labeled "calibrated_label_informed" in all reports.

    Returns scores in [0, 1].
    """
    dists = np.linalg.norm(z_star_seq - normal_centroid, axis=1)
    return _minmax(dists)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 — EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _base_metrics(scores, binary_labels):
    """Shared metric computation for both synthetic and real evaluation."""
    n_normal = int((binary_labels == 0).sum())
    n_anomalous = int((binary_labels == 1).sum())

    if n_anomalous == 0 or n_normal == 0:
        return {
            'auroc': float('nan'), 'aupr': float('nan'), 'f1_opt': float('nan'),
            'threshold_opt': float('nan'), 'n_normal': n_normal,
            'n_anomalous': n_anomalous,
            'anomaly_prevalence': n_anomalous / max(len(binary_labels), 1),
        }

    auroc = float(roc_auc_score(binary_labels, scores))
    aupr = float(average_precision_score(binary_labels, scores))
    prec, rec, thresh = precision_recall_curve(binary_labels, scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best = np.argmax(f1s)

    return {
        'auroc': auroc,
        'aupr': aupr,
        'f1_opt': float(f1s[best]),
        'threshold_opt': float(thresh[best]) if best < len(thresh) else float('nan'),
        'n_normal': n_normal,
        'n_anomalous': n_anomalous,
        'anomaly_prevalence': n_anomalous / len(binary_labels),
    }


def evaluate_synthetic_zero_shot(scores, labels):
    """
    For synthetic network data:
        label 0 = stable (normal)
        label != 0 = anomalous

    Returns standard metrics dict.
    """
    binary = (labels != 0).astype(int)
    return _base_metrics(scores, binary)


def evaluate_real_zero_shot(scores, pseudo_labels):
    """
    EXPLORATORY ONLY — uses weak pseudo-labels from RIPE Atlas.

    Returns metrics with label_type = "weak_pseudo_2std" attached.
    """
    binary = (pseudo_labels != 0).astype(int)
    m = _base_metrics(scores, binary)
    m['label_type'] = 'weak_pseudo_2std'
    m['note'] = 'EXPLORATORY — pseudo-labels only, not definitive'
    return m
