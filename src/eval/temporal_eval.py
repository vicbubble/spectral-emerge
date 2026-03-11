"""
temporal_eval.py — Phase 4: Temporal Evaluation for Anomaly Detection and Regime Discovery

Scientific question:
  "Given an ordered stream of heartbeats from a single patient, do the model
   outputs — z*(x), E(z*, x), and DEQ solver behavior — evolve in a way that
   usefully signals anomalous beats and sustained regime transitions, without
   labels at inference time?"

KEY INVARIANTS:
  - Beat is the fundamental unit of analysis
  - shuffle=False always
  - No record contributes to its own centroids or calibration statistics
  - Fully unsupervised scores are reported separately from label-informed scores
"""
import os
import hashlib
import logging
import warnings
import numpy as np
import torch
from scipy.stats import mannwhitneyu
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — TRAJECTORY COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_patient_trajectory(model, record_id, cfg, device):
    """
    Runs the model over all beats of one patient in temporal order.
    The model processes each beat independently — this is NOT temporal modeling.

    Returns dict:
        z_star_seq: (T, d) np.ndarray
        energy_seq: (T,) np.ndarray
        labels_seq: (T,) np.ndarray
        iter_seq:   (T,) np.ndarray
        beat_index: (T,) np.ndarray
        record_id:  str
    """
    from ..data.timeseries import get_ordered_loader

    loader = get_ordered_loader(record_id, cfg, shuffle=False)
    if len(loader.dataset) == 0:
        logger.warning(f"Record {record_id}: empty dataset — skipping trajectory.")
        return None

    model.eval()
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
            idx_list.append(beat_idx.numpy())

    # Actual beat count T — never reconstructed as n_batches * batch_size
    T = sum(len(z) for z in z_list)
    return {
        'z_star_seq': np.concatenate(z_list, axis=0),    # (T, d)
        'energy_seq': np.concatenate(energy_list, axis=0), # (T,)
        'labels_seq': np.concatenate(label_list, axis=0),  # (T,)
        'iter_seq':   np.concatenate(iter_list, axis=0),   # (T,)
        'beat_index': np.concatenate(idx_list, axis=0),    # (T,)
        'record_id':  str(record_id),
        'T':          T,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — CENTROIDS AND CALIBRATION (LEAKAGE-FREE)
# ─────────────────────────────────────────────────────────────────────────────

def _centroid_cache_path(source_records, suffix, results_dir='experiments/results'):
    """SHA256 hash of sorted record list → cache filename."""
    key = '_'.join(sorted(str(r) for r in source_records))
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    return os.path.join(results_dir, f'{suffix}_{h}.npy')


def compute_centroids(model, source_records, cfg, device,
                      results_dir='experiments/results'):
    """
    Computes emergent latent centroids from source_records ONLY.

    Fits GMM with BIC to choose K.
    centroid[k] = mean z* assigned to cluster k.

    Cached at: experiments/results/centroids_<hash>.npy

    Returns:
        centroids: (K, d) np.ndarray
    """
    cache_path = _centroid_cache_path(source_records, 'centroids', results_dir)
    if os.path.exists(cache_path):
        logger.info(f"Loading cached centroids from {cache_path}")
        return np.load(cache_path)

    z_all = []
    for rec in source_records:
        traj = compute_patient_trajectory(model, rec, cfg, device)
        if traj is not None:
            z_all.append(traj['z_star_seq'])

    if not z_all:
        raise RuntimeError("compute_centroids: no z* collected from source_records.")

    Z = np.concatenate(z_all, axis=0)

    # BIC model selection
    best_bic = np.inf
    best_gmm = None
    for K in range(2, min(11, len(Z) // 20 + 1)):
        try:
            gmm = GaussianMixture(n_components=K, covariance_type='diag',
                                  n_init=3, random_state=42)
            gmm.fit(Z)
            bic = gmm.bic(Z)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except Exception:
            continue

    if best_gmm is None:
        best_gmm = GaussianMixture(n_components=2, covariance_type='diag',
                                   n_init=3, random_state=42)
        best_gmm.fit(Z)

    assignments = best_gmm.predict(Z)
    K = best_gmm.n_components
    d = Z.shape[1]
    centroids = np.zeros((K, d))
    for k in range(K):
        mask = assignments == k
        if mask.sum() > 0:
            centroids[k] = Z[mask].mean(axis=0)

    os.makedirs(results_dir, exist_ok=True)
    np.save(cache_path, centroids)
    logger.info(f"Centroids (K={K}) saved to {cache_path}")
    return centroids


def compute_normal_centroid(model, source_records, cfg, device,
                            results_dir='experiments/results'):
    """
    Computes a label-informed centroid of Normal beats (label=0)
    from source_records ONLY.

    THIS IS NOT FULLY UNSUPERVISED.
    It is a calibration object and must be reported as such.

    Cached at: experiments/results/normal_centroid_<hash>.npy

    Returns:
        normal_centroid: (d,) np.ndarray
    """
    cache_path = _centroid_cache_path(source_records, 'normal_centroid', results_dir)
    if os.path.exists(cache_path):
        logger.info(f"Loading cached normal centroid from {cache_path}")
        return np.load(cache_path)

    z_normal = []
    for rec in source_records:
        traj = compute_patient_trajectory(model, rec, cfg, device)
        if traj is None:
            continue
        mask = traj['labels_seq'] == 0
        if mask.sum() > 0:
            z_normal.append(traj['z_star_seq'][mask])

    if not z_normal:
        raise RuntimeError("compute_normal_centroid: no Normal beats found in source_records.")

    Z_n = np.concatenate(z_normal, axis=0)
    normal_centroid = Z_n.mean(axis=0)

    os.makedirs(results_dir, exist_ok=True)
    np.save(cache_path, normal_centroid)
    logger.info(f"Normal centroid saved to {cache_path}")
    return normal_centroid


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 — ANOMALY SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _minmax(arr):
    """Min-max normalize to [0, 1]. If constant, returns zeros."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def anomaly_score_unsupervised(energy_seq, z_star_seq, centroids, method='both'):
    """
    Fully unsupervised anomaly score. Uses NO label information.

    method="energy":
        min-max normalized energy per record

    method="centroid_distance":
        min-max normalized distance from z* to nearest emergent centroid

    method="both":
        mean of the two normalized scores

    Returns:
        scores: (T,) in [0, 1]
    """
    if method == 'energy':
        return _minmax(energy_seq)

    # Nearest-centroid distance
    dists = np.stack(
        [np.linalg.norm(z_star_seq - c, axis=1) for c in centroids], axis=1
    )  # (T, K)
    min_dist = dists.min(axis=1)  # (T,)
    score_dist = _minmax(min_dist)

    if method == 'centroid_distance':
        return score_dist

    # method == 'both'
    score_energy = _minmax(energy_seq)
    return (score_energy + score_dist) / 2.0


def anomaly_score_normal_distance(z_star_seq, normal_centroid):
    """
    Label-informed anomaly score.
    Uses distance from z* to normal_centroid computed from training records ONLY.
    THIS IS NOT FULLY UNSUPERVISED — report as calibrated score.

    Returns:
        scores: (T,) in [0, 1]
    """
    dists = np.linalg.norm(z_star_seq - normal_centroid, axis=1)
    return _minmax(dists)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 — REGIME CHANGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_regime_changes(z_star_seq, window=5, threshold=0.5):
    """
    Detects sustained changes in latent trajectory.

    Steps:
    1. raw_score[t] = cosine_distance(z[t], z[t-1])
    2. smooth with centered moving average of width=window (mandatory)
    3. detect local peaks above threshold

    Returns:
        change_points: (K,) int indices
        change_scores: (T,) float array
    """
    from scipy.signal import find_peaks
    from scipy.spatial.distance import cosine

    T = len(z_star_seq)
    raw = np.zeros(T)
    for t in range(1, T):
        try:
            raw[t] = cosine(z_star_seq[t], z_star_seq[t - 1])
        except Exception:
            raw[t] = 0.0

    # Centered moving average
    half = window // 2
    smooth = np.convolve(raw, np.ones(window) / window, mode='same')

    peaks, _ = find_peaks(smooth, height=threshold)
    return peaks.astype(int), smooth


def get_true_transitions(labels_seq, min_duration=5):
    """
    Ground truth transition points from SUSTAINED class changes.

    A transition is valid only if the new class persists for at least
    min_duration consecutive beats. Single-beat blips are excluded.

    Returns:
        transitions: (K,) int indices
    """
    T = len(labels_seq)
    transitions = []
    t = 0
    while t < T - 1:
        if labels_seq[t] != labels_seq[t + 1]:
            new_class = labels_seq[t + 1]
            # check how long new_class persists
            end = t + 1
            while end < T and labels_seq[end] == new_class:
                end += 1
            duration = end - (t + 1)
            if duration >= min_duration:
                transitions.append(t + 1)
            t = end  # jump past this segment
        else:
            t += 1
    return np.array(transitions, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 6 — METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_anomaly_detection(scores, labels):
    """
    Evaluates anomaly detection where label 0 = normal, label != 0 = anomalous.

    Returns dict with:
        auroc, aupr, fpr_at_95tpr, threshold_opt,
        precision_opt, recall_opt, f1_opt,
        auroc_per_class, n_normal, n_anomalous, anomaly_prevalence
    """
    binary = (labels != 0).astype(int)
    n_normal = int((binary == 0).sum())
    n_anomalous = int((binary == 1).sum())

    if n_anomalous == 0 or n_normal == 0:
        logger.warning("evaluate_anomaly_detection: only one class present — returning NaN metrics.")
        return {
            'auroc': float('nan'), 'aupr': float('nan'), 'fpr_at_95tpr': float('nan'),
            'threshold_opt': float('nan'), 'precision_opt': float('nan'),
            'recall_opt': float('nan'), 'f1_opt': float('nan'),
            'auroc_per_class': {}, 'n_normal': n_normal,
            'n_anomalous': n_anomalous,
            'anomaly_prevalence': n_anomalous / max(len(labels), 1),
        }

    auroc = roc_auc_score(binary, scores)
    aupr = average_precision_score(binary, scores)

    fpr, tpr, thresh_roc = roc_curve(binary, scores)
    idx_95 = np.searchsorted(tpr, 0.95)
    fpr_at_95tpr = float(fpr[min(idx_95, len(fpr) - 1)])

    prec, rec, thresh_pr = precision_recall_curve(binary, scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_f1_idx = np.argmax(f1s)

    auroc_per_class = {}
    for cls in np.unique(labels):
        if cls == 0:
            continue
        binary_cls = ((labels == 0) | (labels == cls)).astype(bool)
        scores_cls = scores[binary_cls]
        labels_cls = (labels[binary_cls] != 0).astype(int)
        if labels_cls.sum() > 0 and (labels_cls == 0).sum() > 0:
            try:
                auroc_per_class[int(cls)] = float(roc_auc_score(labels_cls, scores_cls))
            except Exception:
                auroc_per_class[int(cls)] = float('nan')

    return {
        'auroc': float(auroc),
        'aupr': float(aupr),
        'fpr_at_95tpr': fpr_at_95tpr,
        'threshold_opt': float(thresh_pr[best_f1_idx]) if len(thresh_pr) > best_f1_idx else float('nan'),
        'precision_opt': float(prec[best_f1_idx]),
        'recall_opt': float(rec[best_f1_idx]),
        'f1_opt': float(f1s[best_f1_idx]),
        'auroc_per_class': auroc_per_class,
        'n_normal': n_normal,
        'n_anomalous': n_anomalous,
        'anomaly_prevalence': n_anomalous / len(labels),
    }


def evaluate_regime_detection(change_points, true_transitions, tolerance=3):
    """
    Matches detected change points to sustained ground-truth transitions.
    Each true transition can be matched at most once.

    Returns dict with:
        precision, recall, f1, mean_delay, n_detected, n_true
    """
    n_detected = len(change_points)
    n_true = len(true_transitions)

    if n_true == 0:
        return {
            'precision': float('nan'), 'recall': float('nan'), 'f1': float('nan'),
            'mean_delay': float('nan'), 'n_detected': n_detected, 'n_true': 0,
        }

    matched_true = set()
    delays = []

    for cp in change_points:
        for j, tt in enumerate(true_transitions):
            if j not in matched_true and abs(int(cp) - int(tt)) <= tolerance:
                matched_true.add(j)
                delays.append(int(cp) - int(tt))
                break

    tp = len(matched_true)
    precision = tp / max(n_detected, 1)
    recall = tp / max(n_true, 1)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    mean_delay = float(np.mean(delays)) if delays else float('nan')

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mean_delay': mean_delay,
        'n_detected': n_detected,
        'n_true': n_true,
    }


def test_iteration_hypothesis(iter_seq, labels_seq):
    """
    Tests whether anomalous beats require more DEQ iterations than normal beats.
    Uses MannWhitneyU (one-sided, 'greater').

    If fewer than 10 anomalous beats: skip test, return hypothesis_supported=False + warning.

    Returns dict with:
        statistic, p_value, normal_mean_iters, anomalous_mean_iters,
        hypothesis_supported, warning (if applicable)
    """
    normal_iters = iter_seq[labels_seq == 0]
    anomalous_iters = iter_seq[labels_seq != 0]

    if len(anomalous_iters) < 10 or len(normal_iters) == 0:
        return {
            'statistic': float('nan'),
            'p_value': float('nan'),
            'normal_mean_iters': float(normal_iters.mean()) if len(normal_iters) > 0 else float('nan'),
            'anomalous_mean_iters': float(anomalous_iters.mean()) if len(anomalous_iters) > 0 else float('nan'),
            'hypothesis_supported': False,
            'warning': f"Only {len(anomalous_iters)} anomalous / {len(normal_iters)} normal beats — test skipped.",
        }

    stat, p = mannwhitneyu(anomalous_iters, normal_iters, alternative='greater')
    return {
        'statistic': float(stat),
        'p_value': float(p),
        'normal_mean_iters': float(normal_iters.mean()),
        'anomalous_mean_iters': float(anomalous_iters.mean()),
        'hypothesis_supported': bool(p < 0.05),
        'warning': None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 7 — BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def compute_baselines(energy_seq, labels):
    """
    Baselines for anomaly detection.

    1. random_baseline: random scores ∈ [0,1], expected AUROC ≈ 0.5
    2. constant_normal_baseline: all zeros, AUROC = 0.5 by convention
    3. raw_energy_baseline: normalized raw energy only

    Returns dict: random_auroc, constant_auroc (0.5), raw_energy_auroc
    """
    binary = (labels != 0).astype(int)
    if len(np.unique(binary)) < 2:
        return {'random_auroc': float('nan'), 'constant_auroc': 0.5, 'raw_energy_auroc': float('nan')}

    np.random.seed(0)
    random_scores = np.random.rand(len(labels))
    random_auroc = float(roc_auc_score(binary, random_scores))

    logger.info("constant_normal_baseline: constant classifier — non-discriminative by definition")
    constant_auroc = 0.5

    raw_energy = _minmax(energy_seq)
    raw_energy_auroc = float(roc_auc_score(binary, raw_energy))

    return {
        'random_auroc': random_auroc,
        'constant_auroc': constant_auroc,
        'raw_energy_auroc': raw_energy_auroc,
    }
