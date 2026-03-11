"""
tests/test_temporal.py — Phase 4 software-behavior tests.

Tests ONLY software invariants, NOT experimental performance assertions.
AUROC thresholds are NOT asserted here.
"""
import os
import sys
import warnings
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.eval.temporal_eval import (
    anomaly_score_unsupervised,
    anomaly_score_normal_distance,
    detect_regime_changes,
    get_true_transitions,
    test_iteration_hypothesis as _iter_hypothesis_fn,
)

# ─ fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_cfg():
    """Minimal config that satisfies OrderedBeatDataset + get_ordered_loader."""
    return {
        'training': {'batch_size': 64, 'seed': 42, 'lr': 1e-4,
                     'epochs': 1, 'lambda_spectral': 0.005,
                     'beta_sparse': 0.0005, 'tau_spectral': 0.8},
        'data': {'type': 'synthetic', 'cache_dir': '/tmp/physionet_test'},
        'model': {'x_dim': 187, 'latent_dim': 8, 'hidden_dim': 32,
                  'deq_max_iter': 5, 'deq_tol': 1e-3},
        'logging': {'project': 'test', 'entity': None, 'log_every': 100},
    }


@pytest.fixture
def fake_ordered_dataset():
    """A synthetic OrderedBeatDataset-like object for testing without wfdb."""
    from torch.utils.data import Dataset

    class _FakeOrderedDataset(Dataset):
        def __init__(self, n=200, x_dim=187):
            self.segments = np.random.randn(n, x_dim).astype(np.float32)
            self.labels = np.random.randint(0, 5, size=n).astype(np.int64)

        def __len__(self):
            return len(self.segments)

        def __getitem__(self, index):
            return (
                torch.tensor(self.segments[index]),
                torch.tensor(self.labels[index]),
                index,
            )

    return _FakeOrderedDataset()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Ordered dataset preserves order
# ─────────────────────────────────────────────────────────────────────────────

def test_ordered_dataset_preserves_order(fake_ordered_dataset):
    """Consecutive items must have consecutive idx."""
    from torch.utils.data import DataLoader
    loader = DataLoader(fake_ordered_dataset, batch_size=16, shuffle=False, drop_last=False)
    prev_last = -1
    for batch in loader:
        _, _, idx = batch
        idx = idx.numpy()
        assert idx[0] == prev_last + 1, \
            f"Expected idx[0]={prev_last + 1}, got {idx[0]}"
        for i in range(1, len(idx)):
            assert idx[i] == idx[i - 1] + 1, \
                f"Non-consecutive idx: {idx[i - 1]} → {idx[i]}"
        prev_last = idx[-1]


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Ordered loader refuses shuffle=True
# ─────────────────────────────────────────────────────────────────────────────

def test_ordered_loader_refuses_shuffle(synthetic_cfg):
    """shuffle=True must raise ValueError immediately."""
    from src.data.timeseries import get_ordered_loader
    with pytest.raises(ValueError, match="shuffle=True is not permitted"):
        get_ordered_loader('100', synthetic_cfg, shuffle=True)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Anomaly score range
# ─────────────────────────────────────────────────────────────────────────────

def test_anomaly_score_range():
    """All anomaly score functions must return values in [0, 1]."""
    T, d = 200, 8
    energy = np.random.randn(T)
    z = np.random.randn(T, d)
    centroids = np.random.randn(3, d)
    normal_centroid = np.random.randn(d)

    for method in ['energy', 'centroid_distance', 'both']:
        scores = anomaly_score_unsupervised(energy, z, centroids, method=method)
        assert scores.shape == (T,), f"shape mismatch for method={method}"
        assert scores.min() >= -1e-9, f"score < 0 for method={method}"
        assert scores.max() <= 1 + 1e-9, f"score > 1 for method={method}"

    scores_li = anomaly_score_normal_distance(z, normal_centroid)
    assert scores_li.shape == (T,)
    assert scores_li.min() >= -1e-9
    assert scores_li.max() <= 1 + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Regime change returns valid indices
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_change_returns_valid_indices():
    """change_points must be in [0, T) and change_scores must have length T."""
    T, d = 300, 8
    z = np.random.randn(T, d)
    change_points, change_scores = detect_regime_changes(z, window=5, threshold=0.3)

    assert len(change_scores) == T, "change_scores length must equal T"
    for cp in change_points:
        assert 0 <= cp < T, f"change point {cp} out of range [0, {T})"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — True transitions respect min_duration
# ─────────────────────────────────────────────────────────────────────────────

def test_true_transitions_min_duration():
    """Short blips (< min_duration) must NOT count as true transitions."""
    # 100 normal, 3 anomalous (blip), 100 normal — no valid transition for min_dur=5
    labels = np.array([0] * 100 + [1, 1, 1] + [0] * 100)
    transitions = get_true_transitions(labels, min_duration=5)
    assert len(transitions) == 0, \
        f"Short blip should not be a transition, got {transitions}"

    # Now long-enough anomalous segment
    labels2 = np.array([0] * 100 + [1] * 10 + [0] * 100)
    transitions2 = get_true_transitions(labels2, min_duration=5)
    assert len(transitions2) > 0, "Long anomalous segment should trigger a transition"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Iteration hypothesis skips sparse anomalies
# ─────────────────────────────────────────────────────────────────────────────

def test_iteration_hyp_skips_sparse_anomalies():
    """Fewer than 10 anomalous beats must not crash and must return supported=False."""
    iter_seq = np.random.randint(3, 30, size=200).astype(float)
    labels = np.zeros(200, dtype=int)
    labels[:5] = 1

    result = _iter_hypothesis_fn(iter_seq, labels)
    assert result['hypothesis_supported'] is False
    assert result['warning'] is not None
    assert 'p_value' in result


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — Centroids cached and correctly reloaded
# ─────────────────────────────────────────────────────────────────────────────

def test_centroids_saved_and_loaded(tmp_path, synthetic_cfg):
    """Cached centroids must be reused on second call (no recomputation)."""
    from src.eval.temporal_eval import compute_centroids, _centroid_cache_path
    import unittest.mock as mock

    results_dir = str(tmp_path)
    source_records = ['100', '101']
    cache_path = _centroid_cache_path(source_records, 'centroids', results_dir)

    # Plant a fake centroid file
    fake_centroids = np.random.randn(3, 8)
    np.save(cache_path, fake_centroids)

    # compute_centroids should load from cache without calling model
    with mock.patch('src.eval.temporal_eval.compute_patient_trajectory') as mock_traj:
        loaded = compute_centroids(
            model=None,
            source_records=source_records,
            cfg=synthetic_cfg,
            device=torch.device('cpu'),
            results_dir=results_dir,
        )
    mock_traj.assert_not_called()
    np.testing.assert_array_equal(loaded, fake_centroids)
