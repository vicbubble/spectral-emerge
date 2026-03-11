import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.eval.metrics import mode_count, spectral_gap, mmd_score
from src.models.full_model import SpectralEmergeModel

def test_mode_count_gmm():
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=500, n_features=4, centers=4, cluster_std=0.1, random_state=42)
    mc = mode_count(torch.tensor(X, dtype=torch.float32), max_components=10)
    assert 3 <= mc <= 5

def test_spectral_gap_positive():
    cfg = {
        'model': {'x_dim': 4, 'latent_dim': 8, 'hidden_dim': 16, 'deq_max_iter': 5, 'deq_tol': 1e-2}
    }
    model = SpectralEmergeModel(cfg)
    x = torch.randn(2, 4)
    gap = spectral_gap(model, x)
    assert gap > 0.0

def test_mmd_identical():
    z = torch.randn(100, 8)
    mmd = mmd_score(z, z)
    assert abs(mmd) < 1e-5
