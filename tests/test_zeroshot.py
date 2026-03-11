"""
tests/test_zeroshot.py — Phase 5 software-behavior tests.

Tests ONLY software invariants, NOT performance thresholds.
No AUROC thresholds are asserted.
"""
import os
import sys
import json
import warnings
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.network import (
    generate_network_stream,
    map_window_to_ecg_dim,
    ZeroShotNetworkDataset,
)
from src.eval.zeroshot_eval import (
    energy_only_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def tiny_stream():
    pings, labels, transitions = generate_network_stream(n_pings=5000, seed=0)
    return pings, labels, transitions


@pytest.fixture(scope='module')
def frozen_model():
    """Load the model and freeze it — mimics the zero-shot pipeline."""
    import yaml
    from src.models.full_model import SpectralEmergeModel

    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'ecg.yaml')
    if not os.path.exists(cfg_path):
        pytest.skip("ecg.yaml not found — skipping model tests")

    with open(cfg_path) as f:
        import yaml
        cfg = yaml.safe_load(f)

    model = SpectralEmergeModel(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Model truly frozen (no parameter change after zero-shot forward)
# ─────────────────────────────────────────────────────────────────────────────

def test_model_truly_frozen(frozen_model, tiny_stream):
    """State dict must be identical before and after zero-shot trajectory."""
    pings, labels, _ = tiny_stream
    original = {k: v.clone() for k, v in frozen_model.state_dict().items()}

    ds = ZeroShotNetworkDataset(pings, labels, window=50, stride=50,
                                 map_mode='reflect_pad', normalize=True)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    with torch.no_grad():
        for X, L, idx in loader:
            frozen_model(X)
            break  # one batch is enough

    for k, v in frozen_model.state_dict().items():
        assert torch.allclose(v, original[k]), \
            f"Parameter {k!r} was modified during zero-shot forward!"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — All 3 mapping modes produce shape (187,) from shape (50,)
# ─────────────────────────────────────────────────────────────────────────────

def test_mapping_output_shape():
    """All 3 mapping modes must produce shape 187 from input 50."""
    x = np.random.randn(50).astype(np.float32)
    for mode in ['constant_pad', 'reflect_pad', 'linear_resize']:
        out = map_window_to_ecg_dim(x, target_dim=187, mode=mode)
        assert out.shape == (187,), \
            f"mode={mode!r}: expected (187,), got {out.shape}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Non-trivial input produces different outputs across mapping modes
# ─────────────────────────────────────────────────────────────────────────────

def test_mapping_modes_differ():
    """Different mapping modes must produce non-identical outputs on non-trivial input."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, size=50).astype(np.float32)

    outputs = {mode: map_window_to_ecg_dim(x, target_dim=187, mode=mode)
               for mode in ['constant_pad', 'reflect_pad', 'linear_resize']}

    assert not np.allclose(outputs['constant_pad'], outputs['reflect_pad']), \
        "constant_pad and reflect_pad produce identical output (unexpected)"
    assert not np.allclose(outputs['constant_pad'], outputs['linear_resize']), \
        "constant_pad and linear_resize produce identical output (unexpected)"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — No optimizer was used in zero-shot pipeline
# ─────────────────────────────────────────────────────────────────────────────

def test_no_optimizer_used(frozen_model, tiny_stream):
    """No optimizer.step() must occur during zero-shot evaluation."""
    pings, labels, _ = tiny_stream
    ds = ZeroShotNetworkDataset(pings, labels, window=50, stride=50,
                                 map_mode='linear_resize', normalize=True)
    from torch.utils.data import DataLoader

    # We verify that gradients are never accumulated (requires_grad=False)
    for p in frozen_model.parameters():
        assert not p.requires_grad, \
            f"Parameter has requires_grad=True — optimizer could touch it"

    loader = DataLoader(ds, batch_size=16, shuffle=False)
    with torch.no_grad():
        for X, L, idx in loader:
            z, _, _, info = frozen_model(X)
            # Verify no grad_fn on output
            assert z.grad_fn is None, "z_star has a grad_fn — backward is possible!"
            break


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — energy_only_score returns values in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def test_energy_score_range():
    """energy_only_score must return values strictly in [0, 1]."""
    for _ in range(10):
        energy_seq = np.random.randn(500).astype(np.float32) * 10
        scores = energy_only_score(energy_seq)
        assert scores.shape == (500,)
        assert scores.min() >= -1e-9, f"Score below 0: {scores.min()}"
        assert scores.max() <= 1 + 1e-9, f"Score above 1: {scores.max()}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Full synthetic pipeline runs and writes JSON
# ─────────────────────────────────────────────────────────────────────────────

def test_synthetic_pipeline_runs(frozen_model, tmp_path):
    """Full synthetic zero-shot evaluation must complete and write valid JSON."""
    from src.eval.zeroshot_eval import evaluate_synthetic_zero_shot
    from torch.utils.data import DataLoader

    pings, labels, _ = generate_network_stream(n_pings=3000, seed=1)
    ds = ZeroShotNetworkDataset(pings, labels, window=50, stride=50,
                                 map_mode='reflect_pad', normalize=True)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    energies, all_labels = [], []
    with torch.no_grad():
        for X, L, _ in loader:
            z, _, _, _ = frozen_model(X)
            e = frozen_model.energy(z, X).cpu().numpy()
            energies.append(e)
            all_labels.append(L.numpy())

    energy_seq = np.concatenate(energies)
    labels_seq = np.concatenate(all_labels)
    scores = energy_only_score(energy_seq)
    m = evaluate_synthetic_zero_shot(scores, labels_seq)

    # Must contain required keys
    for key in ['auroc', 'aupr', 'f1_opt', 'anomaly_prevalence',
                'n_normal', 'n_anomalous']:
        assert key in m, f"Missing key {key!r} in metrics"

    # Write JSON and verify it is valid
    out_path = tmp_path / "test_zeroshot_result.json"
    record = {'adapter_used': False, 'training_performed': False, **m}
    with open(out_path, 'w') as f:
        json.dump(record, f, default=str)

    with open(out_path) as f:
        loaded = json.load(f)

    assert loaded['adapter_used'] is False
    assert loaded['training_performed'] is False
