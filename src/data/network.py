"""
src/data/network.py — Phase 5: Synthetic & Real Network Latency Data

Provides:
  - generate_network_stream: synthetic 4-regime RTT stream
  - map_window_to_ecg_dim: deterministic R^50 → R^187 (no learned params)
  - ZeroShotNetworkDataset: windowed dataset for zero-shot evaluation
  - download_ripe_atlas + RipeAtlasDataset: real RTT data (optional)
"""
import os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# REGIME LABELS
# ─────────────────────────────────────────────────────────────────────────────

REGIME_LABELS = {
    0: "stable",
    1: "congestion",
    2: "packet_loss",
    3: "route_change",
}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1.1 — SYNTHETIC NETWORK STREAM
# ─────────────────────────────────────────────────────────────────────────────

def generate_network_stream(n_pings=50000, seed=42):
    """
    Generates synthetic network latency stream with 4 regimes.

    Structure: 0 -> 1 -> 0 -> 2 -> 0 -> 3 -> 0 (repeated until n_pings)
    Always returns to stable (0) between anomalous regimes.

    Regime 0 — stable:
        ~ N(20ms, 2ms), clipped to [1, 40]
    Regime 1 — congestion:
        gradually rising baseline 20→60ms, variance 2→15ms
    Regime 2 — packet_loss:
        mostly stable N(20, 2) with 10% chance of large spike (N(400, 50))
    Regime 3 — route_change:
        sudden step to N(80, 3) — different stable baseline

    Returns:
        pings:       (N,) float32
        labels:      (N,) int64
        transitions: list of (index, from_regime, to_regime)
    """
    rng = np.random.default_rng(seed)

    pings = np.zeros(n_pings, dtype=np.float32)
    labels = np.zeros(n_pings, dtype=np.int64)
    transitions = []

    # Segment lengths (in pings)
    stable_len = 5000      # normal stable segment
    anomaly_len = 3000     # anomaly duration

    sequence = [0, 1, 0, 2, 0, 3, 0]  # repeated pattern

    t = 0
    seg_idx = 0
    current_regime = 0

    while t < n_pings:
        regime = sequence[seg_idx % len(sequence)]
        length = stable_len if regime == 0 else anomaly_len
        length = min(length, n_pings - t)

        end = t + length
        ts = np.arange(length)

        if regime == 0:  # stable
            seg = rng.normal(20.0, 2.0, size=length)

        elif regime == 1:  # congestion — gradual rise
            baseline = np.linspace(20.0, 60.0, length)
            variance = np.linspace(2.0, 15.0, length)
            seg = baseline + rng.normal(0, 1, size=length) * variance

        elif regime == 2:  # packet_loss — rare large spikes
            seg = rng.normal(20.0, 2.0, size=length)
            spike_mask = rng.random(length) < 0.10
            seg[spike_mask] = rng.normal(400.0, 50.0, size=spike_mask.sum())

        elif regime == 3:  # route_change — step to new baseline
            seg = rng.normal(80.0, 3.0, size=length)

        seg = np.clip(seg, 1.0, 2000.0).astype(np.float32)
        pings[t:end] = seg
        labels[t:end] = regime

        if regime != current_regime:
            transitions.append((t, current_regime, regime))
            current_regime = regime

        t = end
        seg_idx += 1

    return pings, labels, transitions


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1.2 — DETERMINISTIC INPUT MAPPING R^50 → R^187
# ─────────────────────────────────────────────────────────────────────────────

def map_window_to_ecg_dim(x, target_dim=187, mode="reflect_pad"):
    """
    Deterministic mapping from network window length 50 to ECG input length 187.
    NO learned parameters. NO adaptation.

    Allowed modes:
        "constant_pad"  — zero-pad to 187
        "reflect_pad"   — reflection padding via np.pad
        "linear_resize" — deterministic linear interpolation

    Args:
        x: (50,) or (batch, 50)
        target_dim: 187
        mode: mapping mode
    Returns:
        same type, shape (187,) or (batch, 187)
    """
    batched = (x.ndim == 2)
    if not batched:
        x = x[None, :]  # (1, 50)

    src_len = x.shape[1]
    pad_total = target_dim - src_len

    if mode == "constant_pad":
        result = np.pad(x, ((0, 0), (0, pad_total)), mode='constant', constant_values=0)

    elif mode == "reflect_pad":
        # Repeat reflection to fill target_dim
        result = np.pad(x, ((0, 0), (0, pad_total)), mode='reflect')

    elif mode == "linear_resize":
        src_coords = np.linspace(0, src_len - 1, src_len)
        tgt_coords = np.linspace(0, src_len - 1, target_dim)
        result = np.stack([
            np.interp(tgt_coords, src_coords, row) for row in x
        ], axis=0).astype(np.float32)

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose from: constant_pad, reflect_pad, linear_resize")

    return result[0] if not batched else result


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1.3 — ZERO-SHOT NETWORK DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ZeroShotNetworkDataset(Dataset):
    """
    Segments network stream into windows and applies deterministic input mapping.

    Args:
        pings:     (N,) float32 RTT stream
        labels:    (N,) int64 regime labels (0=stable, 1/2/3=anomalous)
        window:    window size (50)
        stride:    stride (1 for temporal evaluation)
        map_mode:  one of {constant_pad, reflect_pad, linear_resize}
        normalize: apply per-window z-score BEFORE mapping (no labels used)

    Returns per item: (x: float32 tensor (187,), label: int64, idx: int)
    """
    def __init__(self, pings, labels, window=50, stride=1,
                 map_mode="reflect_pad", normalize=True):
        self.map_mode = map_mode
        self.normalize = normalize
        self.window = window

        N = len(pings)
        starts = np.arange(0, N - window + 1, stride)
        self.x_raw = np.stack([pings[s:s + window] for s in starts], axis=0).astype(np.float32)
        # Window label = majority label in window
        self.labels = np.array(
            [int(np.bincount(labels[s:s + window]).argmax()) for s in starts],
            dtype=np.int64
        )
        self.indices = starts.astype(np.int64)

    def __len__(self):
        return len(self.x_raw)

    def __getitem__(self, idx):
        x = self.x_raw[idx].copy()
        if self.normalize:
            mu, sigma = x.mean(), x.std()
            sigma = max(sigma, 1e-6)
            x = (x - mu) / sigma

        x = map_window_to_ecg_dim(x, target_dim=187, mode=self.map_mode)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            int(self.indices[idx]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1.4 — RIPE ATLAS REAL DATA
# ─────────────────────────────────────────────────────────────────────────────

def download_ripe_atlas(probe_ids, measurement_id, output_dir, n_results=10000):
    """
    Downloads RTT measurements from RIPE Atlas public API (no auth required).

    Returns:
        pings:      (N,) float32
        timestamps: (N,) int64
    or None if unavailable.
    """
    try:
        import urllib.request
        import json

        os.makedirs(output_dir, exist_ok=True)
        cache_path = os.path.join(output_dir, f"ripe_msm{measurement_id}.json")

        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                data = json.load(f)
        else:
            url = (f"https://atlas.ripe.net/api/v2/measurements/{measurement_id}"
                   f"/results/?format=json&msm_id={measurement_id}"
                   f"&num={n_results}")
            if probe_ids:
                probe_str = ','.join(str(p) for p in probe_ids)
                url += f"&probe_ids={probe_str}"

            req = urllib.request.Request(url, headers={'User-Agent': 'spectral-emerge/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            with open(cache_path, 'w') as f:
                json.dump(data, f)

        pings, timestamps = [], []
        for result in data:
            ts = result.get('timestamp', 0)
            # avg field for RTT
            avg = result.get('avg', None)
            if avg is not None and avg > 0:
                pings.append(float(avg))
                timestamps.append(int(ts))

        if not pings:
            return None

        return np.array(pings, dtype=np.float32), np.array(timestamps, dtype=np.int64)

    except Exception as e:
        warnings.warn(f"download_ripe_atlas failed: {e} — RIPE Atlas data unavailable.")
        return None


class RipeAtlasDataset(ZeroShotNetworkDataset):
    """
    Wraps RIPE Atlas RTT data for zero-shot evaluation.

    For real data, no true anomaly labels exist.
    Pseudo-labels for EXPLORATORY analysis:
        burn_in = first 20% of stream
        threshold = mean + 2*std over burn_in
        pseudo_label = 1 if rtt >= threshold else 0

    These are WEAK PSEUDO-LABELS:
        label_type = "weak_pseudo_2std"
    Must NOT be used as primary zero-shot metric.
    """
    LABEL_TYPE = "weak_pseudo_2std"

    def __init__(self, pings, window=50, stride=10,
                 map_mode="reflect_pad", normalize=True):
        # Build weak pseudo-labels
        burn_in = max(1, int(0.20 * len(pings)))
        mu_burn = pings[:burn_in].mean()
        std_burn = pings[:burn_in].std()
        threshold = mu_burn + 2.0 * std_burn
        pseudo_labels = (pings >= threshold).astype(np.int64)

        super().__init__(pings, pseudo_labels, window=window, stride=stride,
                         map_mode=map_mode, normalize=normalize)
        self.label_type = self.LABEL_TYPE
        self.threshold = float(threshold)
