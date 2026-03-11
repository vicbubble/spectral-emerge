"""
experiments/zeroshot_ablation.py — Phase 5 Zero-Shot Ablation

Distinguishes whether zero-shot success is due to:
  A. meaningful signal structure from the input
  B. padding artifact from the deterministic mapping

Also compares ECG vs. network latent geometry.

Usage:
    python experiments/zeroshot_ablation.py
"""
import os
import sys
import json
import yaml
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.full_model import SpectralEmergeModel
from src.data.network import (
    generate_network_stream,
    ZeroShotNetworkDataset,
    map_window_to_ecg_dim,
)
from src.eval.zeroshot_eval import (
    energy_only_score,
    evaluate_synthetic_zero_shot,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MAP_MODES = ["constant_pad", "reflect_pad", "linear_resize"]


# ─────────────────────────────────────────────────────────────────────────────
# TASK 6.1 — ABLATION: STRUCTURE VS PADDING ARTIFACT
# ─────────────────────────────────────────────────────────────────────────────

def ablation_padding_vs_structure(model, map_mode, n_windows=1000, seed=99):
    """
    For each of n_windows matched normal/anomalous pairs:
      1. Original:  map actual window → energy
      2. Shuffled:  shuffle window contents then map → energy
      3. Noise:     replace window with matched-mean/std random noise → energy

    If shuffled/noise preserves anomaly separation → padding artifact.
    If original >> shuffled == noise → real structural transfer.

    Returns dict with AUROC for each condition per map_mode.
    """
    rng = np.random.default_rng(seed)
    pings, labels, _ = generate_network_stream(n_pings=100000, seed=seed)

    # Sample balanced normal and anomalous window centers
    win = 50
    normal_idxs = np.where(labels[win:] == 0)[0]
    anom_idxs   = np.where(labels[win:] != 0)[0]

    n_each = min(n_windows // 2, len(normal_idxs), len(anom_idxs))
    norm_starts = rng.choice(normal_idxs, size=n_each, replace=False)
    anom_starts = rng.choice(anom_idxs,  size=n_each, replace=False)
    all_starts = np.concatenate([norm_starts, anom_starts])
    all_labels = np.array([0] * n_each + [1] * n_each, dtype=np.int64)

    conditions = {
        'original': [],
        'shuffled': [],
        'noise':    [],
    }

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for s in all_starts:
        segment = pings[s:s + win].astype(np.float32)

        # per-window z-score
        mu, sigma = segment.mean(), segment.std()
        sigma = max(sigma, 1e-6)
        seg_norm = (segment - mu) / sigma

        # shuffled: same values, random order
        shuffled = rng.permutation(seg_norm)

        # noise: matched mean=0, std=1 (z-score space) gaussian noise
        noise = rng.normal(0, 1, size=win).astype(np.float32)

        for name, x in [('original', seg_norm), ('shuffled', shuffled), ('noise', noise)]:
            x_mapped = map_window_to_ecg_dim(x, target_dim=187, mode=map_mode)
            conditions[name].append(x_mapped)

    device = next(model.parameters()).device
    results = {}
    for name, windows in conditions.items():
        X = torch.tensor(np.stack(windows, axis=0), dtype=torch.float32)
        from torch.utils.data import TensorDataset, DataLoader
        ds = TensorDataset(X)
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        energies = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device)
                z, _, _, _ = model(xb)
                e = model.energy(z, xb).cpu().numpy()
                energies.append(e)
        energy_all = np.concatenate(energies)
        scores = energy_only_score(energy_all)
        m = evaluate_synthetic_zero_shot(scores, all_labels)
        results[name] = m
        logger.info(f"  [{map_mode}/{name}] AUROC={m['auroc']:.4f}  "
                    f"AUPR={m['aupr']:.4f}")

    # Interpretation
    orig_auc = results['original']['auroc']
    shuf_auc = results['shuffled']['auroc']
    noise_auc = results['noise']['auroc']

    if np.isnan(orig_auc):
        interpretation = "UNDEFINED (nan)"
    elif orig_auc - max(shuf_auc, noise_auc) > 0.10:
        interpretation = "STRUCTURAL — original much better than shuffled/noise"
    elif max(shuf_auc, noise_auc) > 0.60:
        interpretation = "PADDING_ARTIFACT_SUSPECTED — shuffled/noise also discriminates"
    else:
        interpretation = "AMBIGUOUS — original only modestly better than shuffled/noise"

    logger.info(f"  [{map_mode}] Interpretation: {interpretation}")

    return {
        'map_mode': map_mode,
        'adapter_used': False,
        'training_performed': False,
        'n_windows_each': n_each,
        'results': {k: v for k, v in results.items()},
        'interpretation': interpretation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 6.2 — LATENT GEOMETRY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_latent_geometry(model, ecg_dataset, network_dataset,
                             map_mode, save_dir):
    """
    Compares latent geometry between ECG and network domains.
    network_dataset contains all regimes; stable (label==0) and anomalous
    (label!=0) are split post-hoc from the same forward pass.
    """
    from torch.utils.data import DataLoader

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    device = next(model.parameters()).device

    def extract_z(ds, n_max=2000):
        loader = DataLoader(ds, batch_size=256, shuffle=False, drop_last=False)
        zs, labs = [], []
        count = 0
        for batch in loader:
            X, L, _ = batch
            with torch.no_grad():
                z, _, _, _ = model(X.to(device))
            zs.append(z.cpu().numpy())
            labs.append(L.numpy())
            count += len(X)
            if count >= n_max:
                break
        return np.concatenate(zs)[:n_max], np.concatenate(labs)[:n_max]

    logger.info(f"  [geometry/{map_mode}] Extracting ECG latents...")
    z_ecg, l_ecg = extract_z(ecg_dataset)
    z_net, l_net = extract_z(network_dataset, n_max=4000)
    mask_stable = l_net == 0
    mask_anon   = l_net != 0
    z_stable = z_net[mask_stable]
    z_anon   = z_net[mask_anon]

    if len(z_stable) == 0:
        logger.warning("No stable-regime windows found — skipping geometry comparison.")
        return {}
    if len(z_anon) == 0:
        logger.warning("No anomalous-regime windows found — skipping geometry comparison.")
        return {}

    # ECG normal centroid (label=0)
    mask_ecg_norm = l_ecg == 0
    if mask_ecg_norm.sum() == 0:
        logger.warning("No ECG normal samples — skipping geometry comparison.")
        return {}

    c_ecg_normal = z_ecg[mask_ecg_norm].mean(axis=0)
    c_net_stable = z_stable.mean(axis=0)
    c_net_anon   = z_anon.mean(axis=0)

    def cosine_sim(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return float('nan')
        return float(np.dot(a, b) / (na * nb))

    result = {
        'map_mode': map_mode,
        'cosine_sim_ecg_normal_vs_net_stable': cosine_sim(c_ecg_normal, c_net_stable),
        'l2_ecg_normal_vs_net_stable': float(np.linalg.norm(c_ecg_normal - c_net_stable)),
        'l2_net_stable_vs_net_anon': float(np.linalg.norm(c_net_stable - c_net_anon)),
        'var_ecg_normal': float(z_ecg[mask_ecg_norm].var(axis=0).mean()),
        'var_net_stable': float(z_stable.var(axis=0).mean()),
        'var_net_anomalous': float(z_anon.var(axis=0).mean()),
    }
    logger.info(f"  [geometry/{map_mode}] cos(ECG_norm, net_stable)="
                f"{result['cosine_sim_ecg_normal_vs_net_stable']:.4f}  "
                f"L2={result['l2_ecg_normal_vs_net_stable']:.4f}")

    # 2D PCA projection
    try:
        from sklearn.decomposition import PCA
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        all_z = np.concatenate([z_ecg[:500], z_stable[:500], z_anon[:500]])
        pca = PCA(n_components=2, random_state=42)
        all_2d = pca.fit_transform(all_z)

        n_ecg = min(500, len(z_ecg))
        n_stable = min(500, len(z_stable))
        n_anon = min(500, len(z_anon))
        z_ecg_2d = all_2d[:n_ecg]
        z_stable_2d = all_2d[n_ecg:n_ecg + n_stable]
        z_anon_2d = all_2d[n_ecg + n_stable:n_ecg + n_stable + n_anon]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(*z_ecg_2d.T, s=5, alpha=0.4, color='steelblue', label='ECG normal')
        ax.scatter(*z_stable_2d.T, s=5, alpha=0.4, color='mediumseagreen', label='Net stable')
        ax.scatter(*z_anon_2d.T, s=5, alpha=0.4, color='tomato', label='Net anomalous')
        # Centroids
        for c2d, label, col in [
            (pca.transform(c_ecg_normal[None])[0], 'ECG-normal-C', 'navy'),
            (pca.transform(c_net_stable[None])[0], 'Net-stable-C', 'darkgreen'),
            (pca.transform(c_net_anon[None])[0], 'Net-anon-C', 'darkred'),
        ]:
            ax.scatter(*c2d, s=120, marker='*', color=col, zorder=5, label=label)

        ax.set_title(f'Latent Geometry — ECG vs Network ({map_mode}) — PCA')
        ax.legend(fontsize=8, markerscale=2)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'zeroshot_latent_comparison_{map_mode}.png')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=100)
        plt.close()
        logger.info(f"  Saved {save_path}")
    except Exception as e:
        logger.warning(f"PCA plot failed: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ecg.yaml')
    parser.add_argument('--checkpoint', default='experiments/checkpoints/best_model.pt')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SpectralEmergeModel(cfg).to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                         weights_only=True))
        logger.info(f"Loaded ECG checkpoint: {args.checkpoint}")
    else:
        logger.warning("No checkpoint — using random init.")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    pings, labels, transitions = generate_network_stream(n_pings=100000, seed=42)

    os.makedirs('experiments/results', exist_ok=True)
    fig_dir = 'experiments/results/figures'

    all_ablation = []
    all_geometry = []

    for map_mode in MAP_MODES:
        logger.info(f"\n{'='*60}\n[ABLATION] map_mode={map_mode}")
        ab = ablation_padding_vs_structure(model, map_mode, n_windows=1000)
        all_ablation.append(ab)

        logger.info(f"\n[GEOMETRY] map_mode={map_mode}")
        # Use a single dataset with original labels; geometry fn will split by label
        ds_all = ZeroShotNetworkDataset(pings, labels, window=50, stride=5,
                                        map_mode=map_mode, normalize=True)

        # ECG dataset for geometry comparison (optional)
        try:
            from src.data.timeseries import OrderedBeatDataset
            ecg_ds = OrderedBeatDataset('100', cfg)
        except Exception as e:
            logger.warning(f"ECG dataset for geometry unavailable: {e} — using network ds as proxy")
            ecg_ds = ds_all

        geo = compare_latent_geometry(model, ecg_ds, ds_all,
                                      map_mode, fig_dir)
        all_geometry.append(geo)

    # Save ablation JSON
    ablation_output = {
        'adapter_used': False,
        'training_performed': False,
        'ablation_results': all_ablation,
        'geometry_results': all_geometry,
    }
    for map_mode in MAP_MODES:
        ab = next(a for a in all_ablation if a['map_mode'] == map_mode)
        ab_path = f'experiments/results/zeroshot_ablation_{map_mode}.json'
        with open(ab_path, 'w', encoding='utf-8') as f:
            json.dump(ab, f, indent=2, default=str)
        logger.info(f"Saved {ab_path}")

    with open('experiments/results/zeroshot_ablation_combined.json', 'w',
              encoding='utf-8') as f:
        json.dump(ablation_output, f, indent=2, default=str)
    logger.info("Saved experiments/results/zeroshot_ablation_combined.json")

    # Print ablation summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION SUMMARY")
    for ab in all_ablation:
        orig = ab['results']['original']['auroc']
        shuf = ab['results']['shuffled']['auroc']
        noise = ab['results']['noise']['auroc']
        logger.info(f"  {ab['map_mode']:15s}  orig={orig:.4f}  "
                    f"shuffled={shuf:.4f}  noise={noise:.4f}  "
                    f"→ {ab['interpretation']}")
    logger.info("=" * 60)
    logger.info("Phase 5 ablation complete.")


if __name__ == '__main__':
    main()
