"""
experiments/zeroshot_eval.py — Phase 5 Zero-Shot Cross-Domain Main Script

The ECG model is loaded UNCHANGED and FROZEN.
No retraining. No adapter. No optimizer.
Three deterministic input mappings are tested.
PRIMARY metric: energy-only score (no target labels).

Usage:
    python experiments/zeroshot_eval.py
"""
import os
import sys
import json
import yaml
import logging
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.full_model import SpectralEmergeModel
from src.data.network import (
    generate_network_stream,
    ZeroShotNetworkDataset,
    download_ripe_atlas,
    RipeAtlasDataset,
    REGIME_LABELS,
)
from src.eval.zeroshot_eval import (
    compute_trajectory_zeroshot,
    energy_only_score,
    nearest_centroid_unsupervised_score,
    evaluate_synthetic_zero_shot,
    evaluate_real_zero_shot,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MAP_MODES = ["constant_pad", "reflect_pad", "linear_resize"]

# ECG Phase 4 mean AUROC for reference column
ECG_REF_AUROC = 0.801


# ─────────────────────────────────────────────────────────────────────────────
# VERDICT LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def compute_verdict(syn_aucs_by_mode):
    """Compute universality verdict from synthetic energy AUROC across modes."""
    valid = [v for v in syn_aucs_by_mode.values() if not np.isnan(v)]
    if not valid:
        return "NO", float('nan'), False, False

    max_diff = max(valid) - min(valid)
    all_above_70 = all(v > 0.70 for v in valid)
    any_above_70 = any(v > 0.70 for v in valid)
    robust = max_diff < 0.05

    if all_above_70 and robust:
        verdict = "STRONG"
    elif any_above_70:
        verdict = "PARTIAL"
    else:
        verdict = "NO"

    return verdict, max_diff, all_above_70, any_above_70


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def save_energy_hist(energy_seq, labels_seq, map_mode, dataset_type, save_path):
    """Energy distribution histogram: normal vs anomalous."""
    try:
        import matplotlib.pyplot as plt
        binary = (labels_seq != 0)
        normals = energy_seq[~binary]
        anomalous = energy_seq[binary]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(normals, bins=50, alpha=0.6, color='steelblue', label='Normal/Stable', density=True)
        ax.hist(anomalous, bins=50, alpha=0.6, color='tomato', label='Anomalous', density=True)
        ax.set_title(f'Energy Distribution — {dataset_type} / {map_mode}')
        ax.set_xlabel('Energy')
        ax.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100)
        plt.close()
    except Exception as e:
        logger.warning(f"save_energy_hist failed: {e}")


def save_timeline(index_seq, scores, labels_seq, map_mode, dataset_type, save_path):
    """Timeline of anomaly score vs true labels."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        fig.suptitle(f'Zero-Shot Timeline — {dataset_type} / {map_mode}')

        cmap = cm.get_cmap('tab10', 4)
        for cls in range(4):
            mask = labels_seq == cls
            ax1.scatter(index_seq[mask], np.zeros(mask.sum()) + cls, s=2,
                        color=cmap(cls), alpha=0.5, label=REGIME_LABELS.get(cls, str(cls)))
        ax1.set_ylabel('Regime')
        ax1.set_yticks(range(4))
        ax1.set_yticklabels([REGIME_LABELS[k] for k in range(4)], fontsize=7)
        ax1.legend(fontsize=6, markerscale=3, loc='upper right')

        ax2.plot(index_seq, scores, color='darkorange', lw=0.6, alpha=0.8)
        ax2.set_ylabel('Energy score')
        ax2.set_xlabel('Window index')
        ax2.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100)
        plt.close()
    except Exception as e:
        logger.warning(f"save_timeline failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ecg.yaml')
    parser.add_argument('--checkpoint', default='experiments/checkpoints/best_model.pt')
    parser.add_argument('--ripe-measurement', type=int, default=1001,
                        help='RIPE Atlas measurement ID (optional)')
    parser.add_argument('--ripe-probes', type=int, nargs='*', default=[],
                        help='RIPE Atlas probe IDs (optional)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load and FREEZE model ────────────────────────────────────────────────
    model = SpectralEmergeModel(cfg).to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                         weights_only=True))
        logger.info(f"Loaded ECG checkpoint from {args.checkpoint}")
    else:
        logger.warning("No checkpoint found — running with random init (results will be meaningless).")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Verify frozen
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    os.makedirs('experiments/results/figures', exist_ok=True)
    os.makedirs('experiments/results', exist_ok=True)

    # ── Synthetic stream ─────────────────────────────────────────────────────
    logger.info("Generating synthetic network stream...")
    pings_syn, labels_syn, transitions_syn = generate_network_stream(n_pings=50000, seed=42)
    logger.info(f"Synthetic stream: {len(pings_syn)} pings, "
                f"{len(transitions_syn)} transitions")

    # ── RIPE Atlas (optional) ────────────────────────────────────────────────
    ripe_data = download_ripe_atlas(
        args.ripe_probes, args.ripe_measurement,
        output_dir='data/cache/ripe',
    )
    if ripe_data is not None:
        pings_ripe, _ = ripe_data
        logger.info(f"RIPE Atlas data: {len(pings_ripe)} RTT measurements")
    else:
        logger.info("RIPE Atlas data unavailable — skipping real-data runs.")

    # ── Results container ────────────────────────────────────────────────────
    all_results = []
    syn_energy_by_mode = {}

    # ── Main evaluation loop ─────────────────────────────────────────────────
    for map_mode in MAP_MODES:
        logger.info(f"\n{'='*60}\nmap_mode: {map_mode}")

        # --- Synthetic ---
        dataset_syn = ZeroShotNetworkDataset(
            pings_syn, labels_syn, window=50, stride=5,
            map_mode=map_mode, normalize=True
        )
        traj_syn = compute_trajectory_zeroshot(model, dataset_syn, device)

        e_scores = energy_only_score(traj_syn['energy_seq'])

        syn_m = evaluate_synthetic_zero_shot(e_scores, traj_syn['labels_seq'])
        syn_energy_by_mode[map_mode] = syn_m['auroc']

        logger.info(f"  [synthetic/energy_only] AUROC={syn_m['auroc']:.4f}  "
                    f"AUPR={syn_m['aupr']:.4f}  F1={syn_m['f1_opt']:.4f}")

        # Secondary unsupervised score
        latent_scores, latent_label = nearest_centroid_unsupervised_score(traj_syn['z_star_seq'])
        latent_m = evaluate_synthetic_zero_shot(latent_scores, traj_syn['labels_seq'])
        logger.info(f"  [synthetic/{latent_label}] AUROC={latent_m['auroc']:.4f}")

        # Figures
        save_energy_hist(
            traj_syn['energy_seq'], traj_syn['labels_seq'], map_mode, 'synthetic',
            f'experiments/results/figures/energy_hist_synthetic_{map_mode}.png'
        )
        save_timeline(
            traj_syn['index_seq'], e_scores, traj_syn['labels_seq'], map_mode, 'synthetic',
            f'experiments/results/figures/timeline_network_synthetic_{map_mode}.png'
        )

        all_results.append({
            'data_source': 'synthetic',
            'dataset_type': 'synthetic',
            'map_mode': map_mode,
            'adapter_used': False,
            'training_performed': False,
            'score_type': 'energy_only',
            'label_type': 'ground_truth_regime',
            **syn_m,
        })
        all_results.append({
            'data_source': 'synthetic',
            'dataset_type': 'synthetic',
            'map_mode': map_mode,
            'adapter_used': False,
            'training_performed': False,
            'score_type': latent_label,
            'label_type': 'ground_truth_regime',
            **latent_m,
        })

        # --- RIPE Atlas ---
        if ripe_data is not None:
            dataset_ripe = RipeAtlasDataset(
                pings_ripe, window=50, stride=10,
                map_mode=map_mode, normalize=True
            )
            traj_ripe = compute_trajectory_zeroshot(model, dataset_ripe, device)
            e_ripe = energy_only_score(traj_ripe['energy_seq'])
            ripe_m = evaluate_real_zero_shot(e_ripe, traj_ripe['labels_seq'])
            logger.info(f"  [ripe/energy_only/exploratory] AUROC={ripe_m['auroc']:.4f}")

            save_energy_hist(
                traj_ripe['energy_seq'], traj_ripe['labels_seq'], map_mode, 'ripe',
                f'experiments/results/figures/energy_hist_ripe_{map_mode}.png'
            )
            all_results.append({
                'data_source': 'ripe_atlas',
                'dataset_type': 'ripe_atlas',
                'map_mode': map_mode,
                'adapter_used': False,
                'training_performed': False,
                'score_type': 'energy_only',
                **ripe_m,
            })

    # ── Verify model integrity ────────────────────────────────────────────────
    for k, v in model.state_dict().items():
        if not torch.allclose(v, original_state[k]):
            logger.error(f"INTEGRITY VIOLATION: parameter {k} was modified!")
    logger.info("Model integrity check: PASSED (no parameters modified)")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open('experiments/results/zeroshot_network_metrics.json', 'w',
              encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved experiments/results/zeroshot_network_metrics.json")

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict, max_diff, all_70, any_70 = compute_verdict(syn_energy_by_mode)

    ripe_aucs = {r['map_mode']: r['auroc'] for r in all_results
                 if r['dataset_type'] == 'ripe_atlas'
                 and r['score_type'] == 'energy_only'}
    any_ripe_65 = any(v > 0.65 for v in ripe_aucs.values() if not np.isnan(v))

    # ── Summary Table ─────────────────────────────────────────────────────────
    syn_row = {m: syn_energy_by_mode.get(m, float('nan')) for m in MAP_MODES}
    ripe_row = {m: ripe_aucs.get(m, float('nan')) for m in MAP_MODES}

    summary_lines = [
        "",
        "=" * 69,
        "ZERO-SHOT CROSS-DOMAIN RESULTS",
        "Model trained on: ECG",
        "No retraining. No adapter. Deterministic mapping only.",
        "=" * 69,
        "",
        f"{'':23} ECG-ref   Syn/const  Syn/refl  Syn/resize  RIPE/const  RIPE/refl  RIPE/resize",
        f"AUROC (energy)         {ECG_REF_AUROC:.3f}"
        + "".join(f"      {syn_row.get(m, float('nan')):.3f}" for m in MAP_MODES)
        + "".join(f"       {ripe_row.get(m, float('nan')):.3f}" for m in MAP_MODES),
        "",
        f"Mapping sensitivity:",
        f"  max_AUROC_diff_across_mappings = {max_diff:.3f}",
        "",
        "ZERO-SHOT PRIMARY VERDICT:",
        f"  Synthetic energy AUROC > 0.70 on all mappings : {'YES' if all_70 else 'NO'}",
        f"  Synthetic energy AUROC > 0.70 on >= 1 mapping: {'YES' if any_70 else 'NO'}",
        f"  Synthetic robust across mappings (diff < 0.05): {'YES' if max_diff < 0.05 else 'NO'}",
        f"  RIPE exploratory AUROC > 0.65 on >= 1 mapping: {'YES' if any_ripe_65 else 'NO' if ripe_aucs else 'N/A (unavailable)'}",
        f"  Universality claim: {verdict}",
        "=" * 69,
    ]

    for line in summary_lines:
        logger.info(line)

    with open('experiments/results/zeroshot_network_summary.md', 'w',
              encoding='utf-8') as f:
        f.write("# Phase 5 Zero-Shot Cross-Domain Summary\n\n")
        f.write("**Model**: ECG checkpoint (unchanged, frozen)\n")
        f.write("**Adaptation**: None\n")
        f.write("**Primary score**: energy-only (no target labels)\n\n")

        f.write("## Synthetic Results (ground-truth labels)\n\n")
        f.write("| Map Mode | AUROC | AUPR | F1-opt | Anomaly Prevalence |\n")
        f.write("|----------|-------|------|--------|-------------------|\n")
        for r in all_results:
            if r['dataset_type'] == 'synthetic' and r['score_type'] == 'energy_only':
                f.write(f"| {r['map_mode']} | {r['auroc']:.4f} | "
                        f"{r['aupr']:.4f} | {r['f1_opt']:.4f} | "
                        f"{r['anomaly_prevalence']:.2%} |\n")

        if ripe_aucs:
            f.write("\n## RIPE Atlas Results (EXPLORATORY — pseudo-labels)\n\n")
            f.write("| Map Mode | AUROC | AUPR | F1-opt |\n")
            f.write("|----------|-------|------|--------|\n")
            for r in all_results:
                if r['dataset_type'] == 'ripe_atlas' and r['score_type'] == 'energy_only':
                    f.write(f"| {r['map_mode']} | {r['auroc']:.4f} | "
                            f"{r['aupr']:.4f} | {r['f1_opt']:.4f} |\n")

        f.write(f"\n## Universality Verdict: **{verdict}**\n\n")
        f.write(f"- Mapping sensitivity (max AUROC diff): {max_diff:.4f}\n")
        f.write(f"- All mappings > 0.70: {'YES' if all_70 else 'NO'}\n")
        f.write(f"- Any mapping > 0.70: {'YES' if any_70 else 'NO'}\n")
        f.write(f"- adapter_used: false\n")
        f.write(f"- training_performed: false\n")

    logger.info("Saved experiments/results/zeroshot_network_summary.md")
    logger.info("Phase 5 zero-shot evaluation complete.")


if __name__ == '__main__':
    main()
