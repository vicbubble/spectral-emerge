"""
experiments/temporal_eval.py — Phase 4 Main Evaluation Script

Evaluates temporal signal from PhysioNet ECG beats processed independently.
Uses leave-one-record-out or explicit train/eval split (per cfg).

Usage:
    python experiments/temporal_eval.py --config configs/ecg.yaml [--checkpoint PATH]
"""
import os
import sys
import json
import yaml
import logging
import argparse
import warnings
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.full_model import SpectralEmergeModel
from src.eval.temporal_eval import (
    compute_patient_trajectory,
    compute_centroids,
    compute_normal_centroid,
    anomaly_score_unsupervised,
    anomaly_score_normal_distance,
    detect_regime_changes,
    get_true_transitions,
    evaluate_anomaly_detection,
    evaluate_regime_detection,
    test_iteration_hypothesis,
    compute_baselines,
)
from src.eval.visualize import (
    plot_patient_timeline,
    plot_auroc_curve,
    plot_latent_trajectory_2d,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

SCORE_TYPES = [
    'unsupervised_energy',
    'unsupervised_centroid_distance',
    'unsupervised_both',
    'label_informed_normal_distance',
]


def beats_random(n_beats_for_success):
    """Fraction of records where model beats a baseline."""
    if n_beats_for_success[0] == 0:
        return float('nan')
    return n_beats_for_success[1] / n_beats_for_success[0]


def load_model(cfg, checkpoint_path, device):
    model = SpectralEmergeModel(cfg).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                         weights_only=True))
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning("No checkpoint found — running with random weights.")
    model.eval()
    return model


def determine_split(cfg):
    """
    Returns (train_records, eval_records).
    Prefers explicit cfg.data.train_records / cfg.data.eval_records.
    Falls back to leave-one-record-out over cfg.data.records.
    """
    records = [str(r) for r in cfg['data']['records']]
    train_recs = cfg['data'].get('train_records')
    eval_recs = cfg['data'].get('eval_records')
    if train_recs and eval_recs:
        return [str(r) for r in train_recs], [str(r) for r in eval_recs]
    # Leave-one-record-out
    return None, records  # None → evaluated per-record with LORO


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ecg.yaml')
    parser.add_argument('--checkpoint', default='experiments/checkpoints/best_model.pt')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(cfg, args.checkpoint, device)

    train_records_fixed, eval_records = determine_split(cfg)
    use_loro = (train_records_fixed is None)
    all_records = [str(r) for r in cfg['data']['records']]
    logger.info(f"Protocol: {'leave-one-record-out' if use_loro else 'explicit split'}")
    logger.info(f"Eval records: {eval_records}")

    os.makedirs('experiments/results/figures', exist_ok=True)
    os.makedirs('experiments/results', exist_ok=True)

    per_record_results = []

    # Track baseline beat counts: (total_eval, beats_model_wins)
    beats_vs_random    = {st: [0, 0] for st in SCORE_TYPES}
    beats_vs_constant  = {st: [0, 0] for st in SCORE_TYPES}
    beats_vs_raw_energy = {st: [0, 0] for st in SCORE_TYPES}

    for eval_rec in eval_records:
        logger.info(f"\n{'='*60}\nEvaluating record {eval_rec}")

        # Leakage-free source records
        if use_loro:
            source_records = [r for r in all_records if r != eval_rec]
        else:
            source_records = train_records_fixed

        if not source_records:
            logger.warning(f"No source records for {eval_rec} — skipping.")
            continue

        # Centroids from source only
        try:
            centroids = compute_centroids(model, source_records, cfg, device)
        except Exception as e:
            logger.error(f"compute_centroids failed for record {eval_rec}: {e}")
            continue

        try:
            normal_centroid = compute_normal_centroid(model, source_records, cfg, device)
        except Exception as e:
            logger.error(f"compute_normal_centroid failed: {e}")
            normal_centroid = None

        # Patient trajectory on eval record
        traj = compute_patient_trajectory(model, eval_rec, cfg, device)
        if traj is None:
            logger.warning(f"Skipping record {eval_rec}: empty trajectory.")
            continue

        z = traj['z_star_seq']
        E = traj['energy_seq']
        L = traj['labels_seq']
        iters = traj['iter_seq']

        n_beats = traj['T']
        n_normal = int((L == 0).sum())
        n_anomalous = int((L != 0).sum())
        anomaly_prevalence = n_anomalous / max(n_beats, 1)

        logger.info(f"Record {eval_rec}: {n_beats} beats, "
                    f"{n_normal} normal, {n_anomalous} anomalous "
                    f"(prevalence={anomaly_prevalence:.2%})")

        # ── Anomaly Scores ────────────────────────────────────────────────
        scores = {
            'unsupervised_energy':
                anomaly_score_unsupervised(E, z, centroids, method='energy'),
            'unsupervised_centroid_distance':
                anomaly_score_unsupervised(E, z, centroids, method='centroid_distance'),
            'unsupervised_both':
                anomaly_score_unsupervised(E, z, centroids, method='both'),
            'label_informed_normal_distance':
                anomaly_score_normal_distance(z, normal_centroid)
                if normal_centroid is not None
                else np.zeros(n_beats),
        }

        # ── Detection metrics ─────────────────────────────────────────────
        anomaly_metrics = {}
        for st, sc in scores.items():
            m = evaluate_anomaly_detection(sc, L)
            anomaly_metrics[st] = m
            logger.info(f"  [{st}] AUROC={m['auroc']:.4f}  AUPR={m['aupr']:.4f}  "
                        f"F1opt={m['f1_opt']:.4f}")

        # ── Regime detection ──────────────────────────────────────────────
        change_points, change_scores = detect_regime_changes(z, window=5, threshold=0.5)
        true_transitions = get_true_transitions(L, min_duration=5)
        regime_m = evaluate_regime_detection(change_points, true_transitions, tolerance=3)
        logger.info(f"  [regime] F1={regime_m['f1']:.4f}  "
                    f"delay={regime_m['mean_delay']}  "
                    f"detected={regime_m['n_detected']}/{regime_m['n_true']}")

        # ── DEQ iteration hypothesis ──────────────────────────────────────
        iter_hyp = test_iteration_hypothesis(iters, L)
        logger.info(f"  [iter_hyp] supported={iter_hyp['hypothesis_supported']}  "
                    f"p={iter_hyp['p_value']}")

        # ── Baselines ─────────────────────────────────────────────────────
        baselines = compute_baselines(E, L)
        logger.info(f"  [baselines] random={baselines['random_auroc']:.4f}  "
                    f"raw_energy={baselines['raw_energy_auroc']:.4f}")

        # Track model vs baselines
        for st, m in anomaly_metrics.items():
            if not np.isnan(m['auroc']):
                beats_vs_random[st][0] += 1
                if m['auroc'] > baselines['random_auroc']:
                    beats_vs_random[st][1] += 1
                beats_vs_constant[st][0] += 1
                if m['auroc'] > 0.5:
                    beats_vs_constant[st][1] += 1
                beats_vs_raw_energy[st][0] += 1
                if m['auroc'] > baselines['raw_energy_auroc']:
                    beats_vs_raw_energy[st][1] += 1

        # ── Figures ────────────────────────────────────────────────────────
        primary_score = scores['unsupervised_both']
        plot_patient_timeline(
            labels_seq=L, anomaly_scores=primary_score, z_star_seq=z,
            iter_seq=iters, true_transitions=true_transitions,
            change_points=change_points, record_id=eval_rec,
            save_path=f'experiments/results/figures/timeline_{eval_rec}.png',
        )
        for st, sc in scores.items():
            plot_auroc_curve(
                labels=L, scores=sc, record_id=eval_rec, score_type=st,
                save_path=f'experiments/results/figures/roc_{eval_rec}_{st}.png',
            )
        plot_latent_trajectory_2d(
            z_star_seq=z, labels_seq=L, change_points=change_points,
            record_id=eval_rec,
            save_path=f'experiments/results/figures/trajectory_{eval_rec}.png',
        )

        per_record_results.append({
            'record_id': eval_rec,
            'n_beats': n_beats,
            'n_normal': n_normal,
            'n_anomalous': n_anomalous,
            'anomaly_prevalence': anomaly_prevalence,
            'anomaly_metrics': anomaly_metrics,
            'regime_metrics': regime_m,
            'iteration_hypothesis': iter_hyp,
            'baselines': baselines,
        })

    # ── Aggregate ────────────────────────────────────────────────────────────
    if not per_record_results:
        logger.error("No records evaluated — check wfdb install and record config.")
        return

    aggregate = {}
    for st in SCORE_TYPES:
        vals = [r['anomaly_metrics'][st]['auroc']
                for r in per_record_results
                if not np.isnan(r['anomaly_metrics'][st]['auroc'])]
        if vals:
            aggregate[st] = {
                'auroc_mean': float(np.mean(vals)),
                'auroc_std': float(np.std(vals)),
                'beats_random_pct': beats_random(beats_vs_random[st]),
                'beats_constant_pct': beats_random(beats_vs_constant[st]),
                'beats_raw_energy_pct': beats_random(beats_vs_raw_energy[st]),
            }

    regime_f1s = [r['regime_metrics']['f1']
                  for r in per_record_results
                  if not np.isnan(r['regime_metrics']['f1'])]
    iter_supported = [r['iteration_hypothesis']['hypothesis_supported']
                      for r in per_record_results]

    aggregate['regime'] = {
        'f1_mean': float(np.mean(regime_f1s)) if regime_f1s else float('nan'),
        'f1_std': float(np.std(regime_f1s)) if regime_f1s else float('nan'),
    }
    aggregate['iter_hypothesis_support_rate'] = (
        sum(iter_supported) / len(iter_supported) if iter_supported else float('nan')
    )

    # ── Success criteria check ───────────────────────────────────────────────
    criteria = {}
    for st in ['unsupervised_energy', 'unsupervised_centroid_distance', 'unsupervised_both']:
        if st in aggregate:
            m = aggregate[st]['auroc_mean']
            criteria[f'min_auroc_0.60_{st}'] = bool(m > 0.60)
            criteria[f'strong_auroc_0.75_{st}'] = bool(m > 0.75)
    criteria['beats_random_all_records'] = all(
        beats_vs_random[st][1] == beats_vs_random[st][0]
        for st in SCORE_TYPES if beats_vs_random[st][0] > 0
    )
    criteria['regime_f1_min'] = bool(aggregate['regime']['f1_mean'] > 0.30) \
        if not np.isnan(aggregate['regime']['f1_mean']) else False
    criteria['iter_hyp_majority'] = bool(aggregate['iter_hypothesis_support_rate'] > 0.50) \
        if not np.isnan(aggregate['iter_hypothesis_support_rate']) else False

    # Check if retrain needed
    unsupervised_aucs = [aggregate.get(st, {}).get('auroc_mean', 0.0)
                         for st in ['unsupervised_energy', 'unsupervised_centroid_distance', 'unsupervised_both']]
    if all(a < 0.60 for a in unsupervised_aucs if not np.isnan(a)):
        logger.warning(
            "All fully unsupervised scores AUROC < 0.60. "
            "Consider retraining with lambda_spectral=0.02 and comparing both runs."
        )

    # ── Save JSON ────────────────────────────────────────────────────────────
    output = {
        'per_record': per_record_results,
        'aggregate': aggregate,
        'success_criteria': criteria,
    }
    with open('experiments/results/temporal_metrics.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Saved experiments/results/temporal_metrics.json")

    # ── Write Markdown Summary ───────────────────────────────────────────────
    with open('experiments/results/temporal_summary.md', 'w', encoding='utf-8') as f:
        f.write("# Phase 4 Temporal Evaluation Summary\n\n")
        f.write(f"**Records evaluated**: {len(per_record_results)}\n\n")

        f.write("## Per-Record Overview\n\n")
        f.write("| Record | Beats | Normal | Anomalous | Prevalence |\n")
        f.write("|--------|-------|--------|-----------|------------|\n")
        for r in per_record_results:
            f.write(f"| {r['record_id']} | {r['n_beats']} | {r['n_normal']} | "
                    f"{r['n_anomalous']} | {r['anomaly_prevalence']:.2%} |\n")

        f.write("\n## AUROC (Anomaly Detection)\n\n")
        f.write("| Score Type | Unsupervised? | Mean AUROC ± std |\n")
        f.write("|------------|---------------|------------------|\n")
        for st in SCORE_TYPES:
            if st in aggregate:
                unsup = "[YES]" if "label_informed" not in st else "[NO - calibrated]"
                f.write(f"| {st} | {unsup} | {aggregate[st]['auroc_mean']:.4f} "
                        f"± {aggregate[st]['auroc_std']:.4f} |\n")

        f.write("\n## Baseline Comparison\n\n")
        for st in SCORE_TYPES:
            if st in aggregate:
                ag = aggregate[st]
                f.write(f"**{st}**\n")
                f.write(f"- Beats random: {ag['beats_random_pct']:.0%} of records\n")
                f.write(f"- Beats constant (>0.5): {ag['beats_constant_pct']:.0%} of records\n")
                f.write(f"- Beats raw energy: {ag['beats_raw_energy_pct']:.0%} of records\n\n")

        f.write("## Regime Detection\n\n")
        f.write(f"- Mean F1: {aggregate['regime']['f1_mean']:.4f} "
                f"± {aggregate['regime']['f1_std']:.4f}\n\n")

        f.write("## DEQ Iteration Hypothesis\n\n")
        f.write(f"- Supported on: {aggregate['iter_hypothesis_support_rate']:.0%} of records\n\n")

        f.write("## Success Criteria\n\n")
        for k, v in criteria.items():
            f.write(f"- {'[PASS]' if v else '[FAIL]'} {k}\n")

    logger.info("Saved experiments/results/temporal_summary.md")
    logger.info("Phase 4 evaluation complete.")


if __name__ == '__main__':
    main()
