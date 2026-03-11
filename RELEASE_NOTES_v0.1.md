# Release Notes — v0.1.0

**First public release**  
Date: 2026-03-11

---

## What this release contains

This is the first public release of `spectral-emerge`, an independent
experimental exploration of energy-based anomaly detection using implicit
latent dynamics.

### Included in v0.1.0

- Full source code for Phases 1–5
- Trained ECG model checkpoint (`experiments/checkpoints/best_model.pt`)
  — commit explicitly if pushing; excluded from `.gitignore` by default due to size
- Result summaries:
  - `experiments/results/temporal_summary.md`
  - `experiments/results/zeroshot_network_summary.md`
- Metrics JSON:
  - `experiments/results/temporal_metrics.json`
  - `experiments/results/zeroshot_network_metrics.json`
  - `experiments/results/zeroshot_ablation_combined.json`
- Figures in `experiments/results/figures/`
- Test suite: 13 software-behaviour tests
- Documentation: `docs/`, `paper/`

---

## Key results (honest summary)

### ECG anomaly detection — POSITIVE RESULT

The energy score from the implicit fixed-point model achieved:

- Mean AUROC **0.801** across 10 PhysioNet MIT-BIH records
- Protocol: leave-one-record-out, no labels at inference time
- Primary score: `energy_only` (no target labels, no calibration)

This is the main practical result from this project.

### Temporal evaluation — MIXED RESULT

- Anomaly detection signal confirmed (AUROC ≈ 0.80)
- Regime-transition detection F1 < 0.30 (below target — weak result)
- DEQ iteration hypothesis not consistently supported

### Zero-shot cross-domain transfer — NEGATIVE RESULT

The ECG model was tested directly on synthetic network latency data:

- No retraining, no adapter, no threshold tuning
- Three input mappings tested: constant-pad, reflect-pad, linear-resize
- All mappings: AUROC ≈ 0.51 (random level)
- Ablation confirmed: no structural signal transferred

**Verdict: NO** — universal zero-shot cross-domain transfer is not supported.

---

## Honest scope statement

This release does **not** claim:

- Universal emergent representations across signal domains
- A validated physical theory of emergence
- A new general-purpose anomaly detection product

This release **does** contain:

- A working energy-based ECG anomaly detector
- A documented, reproducible negative result in zero-shot transfer
- Open code, configs, and results for inspection and reuse

---

## Known limitations

- Model checkpoint is large and excluded from git by default
- PhysioNet data requires separate download (auto via `wfdb`)
- RIPE Atlas real-data path is optional and may be unavailable
- The project was developed and tested on Windows / Python 3.13

---

## How to reproduce

See `docs/reproducibility.md` for full instructions.

Quick start:

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
python experiments/temporal_eval.py --config configs/ecg.yaml
python experiments/zeroshot_eval.py
```
