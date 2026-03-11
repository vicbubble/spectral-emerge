# Results Index

*[Read in Portuguese / Leia em Português](results_index.pt-BR.md)*

Quick index of the most important saved artefacts.
Intended to help a new visitor find the key results quickly.

---

## ECG Anomaly Detection

### Core summaries

| File | Description |
|------|-------------|
| `experiments/results/temporal_summary.md` | Human-readable summary of the temporal evaluation across 10 ECG records. Most important outcome document. |
| `experiments/results/temporal_metrics.json` | Full per-record, per-score-type metrics JSON. Ground truth for all reported numbers. |
| `experiments/results/ecg_summary_phase3.md` | Earlier ECG evaluation summary from Phase 3 (pre-temporal). |

### Why they matter

The temporal evaluation established the main positive result:
energy-based anomaly detection in real ECG above random-level performance,
without using labels at inference time.

---

### Selected ECG figures

| File | Description |
|------|-------------|
| `experiments/results/figures/timeline_100.png` | Beat-level anomaly score timeline for record 100. Typical "mostly normal with occasional anomalies" pattern. |
| `experiments/results/figures/timeline_102.png` | Record 102 with high anomaly prevalence. Shows energy score response across the record. |
| `experiments/results/figures/roc_100_unsupervised_energy.png` | ROC curve for the energy score on record 100. Representative of the best-performing unsupervised score. |
| `experiments/results/figures/ecg_latent_tsne_phase3.png` | t-SNE / latent projection from Phase 3. Shows how the ECG latent space clusters — not perfectly separated, but structured. |

**Why selected:** these figures tell the ECG anomaly story concisely — one typical record, one high-prevalence record, one ROC curve, and one latent geometry view.

---

## Zero-Shot Cross-Domain (Negative Result)

### Core summaries

| File | Description |
|------|-------------|
| `experiments/results/zeroshot_network_summary.md` | Human-readable summary of the zero-shot evaluation. Includes the verdict table and final universality claim. |
| `experiments/results/zeroshot_network_metrics.json` | Full per-mapping, per-dataset metrics. All three input mappings tested. |
| `experiments/results/zeroshot_ablation_combined.json` | Ablation results: original vs. shuffled vs. noise windows. Distinguishes structural transfer from padding artefact. |

### Selected zero-shot figures

| File | Description |
|------|-------------|
| `experiments/results/figures/zeroshot_latent_comparison_reflect_pad.png` | PCA of ECG latent space vs. stable-network vs. anomalous-network z* vectors. Shows domain gap visually — ECG and network latent points are clearly separated. |
| `experiments/results/figures/timeline_network_synthetic_reflect_pad.png` | Energy score timeline over synthetic network stream. Score is flat and uninformative — illustrates the failure. |
| `experiments/results/figures/energy_hist_synthetic_reflect_pad.png` | Energy distribution comparison: normal vs. anomalous network windows. Distributions largely overlap — confirms zero-shot failure. |

**Why selected:** the latent geometry figure shows *why* transfer fails (domain gap), and the timeline/histogram pair shows *what* the failure looks like empirically.

---

## Phase 1–2 Synthetic Experiments

| File | Description |
|------|-------------|
| `experiments/results/figures/fixedpoint_landscape.png` | 2D fixed-point energy landscape. Single dominant attractor visible — motivates the collapse analysis. |
| `experiments/results/figures/jacobian_spectrum_comparison.png` | Jacobian spectral analysis comparing DEQ vs. feedforward. |
| `experiments/results/collapse_sweep.json` | Collapse confirmation metrics across configurations. |
