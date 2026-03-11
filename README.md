# Spectral Emerge

*[Read in Portuguese / Leia em Português](README.pt-BR.md)*

**Independent experimental exploration of implicit, energy-based representations for anomaly detection in continuous signals.**

---

## What this project is

This project explores whether useful discrete structure may emerge from
continuous latent dynamics — specifically, whether an implicit equilibrium-based
neural model can learn internal states from continuous signals, and whether
its energy function acts as a useful anomaly signal without explicit supervision.

I am not a physicist or ML theorist. This started as a practical exploration
of a conceptual hypothesis. 

> **Scientific Inspiration:** This project was originally inspired, at a conceptual level, by the paper *"Emergent quantization from a dynamic vacuum"* (White et al., Physical Review Research, 2026), which explores the possibility of discrete structure emerging from continuous dynamics. The experiments in this repository were motivated by that general intuition but **do not** constitute a validation of the physical theory proposed in the article.


The answer so far:

- **Yes, partially** — energy-based anomaly detection works on real ECG data
- **No** — strong zero-shot cross-domain transfer from ECG to network latency was not demonstrated
- **Unclear / not demonstrated** — robust regime discovery; universal emergent-state structure

This repository is shared openly so others can inspect, reuse, critique, or extend the experiments.

---

## Main results

### 1. Real ECG anomaly detection (positive)

Using PhysioNet MIT-BIH ECG data, the model's energy score produced meaningful anomaly
signal without using labels at inference time.

| Score | Mean AUROC |
|-------|-----------|
| `energy_only` (primary, unsupervised) | **0.801** |
| `centroid_distance` (secondary, unsupervised) | 0.627 |

Protocol: leave-one-record-out over 10 records. No labels used at inference.

### 2. Temporal evaluation (mixed)

Ordered beat-level analysis over ECG records:
- Anomaly detection signal confirmed (AUROC ≈ 0.80)
- Regime-transition detection: F1 < 0.30 — below target, weak result
- DEQ iteration hypothesis: not consistently supported

### 3. Zero-shot cross-domain transfer (negative)

The ECG model was tested directly on synthetic network latency data with no adaptation:

| Mapping | Synthetic AUROC |
|---------|----------------|
| constant_pad | 0.519 |
| reflect_pad | 0.510 |
| linear_resize | 0.513 |

Verdict: **NO** — zero-shot transfer not supported. AUROC ≈ random chance across all mappings.
Ablation confirmed no structural artefact from padding.

---

## Honest conclusion

This project does **not** support:
- Universal emergent representations across signal domains
- Validated physical theory of emergence
- Zero-shot cross-domain anomaly detection

This project **does** support:
- Energy-based unsupervised anomaly detection working on real ECG
- Open, reproducible experiment history including negative results

---

## Repository structure

```text
src/                        core model and evaluation code
configs/                    experiment configuration files
experiments/                experiment scripts
experiments/results/        saved JSON metrics, summaries, figures
tests/                      software-behaviour tests (13 tests)
docs/                       project notes, results index, reproducibility guide
paper/                      short report
```

## Recommended reading order

1. [`docs/project_note.md`](docs/project_note.md) — narrative background
2. [`paper/short-report.md`](paper/short-report.md) — structured results
3. [`experiments/results/temporal_summary.md`](experiments/results/temporal_summary.md) — ECG result
4. [`experiments/results/zeroshot_network_summary.md`](experiments/results/zeroshot_network_summary.md) — zero-shot negative result
5. [`docs/results_index.md`](docs/results_index.md) — figure and artefact index

---

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest tests/ -v

# Phase 4 — ECG temporal evaluation
python experiments/temporal_eval.py --config configs/ecg.yaml

# Phase 5 — Zero-shot cross-domain
python experiments/zeroshot_eval.py
python experiments/zeroshot_ablation.py
```

See [`docs/reproducibility.md`](docs/reproducibility.md) for full notes including caveats.

---

## What this project is good for

Realistic current scope:

- **Unsupervised anomaly detection in continuous signals** (ECG, physiological monitoring, sensor streams)
- Settings where labelled anomalies are scarce
- Research prototyping with implicit/DEQ-style architectures

---

## License

MIT — see [`LICENSE`](LICENSE)

## Citation

If you reuse ideas, code, or results, please cite this repository.
See [`CITATION.cff`](CITATION.cff).

---

*This repository documents an independent experiment. It is not a polished research package. Negative results are kept and documented.*