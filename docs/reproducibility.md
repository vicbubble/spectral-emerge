# Reproducibility Notes

*[Read in Portuguese / Leia em Português](reproducibility.pt-BR.md)*

This document describes how to reproduce the key experiments in this repository.
It is intentionally honest about what is straightforward, what requires setup,
and what has known caveats.

---

## Environment

**Python version tested:** 3.13.x (Windows)

Dependencies are listed in `requirements.txt`.

Install with:

```bash
pip install -r requirements.txt
```

> **Note:** `torchdeq` may require a specific install path depending on your
> PyTorch version. If it fails to install from PyPI, try:
> ```bash
> pip install git+https://github.com/locuslab/torchdeq
> ```

---

## Running the test suite

```bash
cd spectral_emerge
python -m pytest tests/ -v
```

The test suite covers software-behaviour invariants.
It does **not** assert performance thresholds.

**Known test categories:**

| Test file | What it checks |
|-----------|---------------|
| `tests/test_temporal.py` | OrderedBeatDataset order, score range, caching |
| `tests/test_zeroshot.py` | Model frozen check, mapping shape, no-optimizer invariant |

All 13 tests should pass on a clean install.

---

## Data download

ECG data is downloaded automatically from PhysioNet on first run via `wfdb`:

```bash
python experiments/temporal_eval.py --config configs/ecg.yaml
```

Required internet access. Data is cached locally in `physionet_data/`.

> **Note:** `physionet_data/` is excluded from version control by `.gitignore`
> because of size. You will need to re-download it.

RIPE Atlas data download is optional and may fail gracefully if the
measurement ID or network is unavailable. The script continues without it.

---

## Reproducing Phase 4 — ECG temporal evaluation

```bash
python experiments/temporal_eval.py --config configs/ecg.yaml
```

This will:
1. Load the ECG checkpoint from `experiments/checkpoints/best_model.pt`
2. Download / cache PhysioNet records 100–109
3. Run leave-one-record-out temporal evaluation
4. Save results to `experiments/results/temporal_metrics.json` and `temporal_summary.md`

**Checkpoint availability:** the model checkpoint is excluded from version
control by default (checkpoints are large). If not present, the model will
run with random initialisation, producing meaningless results.

To commit the checkpoint explicitly:
```bash
git add -f experiments/checkpoints/best_model.pt
```

---

## Reproducing Phase 5 — Zero-shot evaluation

```bash
python experiments/zeroshot_eval.py
python experiments/zeroshot_ablation.py
```

These scripts:
- Do not require the ECG data (synthetic data is generated in-process)
- Do require the ECG checkpoint for meaningful results
- Run three deterministic input mappings

**Expected outcome (with checkpoint):**
- Synthetic energy AUROC ≈ 0.51–0.52 across all three mappings
- Verdict: NO (zero-shot transfer not supported)

This is a negative result and is the expected and correct behaviour.

---

## Known caveats and limitations

| Caveat | Detail |
|--------|--------|
| ECG checkpoint required | Without the checkpoint, all model outputs are meaningless |
| PhysioNet download | Requires internet access and may be slow |
| wfdb encoding issue | On some systems the raw `requirements.txt` had a corrupt `wfdb` line — fixed in v0.1 |
| Positional zero-shot claim | Zero-shot transfer from ECG to network latency was **tested and failed** |
| Regime detection | Temporal F1 for regime detection was weak across all tested records |
| Real RIPE Atlas data | API availability is not guaranteed; the script gracefully skips if unavailable |

---

## Experiment history

| Phase | Script | Status |
|-------|--------|--------|
| 1–2 | `experiments/ablation.py` | Synthetic structure tests. Results stable. |
| 3 | `experiments/ecg_eval.py` | ECG evaluation. Positive result. Results saved. |
| 4 | `experiments/temporal_eval.py` | Temporal evaluation. Positive anomaly result; weak regime result. |
| 5 | `experiments/zeroshot_eval.py` | Zero-shot cross-domain. Negative result. Saved. |
| 5 | `experiments/zeroshot_ablation.py` | Padding-vs-structure ablation. AMBIGUOUS / no artifact. Saved. |
