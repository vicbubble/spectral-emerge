# Publish Checklist

Use this before running `git add / commit / push`.

---

## Documentation

- [x] `README.md` — honest summary of what the project is, what worked, what failed
- [x] `docs/project_note.md` — narrative project explanation
- [x] `docs/results_index.md` — index of key artefacts with descriptions
- [x] `docs/reproducibility.md` — how to reproduce experiments, known caveats
- [x] `paper/short-report.md` — short report-style write-up

## Metadata

- [x] `LICENSE` — MIT license present
- [x] `CITATION.cff` — minimal citation file present
- [x] `requirements.txt` — dependencies listed, encoding bug fixed
- [x] `pytest.ini` — test discovery configured
- [x] `RELEASE_NOTES_v0.1.md` — first release notes present

## Scientific integrity

- [x] Negative zero-shot result **included**, not hidden
- [x] Positive ECG result reported with correct protocol (leave-one-record-out)
- [x] No claims of universal cross-domain transfer remain
- [x] Unsupervised and calibrated scores reported separately
- [x] Ablation included (structure vs. padding artefact)
- [x] Ablation result (AMBIGUOUS) reported as-is, not exaggerated either way

## Repository hygiene

- [x] `.gitignore` reviewed:
  - `__pycache__/`, `.pytest_cache/` — excluded
  - `error.log`, `pytest.log`, `pytest_err.log` — excluded
  - `physionet_data/` (large data cache) — excluded
  - `experiments/checkpoints/` — excluded by default (too large)
  - `centroids_*.npy`, `normal_centroid_*.npy` — excluded (reproducible)
  - Result JSONs and figures — **tracked** (intentionally kept)
- [ ] `experiments/checkpoints/best_model.pt` — decide: commit explicitly or document as manual download
  - To commit: `git add -f experiments/checkpoints/best_model.pt`
- [ ] Large `.npy` / fixture files in `physionet_data/` — NOT committed (excluded)

## Tests

- [x] `tests/test_temporal.py` — 7 tests, all pass
- [x] `tests/test_zeroshot.py` — 6 tests, all pass
- [ ] Run one final `python -m pytest tests/ -v` before push to confirm clean state

## Final manual steps (do NOT run automatically)

```bash
# 1. Check what will be committed
git status

# 2. Stage all tracked files
git add .

# 3. (Optional) Explicitly commit the checkpoint
# git add -f experiments/checkpoints/best_model.pt

# 4. Commit
git commit -m "First public release — v0.1.0

ECG energy-based anomaly detection AUROC 0.80.
Zero-shot cross-domain transfer: negative result documented.
Ablation: ambiguous / no structural artefact.
Full documentation and reproducibility notes included."

# 5. Add your remote (replace with actual URL)
git remote add origin https://github.com/YOUR_USERNAME/spectral-emerge.git

# 6. Push
git push -u origin main
```

---

## Items that may still need manual review

| Item | Note |
|------|------|
| `CITATION.cff` author field | Currently "Independent Author" — update to your preferred attribution |
| `LICENSE` copyright year | Currently 2026 — update name if desired |
| `README.md` repo URL | PLACEHOLDER in CITATION.cff — update after creating GitHub repo |
| `experiments/checkpoints/` | Large files — decide whether to push or link to external release |
