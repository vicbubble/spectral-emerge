# Phase 4 Temporal Evaluation Summary

**Records evaluated**: 10

## Per-Record Overview

| Record | Beats | Normal | Anomalous | Prevalence |
|--------|-------|--------|-----------|------------|
| 100 | 2271 | 2237 | 34 | 1.50% |
| 101 | 1864 | 1859 | 5 | 0.27% |
| 102 | 2187 | 99 | 2088 | 95.47% |
| 103 | 2084 | 2082 | 2 | 0.10% |
| 104 | 2228 | 163 | 2065 | 92.68% |
| 105 | 2572 | 2526 | 46 | 1.79% |
| 106 | 2027 | 1507 | 520 | 25.65% |
| 107 | 2137 | 0 | 2137 | 100.00% |
| 108 | 1762 | 1739 | 23 | 1.31% |
| 109 | 2531 | 2491 | 40 | 1.58% |

## AUROC (Anomaly Detection)

| Score Type | Unsupervised? | Mean AUROC ± std |
|------------|---------------|------------------|
| unsupervised_energy | [YES] | 0.8008 ± 0.1618 |
| unsupervised_centroid_distance | [YES] | 0.7240 ± 0.2629 |
| unsupervised_both | [YES] | 0.7497 ± 0.2510 |
| label_informed_normal_distance | [NO - calibrated] | 0.7420 ± 0.2420 |

## Baseline Comparison

**unsupervised_energy**
- Beats random: 89% of records
- Beats constant (>0.5): 100% of records
- Beats raw energy: 0% of records

**unsupervised_centroid_distance**
- Beats random: 67% of records
- Beats constant (>0.5): 78% of records
- Beats raw energy: 33% of records

**unsupervised_both**
- Beats random: 67% of records
- Beats constant (>0.5): 78% of records
- Beats raw energy: 44% of records

**label_informed_normal_distance**
- Beats random: 67% of records
- Beats constant (>0.5): 78% of records
- Beats raw energy: 22% of records

## Regime Detection

- Mean F1: 0.1749 ± 0.1028

## DEQ Iteration Hypothesis

- Supported on: 20% of records

## Success Criteria

- [PASS] min_auroc_0.60_unsupervised_energy
- [PASS] strong_auroc_0.75_unsupervised_energy
- [PASS] min_auroc_0.60_unsupervised_centroid_distance
- [FAIL] strong_auroc_0.75_unsupervised_centroid_distance
- [PASS] min_auroc_0.60_unsupervised_both
- [FAIL] strong_auroc_0.75_unsupervised_both
- [FAIL] beats_random_all_records
- [FAIL] regime_f1_min
- [FAIL] iter_hyp_majority
