# Phase 5 Zero-Shot Cross-Domain Summary

**Model**: ECG checkpoint (unchanged, frozen)
**Adaptation**: None
**Primary score**: energy-only (no target labels)

## Synthetic Results (ground-truth labels)

| Map Mode | AUROC | AUPR | F1-opt | Anomaly Prevalence |
|----------|-------|------|--------|-------------------|
| constant_pad | 0.5186 | 0.3154 | 0.4614 | 29.98% |
| reflect_pad | 0.5102 | 0.3034 | 0.4617 | 29.98% |
| linear_resize | 0.5128 | 0.3089 | 0.4613 | 29.98% |

## Universality Verdict: **NO**

- Mapping sensitivity (max AUROC diff): 0.0084
- All mappings > 0.70: NO
- Any mapping > 0.70: NO
- adapter_used: false
- training_performed: false
