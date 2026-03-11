# Spectral Emerge: Energy-Based Anomaly Detection with Implicit Latent Dynamics

*[Read in Portuguese / Leia em Português](short-report.pt-BR.md)*

**Independent experimental report — v0.1**

---

## Abstract

We describe an independent experimental exploration of whether implicit
equilibrium-based neural dynamics can produce useful internal representations
for unsupervised anomaly detection in continuous signals.

The architecture combines a fixed-point implicit layer (DEQ-style) with an
energy function computed at the latent equilibrium. The energy is used as
an anomaly score without labels at inference time.

On real PhysioNet MIT-BIH ECG data, the energy score achieved a mean AUROC
of approximately 0.80 across 10 records in a leave-one-record-out protocol.
This is a genuine positive result.

A subsequent zero-shot cross-domain experiment — testing the ECG-trained
model directly on synthetic network latency data with no adaptation — produced
an AUROC of approximately 0.51 across all three tested input mappings,
indistinguishable from random. This negative result sharply limits any claim
of universal latent structure transfer.

Both results are documented transparently.

---

## 1. Motivation

This project was originally inspired, at a conceptual level, by the paper *"Emergent quantization from a dynamic vacuum"* (White et al., Physical Review Research, 2026), which explores the possibility of discrete structure emerging from continuous dynamics. The experiments in this repository were motivated by that general intuition but **do not** constitute a validation of the physical theory proposed in the article.

The starting point was a conceptual question:

> Can a model whose latent state is defined by an implicit equilibrium
> (rather than an explicit RNN hidden state or discrete bottleneck) learn
> useful structure from continuous signals without explicit supervision?

The hypothesis was that energy-like quantities computed at the fixed point
might serve as anomaly signals — without labels, and potentially across domains.

---

## 2. Method Overview

### 2.1 Model architecture

- **Encoder:** small MLP, maps input x ∈ ℝ^187 to context c ∈ ℝ^d
- **DEQ layer:** implicit fixed-point solver, finds z* such that f(z*, c) = z*
- **Energy:** scalar E(z*, x) computed at the equilibrium
- **Spectral regularisation:** penalises the Jacobian spectral radius at z*
  to encourage stable fixed-point landscapes

The model processes each input independently.  
No recurrent connections. No temporal model.

### 2.2 Training

Trained on ECG data (PhysioNet MIT-BIH, records 100–109, excluding one for test).
Loss: reconstruction + spectral penalty.
No anomaly labels used during training.

### 2.3 Anomaly scoring

**Primary score:** `energy_only` — min-max normalised E(z*, x) over each record.  
No labels are used to compute this score.  
Reported as the main zero-shot unsupervised metric.

**Secondary score:** centroid distance in latent space (unsupervised GMM fit).  
**Calibrated score (exploratory):** distance to label-informed normal centroid.  
The latter is explicitly separated from the primary unsupervised claim.

---

## 3. ECG Results (Phase 3–4)

### 3.1 Dataset

PhysioNet MIT-BIH Arrhythmia Database.  
Records 100–109, 2000–3500 beats each.  
Protocol: leave-one-record-out (10 folds).

### 3.2 Anomaly detection

| Score | Mean AUROC | Mean AUPR |
|-------|-----------|-----------|
| `unsupervised_energy` | **0.801** | 0.185 |
| `unsupervised_centroid_distance` | 0.627 | 0.121 |
| `label_informed_normal_distance` | 0.788 | 0.289 |

Primary unsupervised score AUROC > 0.75 — exceeds the "strong" success criterion.

### 3.3 Temporal evaluation (Phase 4)

Beat-level temporal evaluation over ordered sequences:

- Anomaly detection signal: AUROC ≈ 0.80 (consistent with above)
- Regime-transition detection: F1 < 0.30 (weak — below target)
- DEQ iteration hypothesis: not consistently supported

**Interpretation:** the model is a practical energy-based anomaly detector.
The regime-discovery story is not yet demonstrated.

---

## 4. Zero-Shot Cross-Domain Transfer (Phase 5)

### 4.1 Setup

- Model loaded from ECG checkpoint, completely frozen
- No retraining, no adapter, no optimiser
- Input domain: synthetic network latency (RTT) streams, 4 regimes
- Three deterministic input mappings: constant-padding, reflection-padding, linear resize
- Primary score: energy-only (no target-domain labels)

### 4.2 Results

| Mapping | Synthetic AUROC |
|---------|---------------|
| constant_pad | 0.519 |
| reflect_pad | 0.510 |
| linear_resize | 0.513 |

AUROC ≈ 0.51 across all mappings. Sensitivity across mappings: 0.008 (i.e. consistent failure, not mapping-dependent noise).

**Verdict: NO** — zero-shot universal transfer not supported.

### 4.3 Ablation

| Condition | Typical AUROC |
|-----------|--------------|
| Original windows | 0.52 |
| Shuffled windows | 0.52 |
| Gaussian noise | 0.50 |

Interpretation: AMBIGUOUS — original windows do not significantly outperform
shuffled or noise inputs. No evidence of structural transfer from ECG to network
latency. The model is responding primarily to coarse statistical properties of
the mapped input, not to domain-relevant signal structure.

---

## 5. Latent Geometry Analysis

The PCA projection of ECG vs. network latent vectors (z*) shows clear domain
separation. The cosine similarity between the ECG-normal centroid and the
network-stable centroid was approximately 0.23 — far from alignment.

This geometric analysis supports the zero-shot negative result: the two domains
occupy very different regions of the latent space, making direct energy
comparison across domains uninformative.

---

## 6. Conclusion

The energy-based implicit architecture produced a genuine and replicable anomaly
detection signal in real ECG data without using labels at inference time. This
practical result is the project's primary contribution.

The hypothesis of universal zero-shot cross-domain latent structure transfer
was tested and not supported. This negative result is reported transparently
and constitutes an important bound on the project's claims.

Future directions may include:

- lightweight target-domain adaptation (single projection layer, small labelled set)
- evaluation on additional physiological signals (EEG, PPG)
- more rigorous analysis of the energy landscape and Jacobian geometry

---

*This report documents an independent exploration. It is not peer-reviewed.
All code, data paths, and result artefacts are available in the repository.*
