# Project Note

*[Read in Portuguese / Leia em Português](project_note.pt-BR.md)*

## What this is

This repository is not a formal academic paper.
It is an organized record of an independent experimental exploration.

The project began from a simple curiosity:

> Can a model with implicit equilibrium dynamics develop useful internal
> structure from continuous signals, and can that structure be used
> in a practical way — without explicit supervision?

The implementation evolved through five phases.
Some ideas worked better than expected.
Others failed clearly.
Both kinds of result are documented here.

---

## Original motivation

The original motivation for this project came from reading the paper *“Emergent quantization from a dynamic vacuum”* by Harold White and collaborators, published in *Physical Review Research* in 2026. That article explores, within theoretical physics, the idea that discrete structures might emerge from continuous dynamics. 

This computational exploration was inspired by that conceptual intuition. However, the results presented here should be read as an independent investigation into modelling and anomaly detection in signals, and **not** as a direct test or validation of the physical framework proposed in that paper.

The original inspiration was conceptual, not product-driven.

I was interested in whether:

- continuous latent dynamics converging to a fixed point  
- energy-like quantities over those fixed points  
- spectral regularisation on the Jacobian at equilibrium  

might produce useful internal states without explicitly imposing discrete
structure on the model. The intuition was that useful discreteness might
*emerge* rather than be designed in.

That broad motivation is still intellectually interesting.
But the results obtained so far do not justify large claims about universal
emergent representations.

The strongest practical result is narrower:
the model learned a useful anomaly signal in real ECG data using only the
energy at its implicit fixed point.

---

## What was actually tested

### Phase 1–2: Synthetic structure experiments

Early phases explored:

- whether latent modes emerged in controlled data
- whether spectral / Jacobian regularisation changed the latent geometry
- whether collapse occurred in higher-dimensional latent spaces

These experiments were useful for diagnosing behaviour and shaping intuitions,
but they did not demonstrate a strong new principle in their own right.

### Phase 3: ECG data — initial positive result

The first convincing applied result came from PhysioNet MIT-BIH ECG data:

- real annotated ECG records
- unsupervised evaluation framing (no labels at inference time)
- anomaly signal above random in several records

A leave-one-record-out protocol was used to avoid leakage.
The primary score — model energy at the implicit fixed point — was computed
without any access to anomaly labels.

This was the first point where the project started to feel practically meaningful.

### Phase 4: Temporal evaluation

A temporal evaluation over ordered heartbeat sequences was added.
The model still processed each beat independently; the temporal layer was
about *evaluation*, not about adding a temporal model.

Key findings:

- decent anomaly signal (mean AUROC ≈ 0.80 with energy score)
- weak regime-transition signal (F1 below target)
- DEQ solver iteration count was not a reliable anomaly cue

This phase clarified what the model was actually doing:
useful event-level anomaly detection, weaker regime-level analysis.

### Phase 5: Zero-shot cross-domain transfer

A critical experiment tested whether the ECG-trained model generalised to
a very different signal domain (synthetic network latency / RTT data):

- no retraining
- no adapter
- no learned bridge
- only deterministic input mapping (zero-padding / reflection / linear resize)

Three mapping modes were all tested. All failed:

- AUROC ≈ 0.51–0.52 across all mappings (random-level)
- Ablation confirmed: shuffled and random-noise inputs gave similar scores to
  real inputs, ruling out a strong structural transfer claim

This negative result is extremely important because it sharply limits the
strongest possible claim one might have wanted to make.

---

## Honest interpretation

The project currently supports:

1. The architecture produces a useful energy-based anomaly signal in real ECG.
2. The temporal/regime discovery story remains weaker than the anomaly story.
3. Strong universal zero-shot cross-domain transfer is not supported.

---

## Why this is worth sharing

A clean negative result is useful.
A modest real positive result is useful.
Open, legible experiment logs are useful.

This project is not presented as final truth.
It is presented as a serious exploration with inspectable artefacts.

That alone may help others:

- avoid overstating similar ideas
- reuse the anomaly-detection pipeline
- test stronger versions of the original hypothesis
- or build in a more rigorous direction
