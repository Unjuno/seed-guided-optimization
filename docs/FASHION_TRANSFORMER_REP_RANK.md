# FashionMNIST / Tiny Transformer prospective representation-rank test

## Purpose

This experiment is a prospective falsification test of the frozen training-only rule:

> **sign(delta hidden-representation effective rank) predicts sign(mean held-out benefit).**

The protocol was registered in Issue #20 before the CIFAR/ResNet prospective result was known. It deliberately changes dataset, architecture, and stochastic image generator relative to the small-MLP discovery studies.

## Fixed protocol

- FashionMNIST
- stratified 3,000-image training subset and 1,000-image test subset
- Tiny Patch Transformer: 7x7 non-overlapping patches, 16 patch tokens + CLS token, hidden width 48, 2 encoder layers, 4 attention heads, MLP width 96, dropout 0
- clean pretraining: 5 epochs
- seed-guided fine-tuning: 2 epochs
- AdamW: pretrain lr 0.001, fine-tune lr 0.0008, weight decay 1e-4
- 64 training environment seeds beginning at 50000
- 24 disjoint held-out environment seeds beginning at 60000
- K=8 candidate environments, Q=4 backward environments
- loss-hard vs gradient-novel with novelty weight 0.6, reused without task-specific tuning
- 10 paired replicates

The stochastic environment family varies rotation, translation, contrast, brightness, and additive Gaussian noise.

## Training-only diagnostic

Each paired replicate uses a fixed 256-example training probe and the fixed training-environment indices `[1, 9, 17, 25, 33, 41, 49, 57]`. The final 48-dimensional CLS representation is concatenated across those environments and its effective rank is computed from normalized squared singular values.

Both methods are fully trained and both training-only ranks are sealed before any held-out environment is constructed or evaluated. The per-replicate prediction is emitted before held-out evaluation.

The aggregate preregistered rule is:

- if `abs(mean delta representation rank) < 0.01`: UNCERTAIN;
- otherwise the sign of the rank difference predicts the sign of mean held-out accuracy difference;
- matching signs: PASS;
- opposite signs: FAIL.

Paired p-values are descriptive and do not change this decision.

## Result

Across 10 paired replicates, gradient-novel minus loss-hard produced:

- mean representation effective-rank difference: **+0.05415**;
- rank-difference SE: 0.02355;
- paired raw p for representation rank: 0.04708;
- mean held-out accuracy difference: **+0.7363 percentage points**;
- held-out mean-difference SE: 0.2990 pp;
- paired raw p for held-out mean: 0.03603.

The 95% t-intervals are approximately:

- representation-rank difference: **[+0.00087, +0.10743]**;
- held-out mean difference: **[+0.0598 pp, +1.4127 pp]**.

The mean rank difference exceeds the frozen 0.01 tolerance and therefore registered a **positive** prediction. The observed mean benefit was also positive.

**Decision: PASS.**

All 10 replicate-level held-out mean differences were positive, although the magnitude varied substantially.

## Secondary outcomes

For completeness, the mean paired differences were:

- p10: +0.445 pp;
- minimum environment: +1.620 pp;
- clean accuracy: +0.520 pp;
- held-out environment SD: -0.130 pp.

These are secondary descriptive outcomes. The experiment did not preregister a tail prediction, and the previously failed representation-rank-SD tail rule remains retired. No tail-safety claim is promoted from this run.

## Interpretation boundary

This result materially strengthens the representation-rank candidate because it survives a simultaneous dataset, generator, and architecture change into a small Transformer. It does **not** establish a universal gate, causal mechanism, calibrated threshold, large-model result, or general Transformer validation.

The strongest next falsification remains a larger/different architectural condition and the separately preregistered CIFAR-10 / ResNet-20 trajectory audit.

## Evidence

- `results/fashion_transformer_rep_rank_all10.csv`
- `results/fashion_transformer_rep_rank_deltas10.csv`
- `results/fashion_transformer_rep_rank_decision10.csv`
- reproduction: `experiments/fashion_transformer_rep_rank_audit.py`
- preregistration: Issue #20
