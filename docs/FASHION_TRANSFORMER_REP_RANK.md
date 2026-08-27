# FashionMNIST Tiny Transformer representation-rank falsification

## Status

Prospective falsification test registered in Issue #20 before the CIFAR/ResNet prospective result was known.

**Frozen aggregate decision: PASS.**

The training-only hidden representation effective-rank direction predicted a positive held-out mean benefit, and the observed mean benefit was positive.

## Frozen protocol

- FashionMNIST, stratified 3,000-train / 1,000-test subsets.
- Tiny patch Transformer: 7x7 non-overlapping patches, 16 patch tokens plus CLS, width 48, 2 encoder layers, 4 attention heads, MLP width 96, dropout 0.
- Clean pretraining 5 epochs; seed-guided fine-tuning 2 epochs.
- AdamW: pretrain lr 0.001, fine-tune lr 0.0008, weight decay 1e-4.
- Batch 128, K=8 candidates, Q=4 backward environments.
- 64 training environments beginning at seed 50000; 24 disjoint held-out environments beginning at 60000.
- loss4 versus gradnov4; novelty weight 0.6 inherited without tuning.
- 10 paired replicates.
- Fixed 256-example training probe and fixed training-environment indices [1,9,17,25,33,41,49,57].
- Predictor: effective rank of the concatenated final 48-d CLS representations.

Both methods were trained and both training-only ranks were emitted before any held-out evaluation. Workflow logs show `TRAINING_ONLY` for both methods, then `SEALED_PREDICTION`, then `FINAL` held-out metrics.

## Frozen decision rule

Let `delta_rank` be the mean paired difference in final training-only representation effective rank and `delta_mean` the mean paired difference in held-out environment accuracy.

- `abs(delta_rank) < 0.01` -> UNCERTAIN.
- Otherwise `delta_rank > 0` predicts POSITIVE mean benefit and `delta_rank < 0` predicts NONPOSITIVE mean benefit.
- PASS iff predicted and observed mean directions agree.

Paired p-values and confidence intervals are descriptive and do not change this decision.

## Result

| quantity | estimate | descriptive SE | descriptive paired p |
|---|---:|---:|---:|
| delta representation effective rank | +0.054150 | 0.023554 | 0.04708 |
| delta held-out mean accuracy | +0.0073625 (+0.73625 pp) | 0.002990 | 0.03603 |

Frozen prediction: **POSITIVE**.

Observed held-out mean direction: **POSITIVE**.

Decision: **PASS**.

Approximate two-sided 95% t intervals are +0.00087 to +0.10743 for rank difference and +0.0598 pp to +1.4127 pp for held-out mean benefit.

All ten replicate-level held-out mean differences were positive. This is descriptive; the predeclared test concerns only the condition-average direction.

## Secondary outcomes

- p10 delta: +0.445 pp; its descriptive 95% interval includes zero.
- minimum delta: +1.620 pp; descriptive 95% interval approximately +0.370 to +2.870 pp.
- clean delta: +0.520 pp; its descriptive interval includes zero.

No tail claim is made. The previously failed representation-rank-SD tail rule remains retired.

## Critical limitation

This result does **not** validate representation rank as a reliable per-replicate gate. Applying the same +/-0.01 tolerance retrospectively at replicate level gives 7 directional matches, 2 mismatches, and 1 uncertain replicate. Two replicates had negative rank differences even though their held-out mean differences were positive.

The supported statement is therefore narrower: under this independently preregistered FashionMNIST / Tiny Transformer condition, the **condition-average** training-only representation-rank direction correctly predicted the sign of the condition-average novelty benefit.

It remains neither a calibrated universal threshold nor a causal mechanism.

## Runtime / implementation assumptions

The successful workflow used GitHub-hosted Ubuntu 24.04 CPU, Python 3.12.14, PyTorch 2.10.0, torchvision 0.25.0, deterministic CPU algorithms. Mean measured fine-tune time was about 6.48 s for loss4 and 6.55 s for gradnov4 on that hosted runner. These times are not GPU benchmarks and should not be used to claim production efficiency.

## Evidence

- Issue #20: preregistration and final interpretation.
- PR #21: implementation provenance.
- Actions run 33122703271: successful 10-pair execution and frozen aggregation.
- `results/fashion_transformer_rep_rank_decision10.csv`: frozen aggregate decision.
- `results/fashion_transformer_rep_rank_deltas10.csv`: replicate-level paired deltas.
