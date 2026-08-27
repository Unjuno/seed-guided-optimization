# CIFAR-10 / ResNet-20 representation-rank falsification

## Result

**Frozen decision: PASS.** Issue #12 registered the rule before held-out evaluation. Reps 40-49 were completed in Actions run 33124548679 under the fixed CIFAR-10 / ResNet-20 protocol.

The registered aggregate rule was: if `abs(mean delta rank) < 0.01`, return UNCERTAIN; otherwise the sign of the training-only representation effective-rank difference predicts the sign of the mean held-out accuracy difference.

| quantity | estimate | SE | descriptive paired p | approx. 95% CI |
|---|---:|---:|---:|---:|
| representation effective-rank delta | +0.032051 | 0.020164 | 0.14642 | -0.01356 to +0.07767 |
| held-out mean accuracy delta | +0.0017031 (+0.17031 pp) | 0.0006775 | 0.03310 | +0.01705 to +0.32357 pp |

Because the rank delta exceeded +0.01, the sealed prediction was **POSITIVE**. The observed held-out mean direction was also **POSITIVE**, therefore the preregistered sign test passed.

The rank p-value and interval are descriptive only; PASS is defined by the frozen practical-threshold sign rule, not by rank significance.

## Protocol

- CIFAR-10: 6,000 train / 3,000 test stratified subsets.
- ResNet-20-style GroupNorm model; final 64-d pooled representation.
- 10 clean pretraining epochs; 2 seed-guided fine-tuning epochs.
- AdamW; pretrain lr 0.003, fine-tune lr 0.001, weight decay 1e-4.
- batch 128; K=8; Q=4; `loss4` vs `gradnov4`; novelty weight 0.6.
- 64 training environments from seed 20000; 32 held-out environments from seed 40000.
- fixed 256-example training probe and audit indices `[1,9,17,25,33,41,49,57]`.
- training-only rank and prediction emitted before held-out evaluation.

The workflow was changed from two sequential five-replicate jobs to ten one-replicate jobs only to reduce wall-clock time. Scientific code, seeds and decision rule were unchanged.

## Secondary outcomes

Descriptive only:

- p10 delta: +0.27467 pp; 95% CI includes zero.
- minimum delta: +0.16000 pp; 95% CI includes zero.
- clean delta: +0.18000 pp; 95% CI includes zero.

No tail-safety claim is made. The prior representation-rank-SD tail rule remains retired.

Applying the same rank tolerance at replicate level gives 6 matches, 3 mismatches and 1 uncertain case. The evidence therefore remains about **condition-average direction**, not a calibrated per-run gate.

## Numerical reproducibility boundary

A prior Actions execution used the same scientific code and reps 45-49 but a different workflow topology. Outputs were not bitwise identical. The five-replicate aggregate direction was nevertheless stable: prior execution had rank delta +0.03549 and mean benefit +0.22729 pp; the current run had +0.06346 and +0.26292 pp.

The repository result uses only the internally consistent ten-pair artifact from run 33124548679; rows are not mixed across runs. The cause of cross-run numerical drift has not yet been isolated, so bitwise reproducibility across hosted CPU runners is not claimed.

## Scope

This extends the prospective condition-average sign relation to a ResNet-family convolutional model and strengthens the MLP plus Tiny-Transformer evidence. It does not establish a causal mechanism, universal magnitude mapping, reliable per-run gate, tail predictor, or bitwise cross-run reproducibility.

Evidence files:
- `results/cifar_resnet_rep_rank_all10.csv`
- `results/cifar_resnet_rep_rank_deltas10.csv`
- `results/cifar_resnet_rep_rank_decision10.csv`
