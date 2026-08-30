# FashionMNIST Medium Transformer prospective representation-rank test

## Status

Issue #23 preregistered an architecture-capacity falsification of the frozen condition-average representation-rank direction rule. PR #24 implemented the test and Actions run `33123590973` completed all 10 paired replicates.

**Frozen decision: PASS.**

## Protocol

Relative to the completed Tiny Transformer condition, this test held the FashionMNIST split, stochastic environment generator, optimizer family and hyperparameters, K/Q selector budget, novelty coefficient, training-only probe, and held-out definition fixed while increasing Transformer capacity:

- hidden width: 48 -> **96**;
- encoder layers: 2 -> **4**;
- attention heads: 4 -> **8**;
- MLP width: 96 -> **192**;
- paired replicate IDs: 10-19;
- candidate K=8, backward Q=4;
- novelty weight 0.6;
- training-only representation: final 96-d CLS vector;
- fixed 256-example probe across eight training environments.

Both methods were fully trained and the training-only rank direction was sealed before held-out environment evaluation.

## Frozen rule

Across the 10 paired replicates:

- `abs(mean delta representation effective rank) < 0.01` -> UNCERTAIN;
- positive rank delta predicts positive condition-average held-out mean benefit;
- negative rank delta predicts nonpositive benefit;
- sign agreement -> PASS; disagreement -> FAIL.

Paired p-values are descriptive and do not alter the frozen decision.

## Result

Gradient-novel minus loss-hard:

| quantity | estimate | descriptive SE | descriptive paired p |
|---|---:|---:|---:|
| representation effective-rank delta | **+0.102929** | 0.023926 | 0.001985 |
| held-out mean accuracy delta | **+0.009579 (+0.9579 pp)** | 0.003979 | 0.039433 |
| held-out p10 delta | +1.226 pp | 0.586 pp | descriptive only |
| held-out minimum delta | +1.700 pp | 0.767 pp | descriptive only |
| clean accuracy delta | +1.050 pp | 0.822 pp | descriptive only |

The mean rank delta was above the fixed 0.01 tolerance, so the sealed prediction was **POSITIVE**. The observed condition-average held-out mean delta was also positive. Therefore the preregistered decision is **PASS**.

At replicate level, held-out mean benefit was positive in 7/10 runs and nonpositive in 3/10. This again supports a condition-average predictor, not a reliable per-run deployment gate.

## Interpretation

The Tiny Transformer result was not confined to the original 48-dimensional, 2-layer representation. Under a 2x-width / 2x-depth architecture increase with no selector or optimizer retuning, gradient novelty again increased training-only representation effective rank and improved condition-average held-out mean accuracy.

This strengthens architecture-capacity robustness of the directional relation. It does not establish causality, a universal magnitude law, large-Transformer validity, or tail robustness.

## Evidence

- Issue #23: preregistration.
- PR #24: implementation.
- Actions run `33123590973`: two 5-replicate shards plus frozen aggregate.
- `results/fashion_medium_transformer_rep_rank_all10.csv`
- `results/fashion_medium_transformer_rep_rank_deltas10.csv`
- `results/fashion_medium_transformer_rep_rank_decision10.csv`
