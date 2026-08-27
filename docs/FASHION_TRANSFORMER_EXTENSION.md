# FashionMNIST Tiny Transformer independent extension

## Status

The preregistered independent extension in Issue #26 completed all 20 new paired replicates (IDs 10-29) under the unchanged protocol.

**Extension-only decision: REPLICATES / PASS under the frozen condition-average sign rule.**

## Primary extension-only result

The extension was evaluated independently of the original reps 0-9 result.

| quantity | estimate | descriptive SE | descriptive paired p | approx. 95% t interval |
|---|---:|---:|---:|---:|
| delta representation effective rank | +0.078534 | 0.023489 | 0.003414 | +0.02937 to +0.12770 |
| delta held-out mean accuracy | +0.004377 (+0.4377 pp) | 0.001918 | 0.03420 | +0.0362 to +0.8392 pp |

The frozen practical tolerance was 0.01, so the training-only rank difference registered a **POSITIVE** prediction. The observed held-out mean difference was also positive. Therefore the independent extension reproduced the directional relation.

## Secondary outcomes

These are descriptive and do not alter the frozen decision:

- p10 delta: +0.441 pp; approximate 95% interval -0.213 to +1.095 pp;
- minimum delta: -0.095 pp; interval -0.910 to +0.720 pp;
- clean delta: +0.700 pp; interval +0.186 to +1.214 pp.

There is still no validated tail-safety rule.

## Per-replicate limitation

Using the same +/-0.01 rank tolerance descriptively at the replicate level gives 13 directional matches, 5 mismatches, and 2 uncertain cases among reps 10-29. Held-out mean benefit was positive in 14/20 extension replicates and nonpositive in 6/20.

Thus the evidence supports a **condition-average directional predictor**, not a reliable per-run selector gate.

## Combined 30-pair precision estimate

After the extension-only decision was frozen, reps 0-29 were combined for precision only:

- mean delta representation effective rank: +0.070406, SE 0.017405, descriptive p=0.000354, approximate 95% interval +0.03481 to +0.10600;
- mean held-out accuracy delta: +0.005372 (+0.5372 pp), SE 0.001613, descriptive p=0.002367, approximate 95% interval +0.2074 to +0.8670 pp.

This combined estimate does not replace the independent extension-only replication result.

## Interpretation

The original n=10 FashionMNIST/Tiny Transformer prospective PASS was followed by an exact-protocol independent n=20 extension that again produced positive training-only representation-rank change and positive held-out mean benefit. This materially strengthens evidence that the aggregate directional relation transfers beyond the small-MLP discovery family.

It does not establish causality, a universal magnitude mapping, a calibrated per-run threshold, or tail robustness.

## Runtime / reproducibility

Execution used the same deterministic CPU protocol as the original Fashion experiment: GitHub-hosted Ubuntu, Python 3.12, PyTorch 2.10.0, torchvision 0.25.0. No GPU or production-efficiency claim is inferred from hosted-runner wall-clock measurements.

## Evidence

- Issue #26: preregistration and final interpretation.
- PR #27: extension workflow and result integration.
- Actions run 33124018102: four 5-replicate shards plus successful frozen aggregation.
- `results/fashion_transformer_rep_rank_extension_all20.csv`
- `results/fashion_transformer_rep_rank_extension_deltas20.csv`
- `results/fashion_transformer_rep_rank_extension_decision20.csv`
- `results/fashion_transformer_rep_rank_combined_decision30.csv`
