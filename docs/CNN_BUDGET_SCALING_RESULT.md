# SmallCNN finite-budget Q-scaling replication (Issue #70)

Authoritative Actions run: **34002906540**. Scientific head: `a7fc76ed83ce8060d599e6601720d3413c83d317`.

## Frozen decision

**CNN FINITE-BUDGET COVERAGE REPLICATES**.

Fresh paired reps 1500-1529 used the existing `SmallCNN`, AdamW lr=5e-3 / wd=1e-3, 10 epochs, batch128, K=16, and Q in {2,4,8,12,16}. Training environment seeds were 48000-48063. Fresh full-strength geometric heldout seeds were 49000-49079 and evaluation used all 809 canonical nontraining Digits images.

For each rep and Q, `B[r,Q] = mean_accuracy_gradnov - mean_accuracy_loss_hard`. The preregistered primary contrast was `A_B = mean(B[2],B[4]) - mean(B[12],B[16])`.

- mean low-Q benefit: **+2.05380 pp**;
- mean high-Q benefit: **+0.78886 pp**;
- attenuation `A_B`: **+1.26494 pp**;
- paired SE: **0.39198 pp**;
- t29 95% CI: **[+0.46325,+2.06663] pp**;
- one-sided p: **0.00154834**;
- positive attenuation pairs: **20/30**.

## Per-Q heldout mean benefit

| Q | coverage Q/K | gradnov - loss-hard | paired SE | one-sided p | positive pairs |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.125 | +2.03446 pp | 0.42033 pp | 1.98e-5 | 24/30 |
| 4 | 0.250 | +2.07314 pp | 0.50235 pp | 1.42e-4 | 24/30 |
| 8 | 0.500 | +1.19499 pp | 0.44134 pp | 0.00562 | 21/30 |
| 12 | 0.750 | +1.57772 pp | 0.31754 pp | 1.38e-5 | 24/30 |
| 16 | 1.000 | 0 exactly | 0 | 0.5 by zero-effect convention | 0/30 |

The curve is **not strictly monotonic**: Q=12 exceeds Q=8. The supported claim is the frozen low-vs-high attenuation plus exact disappearance at Q=K, not a universal monotone Q law.

## Q=K exact identity

Q=16 had **zero mismatches** across all 30 paired reps. Loss-hard and gradnov had identical:

- canonical SHA256 digest of model state tensors;
- every parameter tensor bitwise;
- selected pairwise-gradient novelty;
- candidate and selected training loss diagnostics;
- heldout mean, SD, p10, minimum and clean accuracy.

This is stronger than an accuracy-only identity control: when all K candidates contribute to every update, there is no remaining selector degree of freedom and the learned model state itself is exactly identical under this deterministic CPU execution.

## Manipulation diagnostic

Mean gradnov-minus-loss-hard selected pairwise-gradient novelty was:

- Q2: +0.30948;
- Q4: +0.14752;
- Q8: +0.05465;
- Q12: +0.02325;
- Q16: 0 exactly.

Thus the selector's non-redundancy manipulation shrinks as the subset constraint relaxes and disappears when Q=K.

## Independent verification

The six original shard ZIP SHA256 digests exactly matched GitHub artifact metadata. All 300 checkpoint file hashes matched their sealed manifests. All archived source hashes matched their manifests. Canonical state-tensor digests recomputed from every checkpoint matched the training CSVs. The 300 aggregate rows matched the six shard CSVs up to a maximum CSV round-trip numerical difference of `4.44e-16`. Reaggregation of all **24,000 environment rows** reproduced heldout mean/SD/p10/minimum with maximum error `1.11e-16`. Q16 parameter tensors were independently compared with `torch.equal` and had zero mismatches.

Hosted runners included AMD EPYC 7763, AMD EPYC 9V74 and Intel Xeon Platinum 8573C. All scientific comparisons are paired within replicate; no speed or cross-hardware bitwise-reproducibility claim is made.

## Mechanistic consequence

The finite-budget coverage result is no longer confined to the MLP parameterization. Two preregistered MLP n=30 Q-scaling experiments previously produced low-minus-high attenuations of about +2.03 pp and +2.09 pp, each with exact Q=K identity. This fresh SmallCNN experiment produces +1.26 pp with the same qualitative control structure.

The current supported scope is therefore:

> Within the Digits/geometric family, under both the tested MLP and SmallCNN parameterizations, the advantage of hard + gradient-nonredundant environment selection is larger when only a small subset of K candidates can contribute, attenuates in the preregistered low-vs-high contrast as Q approaches K, and becomes exactly zero when Q=K.

This remains **within one dataset/generator family**. It does not establish cross-dataset universality, a universal monotone Q curve, or the downstream function-space mediator.

## Reproduction

Primary statistics can be recomputed from public paired rows without retraining:

```bash
python experiments/check_cnn_budget_paired.py --input-dir results
```

Original Actions artifacts retain checkpoints, per-environment outputs, manifests, runtime metadata and source snapshots.