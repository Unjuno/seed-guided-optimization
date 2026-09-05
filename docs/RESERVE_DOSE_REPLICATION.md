# Reserve-image dose-interaction replication (Issue #64)

Authoritative run: `33995390031`. Scientific head: `848e64e253e711f11ae0366291647df17901cfc2`. Aggregate artifact: `9977916390`, SHA256 `cdb96489d2b1178d6d3820658b3dbaedc65c4a6f0f1a131b8d0abefe7e7cd838`.

## Purpose

Issue #61 gave a strong full-strength geometric benefit but only a borderline full-minus-clean interaction on the usual 445-image Digits test subset. Issue #64 preregistered a fixed n=30 replication using the **364-image first-return reserve subset** that the canonical `common.load_digits_split()` discards before retaining those 445 test images.

This is an independent training-replicate/evaluation-image replication within the same Digits dataset and training protocol. It is not an independent dataset, a mediation test, or a post-hoc extension of Issue #61.

## Frozen protocol

- reps `1300-1329`, n=30, no interim stopping or extension;
- MLP `64 -> 128 ReLU -> 10`;
- AdamW lr `0.01`, weight decay `0.001`;
- 10 epochs, batch 128;
- candidate budget K=16, update subset Q=4;
- loss-hard vs gradnov, novelty weight `0.6`;
- one CPU thread and deterministic PyTorch algorithms;
- training environment seeds `44000-44063`;
- evaluation environment seeds `45000-45079`;
- primary strengths: clean `lambda=0` and full geometric `lambda=1`;
- `lambda=0.10` and `0.50` are descriptive only.

All 60 model checkpoints and source hashes were sealed before constructing reserve evaluation environments.

## Frozen result

**DOSE-DEPENDENT BENEFIT REPLICATES ON RESERVE IMAGES.**

Accuracy differences are gradnov minus loss-hard.

| Endpoint | Mean | Paired SE | two-sided t29 95% CI | one-sided p | positive pairs |
|---|---:|---:|---:|---:|---:|
| Full-strength geometric benefit | **+2.27370 pp** | 0.30623 pp | **[+1.64738, +2.90001] pp** | **1.75353e-8** | **29/30** |
| Clean benefit | -0.63187 pp | 0.91602 pp | [-2.50533, +1.24160] pp | 0.75210 | 13/30 |
| Full-minus-clean interaction | **+2.90556 pp** | 0.91238 pp | **[+1.03954, +4.77159] pp** | **0.00172575** | **20/30** |

The two preregistered primary requirements both pass.

## Fixed-strength functional response

The intermediate strengths were frozen before execution but were secondary only.

| Geometric image mixture weight | loss-hard mean accuracy | gradnov mean accuracy | gradnov - loss-hard | one-sided p |
|---:|---:|---:|---:|---:|
| 0.00 | 47.9029% | 47.2711% | -0.6319 pp | 0.75210 |
| 0.10 | 49.5074% | 48.6730% | -0.8345 pp | 0.80245 |
| 0.50 | 57.9129% | 57.1652% | -0.7477 pp | 0.76283 |
| 1.00 | 50.6168% | 52.8905% | **+2.2737 pp** | **1.75353e-8** |

This response is plainly **non-monotone**. The result therefore supports an endpoint/regime interaction at full geometric strength, not a claim that the benefit increases continuously with corruption strength.

## Relation to Issue #61

Issue #61 used a disjoint 445-image evaluation subset and fresh reps `1200-1229`:

- full-strength benefit: +2.84223 pp, one-sided p `8.68e-11`;
- clean benefit: +1.22097 pp, one-sided p `0.10429`;
- full-minus-clean interaction: +1.62125 pp, one-sided p `0.048974` (borderline; two-sided CI crossed zero).

Issue #64 now shows:

- full-strength benefit: +2.27370 pp, p `1.75e-8`;
- clean benefit: -0.63187 pp, p `0.75210`;
- interaction: +2.90556 pp, p `0.001726`.

The full-strength advantage is positive and precisely estimated in both disjoint evaluation-image subsets. Clean benefit is unstable and not significant in either block. This strengthens the narrow claim that the advantage is associated with the full structured-shift regime rather than a generic clean-input improvement.

A descriptive pooling of the two independently trained n=30 blocks gives a 60-replicate full-strength mean benefit of +2.55796 pp and a full-minus-clean interaction of +2.26341 pp. That pooling was not preregistered and does not alter either frozen decision.

## Integrity verification

Independent post-run verification reproduced the aggregate without using the summarizer's result as input:

- all six shard ZIP SHA256 digests matched GitHub artifact metadata;
- all **60/60 checkpoint SHA256 hashes** matched their sealed manifests;
- source-hash sets and reserve-split hash sets were identical across all six shards;
- each manifest recorded `n_train=988`, `n_reserve=364`;
- the reconstructed training and usual-test arrays exactly matched `common.load_digits_split()`;
- all **240** replicate/method/strength metric rows matched the aggregate exactly;
- all **14,460** environment rows were re-aggregated independently;
- maximum environment-to-aggregate error was `2.220446049250313e-16`;
- paired values reconstructed from the 240 metric rows matched the public paired CSV to numerical precision.

Five shards ran on AMD EPYC 7763 and one on AMD EPYC 9V74. The primary comparisons are paired within replicate on the same runner. No cross-hardware bitwise or efficiency claim is made.

## Interpretation boundary

This experiment materially strengthens two narrow observations within the Digits/MLP geometric protocol:

1. gradnov has a repeatable positive mean-accuracy advantage over loss-hard at full geometric shift;
2. that full-strength advantage exceeds the clean-input advantage in a second, disjoint evaluation-image subset.

It **does not** establish that reusable latent factors are the causal mediator, that the response is monotone in shift strength, that every structured shift benefits, or that SGO is universally superior. The current mechanistic core remains finite-budget selection of hard, non-redundant gradients; the downstream internal mediator remains unidentified.
