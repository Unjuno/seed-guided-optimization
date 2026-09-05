# SmallCNN full-strength regime-interaction replication (Issue #67)

Authoritative run: `33995962430`. Scientific head: `f776d9742dd1b086b3cf5d7b2b0a0a6b0e99d2c6`. Aggregate artifact: `9978099785`, SHA256 `93cff5313c772db20143da24e3847f988e0968e1bd2f03787e9bc4326288d9a7`.

## Purpose

Issues #61 and #64 established a positive full-strength geometric benefit for the MLP and produced a replicated full-minus-clean interaction in the reserve-image block. Issue #67 preregistered a cross-architecture falsification using the existing `SmallCNN` and its established AdamW learning rate.

The primary question was deliberately strict: does SmallCNN reproduce both the positive full-strength effect and the claim that this effect is larger than its clean-input benefit?

## Frozen protocol

- reps `1400-1429`, n=30;
- `common.SmallCNN` unchanged;
- AdamW lr `5e-3`, weight decay `1e-3`;
- 10 epochs, batch128, K=16, Q=4;
- loss-hard vs gradnov, novelty weight 0.6;
- canonical 988-image training split;
- full 809-image nontraining union for evaluation;
- train environment seeds `46000-46063`;
- evaluation environment seeds `47000-47079`;
- fixed lambda `[0, 0.10, 0.50, 1.0]` with no calibration or outcome-dependent strength selection;
- all checkpoints and source hashes sealed before evaluation.

## Frozen primary result

**CNN FULL EFFECT REPLICATES / CLEAN INTERACTION DOES NOT.**

| Endpoint | Mean | Paired SE | two-sided t29 95% CI | one-sided p | positive pairs |
|---|---:|---:|---:|---:|---:|
| Full-strength benefit | **+2.41698 pp** | 0.50567 pp | **[+1.38276,+3.45119] pp** | **2.33876e-5** | **26/30** |
| Clean benefit | **+3.43634 pp** | 1.06931 pp | **[+1.24935,+5.62333] pp** | **0.00160254** | 21/30 |
| Full-minus-clean interaction | -1.01937 pp | 1.23411 pp | [-3.54341,+1.50468] pp | 0.79222 | 12/30 |

The preregistered full-strength effect passes. The preregistered cross-architecture interaction fails and is directionally negative.

## Fixed-strength response

| Geometric image mixture weight | loss-hard mean accuracy | gradnov mean accuracy | gradnov - loss-hard | one-sided p |
|---:|---:|---:|---:|---:|
| 0.00 | 51.3350% | 54.7713% | **+3.4363 pp** | 0.001603 |
| 0.10 | 53.2543% | 56.7826% | **+3.5282 pp** | 0.001344 |
| 0.50 | 60.3283% | 63.5789% | **+3.2506 pp** | 0.000364 |
| 1.00 | 50.0485% | 52.4655% | **+2.4170 pp** | 2.3388e-5 |

Unlike the MLP reserve-image block, SmallCNN shows a positive gradnov advantage at every frozen strength, including clean input. Therefore the MLP's full-specific endpoint interaction is **not architecture-general** in this Digits protocol.

## Prespecified secondary difficulty diagnostic

The lambda=.10 and lambda=1 loss-hard baselines were 53.2543% and 50.0485%, an absolute difference of **3.20581 pp**. This narrowly exceeded the preregistered 3 pp proximity threshold.

The full-minus-lambda=.10 benefit contrast was **-1.11125 pp**, 95% CI [-3.61002,+1.38752] pp, one-sided p `0.81472`. Thus the secondary difficulty-proximate support condition also fails; it cannot be used to rescue a regime-specific explanation.

## Integrity verification

Independent post-run verification found:

- all six shard ZIP SHA256 digests matched GitHub artifact metadata;
- all **60/60 checkpoint hashes** matched sealed manifests;
- source-hash and split-hash sets were identical across all six shards;
- every manifest reported `n_train=988`, `n_eval=809`;
- all **240** aggregate metric rows matched shard outputs exactly;
- all **14,460** environment rows reaggregated successfully;
- maximum environment-to-aggregate numerical error was `2.220446049250313e-16`;
- paired statistics were independently recomputed and matched the frozen aggregate.

Runner CPU families included AMD EPYC 7763, AMD EPYC 9V74, Intel Xeon Platinum 8370C and Intel Xeon Platinum 8573C. All scientific comparisons are paired within replicate; no cross-hardware bitwise or efficiency claim is made.

## Mechanistic update

The combined MLP/CNN evidence now separates two claims that should not be conflated:

1. **Full-strength geometric benefit is robust within this Digits generator.** Fresh MLP blocks and this SmallCNN block all show a positive gradnov-minus-loss-hard mean advantage at full geometric strength.
2. **Full-strength specificity is architecture-dependent.** MLP reserve data showed little/negative benefit at clean and intermediate strengths with a strong positive full effect; SmallCNN shows significant positive benefit across clean, weak, mid and full conditions.

Therefore a universal mechanism of the form `structured-shift strength -> larger SGO benefit` is not supported. The most stable causal-level statement remains upstream: under a binding subset budget, hard/non-redundant gradient selection changes the optimization trajectory and can improve held-out mean performance. How that coverage is converted into clean-versus-shifted function behavior depends on the architecture/optimization regime and remains unidentified.

This result weakens a simple reusable-factor or regime-specific conversion theory as a universal explanation. It does **not** weaken the independently replicated finite-budget coverage result or the positive full-strength performance result itself.
