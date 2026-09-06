# CNN Q-scaling: sensitivity to the structurally identical endpoint

**Status: post-hoc analysis of existing data, not a fresh replication.** Original Issue #70 / result PR #72 / authoritative run `34002906540` remain unchanged. The registered decision remains **CNN FINITE-BUDGET COVERAGE REPLICATES**.

## Why this additional analysis is necessary

At Q=K=16, both selectors choose the same complete candidate set, reduced in the same sorted order. Given identical initialization, schedule, optimizer state and deterministic execution, their updates coincide. The observed 30/30 exact model-state and metric identities therefore validate an implementation control. They are not independent evidence that gradient-subspace coverage is a causal mediator.

The original high-Q average includes this forced-zero endpoint. We recompute a different, explicitly post-hoc contrast that compares low Q only against Q=12, where selection still operates. This does not retroactively change the preregistered test or its decision.

## Variables and exact contrast relationship

| Symbol | Meaning (日本語) | SI unit | Definition | Domain / premise | Type |
|---|---|---|---|---|---|
| r | 対応付き学習反復 | 1 | replicate identifier | integers 1500-1529 | scalar index |
| K | 候補環境数 | 1 | fixed candidate count | 16 | integer scalar |
| Q | 選択環境数 | 1 | selected subset size | 2,4,8,12,16 | integer scalar |
| B(r,Q) | 手法間の平均正答率差 | 1 | gradnov minus loss-hard, averaged over 80 heldout environments | [-1,1], fixed Digits image/environment samples | real scalar |
| L(r) | 低Qの平均改善 | 1 | [B(r,2)+B(r,4)]/2 | [-1,1] | real scalar |
| H(r) | 元の高Qの平均改善 | 1 | [B(r,12)+B(r,16)]/2 | [-1,1] | real scalar |
| C(r) | 事前登録された差 | 1 | L(r)-H(r) | [-2,2] | real scalar |
| E(r) | Q16を除く事後的な差 | 1 | L(r)-B(r,12) | [-2,2] | real scalar |

The verified identity gives B(r,16)=0 for every replicate. Substituting this into H(r) gives H(r)=B(r,12)/2 and hence B(r,12)=2H(r). Therefore E(r)=L(r)-2H(r)=C(r)-H(r). No independence assumption is needed for this algebra; statistics are computed on the paired replicate-level contrasts, not by treating Q levels as independent samples.

**Unit check:** every term is an accuracy fraction, so subtraction preserves SI unit 1. Multiplication by 100 converts an accuracy difference to percentage points (pp), not a relative percentage change.

## Results

| Analysis | Mean (pp) | Paired SE (pp) | Two-sided t29 95% CI (pp) | One-sided p | Positive pairs |
|---|---:|---:|---:|---:|---:|
| Registered: low Q minus mean(Q12,Q16) | +1.26494 | 0.39198 | [+0.46325,+2.06663] | 0.00154834 | 20/30 |
| **Post-hoc: low Q minus Q12** | **+0.47608** | **0.49554** | **[-0.53741,+1.48956]** | **0.172315** | **16/30** |

The first row reproduces the frozen primary result. The second is exploratory: its p-value is descriptive, not a newly preregistered confirmatory test. Its interval includes zero and meaningful positive differences. This is neither proof of an exactly zero difference nor evidence that the original improvement disappears. It does mean that the original low-vs-high PASS should not be promoted to a confirmed attenuation law among non-full subset sizes.

Per-Q observed mean benefits remain Q2 +2.03446, Q4 +2.07314, Q8 +1.19499, Q12 +1.57772, Q16 0 pp. This pattern is not monotone. In particular, Q12 has a positive mean benefit; including the forced-zero Q16 halves that component in the registered high-Q average.

## H / T / D / C / U

**H (exploratory):** the mean low-Q benefit exceeds Q12 benefit even after the forced-zero endpoint is excluded.

**T:** reanalyze all 30 original paired replicates, without retraining, discarding rows, changing environments, or adding repetitions. No new experiment has been preregistered here.

**D:** report the estimate, interval and descriptive p-value. There is no new confirmatory PASS/FAIL label. The original registered PASS is retained; endpoint-excluded attenuation is not established by this analysis.

**C:** positive gradnov benefits may persist across several non-full subset sizes, with exact disappearance only when the selectors become identical. Other explanations include changes in objective weighting and optimization dynamics; coverage mediation remains unidentified.

**U:** paired standard errors quantify training-replicate variability conditional on the fixed image/environment samples. The t29 coverage factor is 2.0452296421. Dataset-level, environment-population and heterogeneous-hardware uncertainty are not included in a combined uncertainty estimate. No total combined uncertainty is claimed. Post-hoc choice of the new contrast is an additional inferential limitation.

## Reproduction and audit

Original conditions: SmallCNN; Digits 988 train /809 nontraining images; ten epochs; batch128; AdamW learning rate0.005, weight decay0.001; K16; novelty weight0.6; one CPU thread. Shards used AMD EPYC 7763/9V74 and Intel Xeon Platinum8573C. Recorded CPU clock snapshots were not fixed clocks; there is no speed or FLOP-savings claim.

The existing public `results/cnn_budget_attenuation30.csv` is sufficient because exact Q16 identity supplies the algebra above. Its Git blob is `b763dbd54d594ac823e4228cbdfad62ad1c09084`; SHA256 is `78b2506e26eb8299884db7b145744c7395dd92c68b1820ec349d06e0f00809c7`.

```bash
python experiments/check_cnn_budget_endpoint_sensitivity.py --selftest
python experiments/check_cnn_budget_endpoint_sensitivity.py --input-dir results
```

The checker also requires the published Q16 identity decision and empty mismatch file. It independently computes means, paired standard errors, t29 confidence intervals, and directional p-values with NumPy/pandas/SciPy. It does not verify checkpoint binaries itself; the archive audit below supplies that independent check. New summary CSV: `results/cnn_budget_endpoint_sensitivity.csv`. Scientific CSV values are stored as fractions.

A separate archive-level audit rechecked all seven ZIP digests, 300 checkpoint file hashes, 300 canonical tensor digests, 30 exact Q16 tensor pairs and 24,000 environment rows. Environment metrics reaggregated exactly under round-trip parsing; aggregate CSV comparison differed by at most 4.440892098500626e-16. Missing/duplicate/nonfinite/inconsistent paired rows are rejected in the new checker's synthetic tests. This audit is not another independent training replication.

The original scientific script, primary CSVs, registered decisions and ongoing CIFAR protocol are unmodified by this note.
