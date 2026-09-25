# Issue115: mean-tail replication did not meet the primary criteria

Outcome generated 2026-09-12; independently audited and documented 2026-09-25.

**Frozen scientific decision: NO MEAN-TAIL TRADEOFF REPLICATION.**

This is not evidence of equal performance, zero effect or tail safety. Successful execution and an audit PASS are distinct from support for a scientific hypothesis. The earlier Issue104 positive mean contrast is retained; this fresh replication did not confirm it at the registered threshold.

## Registration and provenance

- Registration: [Issue115](https://github.com/Unjuno/seed-guided-optimization/issues/115).
- Implementation: [PR119](https://github.com/Unjuno/seed-guided-optimization/pull/119), head `4a2c36960cf8c73cb21778f1f6ddf735e2ab6c39`.
- First complete run: [34697985023](https://github.com/Unjuno/seed-guided-optimization/actions/runs/34697985023).
- Complete evidence artifact: `10298969508`, `rank-tail-tradeoff-evidence`.
- Archive SHA256: `cb26ff91a708bd5eb6e7efa049bbc93721fa3df62d728ffd4f580b79de530cb7`.
- Protocol SHA256: `e27c10bd5643f173632387011a41f703686e2d47f9cbb7b4cb58d47bfccd509a`.
- Machine-readable audit: [rank_tail_independent_audit.json](../results/rank_tail_independent_audit.json).

## H / T: what was tested

The hypothesis was a conjunction: relaxing concentration on the hardest candidates increases mean heldout accuracy AND decreases minimum accuracy over the fixed 80 heldout environments. Both directional paired tests had to pass their prospectively fixed 0.05 thresholds, after manipulation and matching gates. Secondary endpoints cannot replace either primary endpoint.

The unchanged Issue104 selection procedure maximizes the high-minus-low mean candidate-loss rank difference among eligible subsets. Each subset retains the hardest candidate and chooses three of the remaining fifteen. Eligibility matches loss variance, total physical diversity, translation allocation, mean gradient novelty, centered residual spectral effective rank and maximum pairwise opposition. Mean selected loss itself is intentionally not matched; rank and hardness therefore remain a joint intervention, not separately identified causes.

Canonical Digits images: 988 training, 809 nontraining evaluation union. SmallCNN; AdamW learning rate0.005, weight decay0.001; 10epochs, batch128 (last92), 80updates; 16candidate environments, 4adopted. Fresh blocks4100-4159, training environment seeds101000-101063, heldout seeds102000-102079. All60 reference schedules were sealed before intervention training, then all180 final reference/intervention states before heldout construction. No new repetitions were added after viewing the results.

## D: primary results

All accuracy changes below are high-rank minus low-rank and expressed in percentage points, not relative percent improvement.

| Primary endpoint | Mean difference (pp) | Paired SE (pp) | Two-sided95% CI (pp) | Registered one-sided p | Decision |
|---|---:|---:|---|---:|---|
| Mean heldout accuracy | +0.297976 | 0.270092 | [-0.242477,+0.838429] | 0.137202, positive direction | Not supported |
| Minimum heldout accuracy | -0.070045 | 0.427554 | [-0.925578,+0.785487] | 0.435213, negative direction | Not supported |

There were34/60 positive mean contrasts. The minimum was positive in29/60 blocks. The conjunction failed because neither primary directional criterion passed. The intervals permit positive and negative differences; a performance-equivalence test was not registered.

The intended intervention and balance gates did pass: pooled support97.0625%, minimum per-block93.75%, 141 identical-subset fallback steps, and all registered equivalence tests. The mean selected-rank gap was1.656146. A successful intervention with an inconclusive performance contrast is not a failed software execution.

## Exploration retained separately

| Secondary endpoint | Mean difference (pp) | Two-sided95% CI (pp) | Status |
|---|---:|---|---|
| Clean-image accuracy | +3.170581 | [+1.867569,+4.473592] | Exploratory |
| Across-environment accuracy standard deviation | +0.719589 | [+0.333773,+1.105404] | Exploratory |

These observations motivate clean-versus-transformation specificity, not a claim that the preregistered mean/minimum tradeoff succeeded. Increased environment-level SD is not by itself evidence of a reduced minimum or adversarial failure. The same observations must not be reused as an independent confirmation of a new hypothesis. Comparisons with the separately trained loss-hard reference are secondary and may cross training CPUs.

## Independent verification

A standalone audit reconstructed all2,184,000 archived subset summaries into the frozen optimal-pair choices across4,800 steps: 4,659 eligible optimal pairs and141 identical-subset fallbacks. It checked660 source/artifact hashes and180 checkpoint tensor digests, reconstructed60 paired contrasts from14,400 per-environment accuracy rows, and recomputed paired t intervals, equivalence gates and the frozen decision. Maximum pair-contrast, environment-reaggregation and paired-result differences were0.0 in this audit.

This verification is conditional on the archived measurements. It did NOT regenerate raw gradients, run image inference, retrain the networks or establish cross-hardware reproducibility. The recorded general-identity residual was threshold-checked rather than recomputed from unavailable raw signatures.

## C / U: remaining alternatives and uncertainty

The mean rank manipulation also changes actual selected hardness; higher-order loss shape, task alignment and full-network updates can differ despite the registered summaries being balanced. The earlier rank/hardness separation calibrations failed to obtain the required support/manipulation; they do not establish either variable as an isolated cause.

Statistical repetitions are the60 training blocks, not environments, images or steps. All blocks reuse a fixed image split and a fixed heldout environment pool. The minimum is a pool-dependent order statistic. The paired intervals do not include combined uncertainty across datasets, random environment pools or backends. A total combined uncertainty cannot be estimated from this design and is not invented.

Training used Python3.12.14, torch2.10.0+cpu, numpy2.3.5, pandas2.2.3, scipy1.17.0 and scikit-learn1.8.0, one deterministic CPU thread. The intervention runners included AMD EPYC7763/9V45/9V74 and Intel Xeon8573C/6973P-C/8370C; both intervention arms shared a runner within a block. Clocks were uncontrolled. The independent numerical audit used Python3.13.5 and the same listed NumPy/Pandas/SciPy/Torch versions. Training tensors were float32; statistical calculations were float64. No quantized backend or zero-point is involved. No timing benchmark or GPU-speed claim follows.

## Variable and unit table

| Symbol | Meaning (Japanese) | SI unit | Definition | Domain / assumptions | Type |
|---|---|---|---|---|---|
| n | 学習反復数 | 1 | fixed60 independent training blocks conditional on pools | positive integer | integer scalar |
| R | 採用候補の平均損失順位 | 1 | mean of4 loss ranks, largest loss ranked1 | finite; hardest retained | real scalar |
| M | 平均正答率のペア差 | 1 | high-rank minus low-rank average over80 environments | [-1,1] | real scalar per block |
| W | 最小正答率のペア差 | 1 | high-rank minus low-rank minimum over80 environments | [-1,1] | real scalar per block |
| SE | ペア平均差の標準誤差 | 1 | sample SD of block differences divided by square root of60 | conditional design | nonnegative scalar |
| k | 95%区間の被覆係数 | 1 | t59 97.5th percentile,2.0009953781 | two-sided t interval | real scalar |

Unit check: normalized/standardized summaries and accuracy fractions are dimensionless. Percentage-point values equal accuracy-fraction differences multiplied by100; SD differences use the same display conversion. They are not relative improvement percentages.

## Consequence for the roadmap

Issue115 is completed with primary non-replication, not upgraded via clean or SD findings. The original online policy must be tested against simple and sham-gradient controls before attributing an offline-schedule observation to online SGO; see [the separately preregistered online controls](ONLINE_CONTROL_PROTOCOL.md). X announcement remains paused.
