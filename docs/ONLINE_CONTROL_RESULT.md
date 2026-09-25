# Issue121: online gradient-information controls

Completed2026-09-25. **Frozen decision: ONLINE GRADIENT-INFORMATION CONTROL SUPPORT.** All three registered directional mean-accuracy comparisons pass Holm3 adjustment. This establishes a policy-level advantage in the registered setting, not a unique gradient-geometric mediator or a universal optimizer law.

## Registration and provenance

The [full protocol](ONLINE_CONTROL_PROTOCOL.md) was committed at `6c6b73ca75e62844ac681ee6d56dc4d672a91e27` before new outcomes. [Issue121](https://github.com/Unjuno/seed-guided-optimization/issues/121) records the experiment; [PR122](https://github.com/Unjuno/seed-guided-optimization/pull/122) its workflow and results.

First complete run: [36137245500](https://github.com/Unjuno/seed-guided-optimization/actions/runs/36137245500), attempt1, head `247cc216fe0c6bead7ae45aff16d0c1d5b302d82`. Protocol hash `b4331a2f90d58cb7499a575bed13262f672eeec102207edb1a6f464785ef85e4`. Evidence artifact `10864833373`, name `online-information-controls-evidence`, ZIP SHA256 `a52738d402ee4ff870e6826a864665be60377c7dd06fa82f1ed916a39ceecaa9`. The archive contains sources, states, training/evaluation manifests, correct counts and full CSVs. Retention is90days. The tracked [primary summary](../results/online_controls_summary60.csv), [all60 lossless contrasts](../results/online_controls_contrasts60.csv), and [independent audit](../results/online_controls_independent_audit.json) preserve the primary statistical evidence separately.

## H / T: comparison and environment

The question was whether correctly associated gradient information improves the original online policy beyond loss-hard and two controls. Each policy selects using its own current model state, unlike the preceding offline schedule interventions.

| Policy | Definition |
|---|---|
| loss_hard | Four largest current candidate losses |
| gradnov | Existing greedy hardness-plus-gradient-direction selector, novelty weight0.6 |
| anchor_random | Largest-loss candidate plus3 uniformly selected candidates among the other15 |
| sham_gradnov | Same gradnov rule with randomly permuted row-and-column correspondence of the gradient Gram matrix to candidate losses |

All methods evaluate16candidate environments and compute normalized head-gradient signatures;4contribute to each update. Minibatch/candidate sequences and model initialization are paired within a block. Every method in a block trains on the same CPU runner. Equal statistical budgets do not establish equal wall-clock efficiency.

Canonical Digits:988training and809nontraining images. SmallCNN, AdamW lr0.005/wd0.001, 10epochs, batch128/last92, 80updates. Exactly60fresh blocks5000-5059 across three fixed pools of20blocks. Training pools start200000/202000/204000 with64environments each; heldout pools201000/203000/205000 with80each. All240final states were globally sealed before any heldout construction. Images are reused; this is not a fresh-dataset study.

## D: primary results

Effects are **gradnov minus comparator**, in percentage points. Intervals are marginal two-sided95% t59 intervals, not simultaneous intervals. Tests were preregistered one-sided in the positive direction with Holm adjustment across three comparisons.

| Comparator | Mean difference (pp) | Paired SE (pp) | Two-sided95% CI (pp) | Holm-adjusted one-sided p | Positive blocks | Frozen result |
|---|---:|---:|---|---:|---:|---|
| loss_hard | +2.671688 | 0.325786 | [+2.019792,+3.323585] | 3.759003e-11 | 53/60 | Supported |
| anchor_random | +0.677019 | 0.365919 | [-0.055184,+1.409222] | 0.034648 | 39/60 | Supported under registered one-sided rule |
| sham_gradnov | +2.070277 | 0.317844 | [+1.434274,+2.706281] | 1.791759e-8 | 51/60 | Supported |

The anchor_random comparison is less decisive: its two-sided interval crosses zero. Passing the preregistered one-sided test is not equivalent to a strictly positive two-sided95% interval. No threshold was changed after seeing the results.

## What was distinguished

Correctly associated online gradient selection outperformed the gradient-shuffled policy. Its benefit was not reproduced merely by retaining a Gram matrix's entry distribution/spectrum while randomizing correspondence with actual environments. The anchored-random contrast also passes its registered directional criterion, but is smaller and not uniformly positive across pools.

This is a total policy effect. Policies can select different losses, loss ranks and transformations and follow different update trajectories. Those pathways are not removed by this control. The result does not establish centered effective rank, maximum opposition, raw gradient magnitude or task alignment as the unique cause. Earlier negative and infeasible experiments remain unchanged.

## Exploration: clean-versus-shift/tail tradeoff

Against anchor_random, gradnov showed:

| Secondary endpoint | Mean difference (pp) | Two-sided95% CI (pp) |
|---|---:|---|
| Clean-image accuracy | -7.1487 | [-8.6207,-5.6768] |
| Minimum accuracy over80heldout environments | +3.7536 | [+2.6534,+4.8538] |
| Across-environment accuracy SD | -2.0288 | [-2.3817,-1.6760] |

These suggest uniform relaxation may favor clean accuracy while informed selection favors performance on difficult transformations. These endpoints were secondary and unadjusted. They are not an independent confirmation of a newly named explanation. Minimum over80environments is not a worst-case guarantee over all transformations.

Pool-specific mean gradnov-minus-anchor_random estimates were -0.2501pp, +0.8251pp and +1.4560pp. They are descriptive; no between-pool interaction was registered. Do not claim superiority in every pool or estimate general random-pool uncertainty from the pooled block-level interval.

Descriptive mean selected ranks were3.5319 for gradnov,3.3556 for sham_gradnov,7.0122 for anchor_random and2.5001 for loss_hard. Candidate losses and physical/gradient properties also evolve differently. These summaries do not identify causal mediation.

## Independent audit and implementation conditions

An independent implementation verified440source/artifact hashes,240checkpoint tensor digests,19,200per-environment rows and19,200training-selection records. It reconstructed registered candidate/minibatch sequences, checked adopted-environment mappings and counts, reconstructed accuracy from integer correct counts over809images, and separately computed paired t statistics and Holm adjustment. Maximum environment-reaggregation and primary-statistic discrepancies were0.0. Archived experiment source matched committed Git blob `d94e2e8cf47f618751260bdd6438a94ff5349360`.

The audit did not rerun raw-image inference or training, reconstruct raw gradients, or verify each selector choice against complete candidate gradients, which were not archived. Initialization pairing was checked via metadata/digests, not rerun. A hash match alone is not a scientific validation of the underlying raw measurements.

Training/evaluation used Python3.12.14, torch2.10.0+cpu, numpy2.3.5, pandas2.2.3, scipy1.17.0, scikit-learn1.8.0, one deterministic CPU thread, MKLDNN enabled. CPUs included AMD EPYC7763/9V45/9V74 and Intel Xeon8573C/6973P-C. Training clock snapshots ranged2.300-4.081GHz and evaluation2.445-4.284GHz; clocks were uncontrolled and these are not timing benchmarks. Training was float32, counts integer, aggregation float64; no quantization or zero-point assumption. Independent audit used Python3.13.5 with the same listed NumPy/Pandas/SciPy/Torch versions.

## C / U and variable table

Uncertainty includes training randomness conditional on fixed pools, reused images, pool-specific effects and backend drift. Paired SE is not total combined uncertainty over datasets, hardware or environmental distributions. Such combined uncertainty is not identifiable from this design. No GPU-speed, universal-robustness, adversarial-guarantee or unique-mediator claim follows.

| Symbol | Meaning (Japanese) | SI unit | Definition | Domain / assumptions | Type |
|---|---|---|---|---|---|
| n | 学習反復数 | 1 | fixed60 training blocks | three fixed pools | integer scalar |
| K | 候補環境数 | 1 | 16 per update | distinct candidates | integer scalar |
| Q | 採用環境数 | 1 | 4 per update | selected from16 | integer scalar |
| B | 平均正答率のペア差 | 1 | gradnov minus comparator | [-1,1] per block | real scalar |
| SE | ペア平均差の標準誤差 | 1 | sample SD of60differences divided by square root of60 | fixed design | nonnegative scalar |
| k | 95%区間の被覆係数 | 1 | t59 97.5th percentile,2.0009953781 | marginal two-sided interval | real scalar |
| p | 片側検定p値 | 1 | positive-direction paired t probability | [0,1] | probability scalar |

Unit check: scores and accuracy fractions are dimensionless. Percentage-point differences, SE and intervals are fractions multiplied by100, not relative percentage improvement.

## Next question

Confirm the exploratory clean/tail tradeoff using prospectively fixed fresh pools, and separately examine loss/rank-matched online controls before treating policy specificity as a unique geometric mechanism. X announcement remains paused.
