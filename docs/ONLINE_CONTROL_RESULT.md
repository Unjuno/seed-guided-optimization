# Issue121: online gradient-information controls

Completed2026-09-25. **Frozen decision: ONLINE GRADIENT-INFORMATION CONTROL SUPPORT.** All three registered directional mean-accuracy comparisons pass Holm3 adjustment. This identifies a policy-level advantage in the registered setting, not a unique gradient-geometric mediator or a universal optimizer law.

## Registration and provenance

The [full protocol](ONLINE_CONTROL_PROTOCOL.md) was committed at `6c6b73ca75e62844ac681ee6d56dc4d672a91e27` before new outcomes. [Issue121](https://github.com/Unjuno/seed-guided-optimization/issues/121) records the experiment, [PR122](https://github.com/Unjuno/seed-guided-optimization/pull/122) its workflow and results.

First complete run: [36137245500](https://github.com/Unjuno/seed-guided-optimization/actions/runs/36137245500), attempt1, head `247cc216fe0c6bead7ae45aff16d0c1d5b302d82`. Protocol hash `b4331a2f90d58cb7499a575bed13262f672eeec102207edb1a6f464785ef85e4`. Evidence artifact `10864833373`, name `online-information-controls-evidence`, ZIP SHA256 `a52738d402ee4ff870e6826a864665be60377c7dd06fa82f1ed916a39ceecaa9`. The archive contains actual sources, all states, training/evaluation manifests, correct counts and complete CSVs. Artifact retention is90days; the tracked summaries and paired contrasts do not depend on that retention for statistical recalculation.

## H / T: comparison and environment

The question was whether correctly associated gradient information improves the original online policy beyond both loss-hard selection and two simple controls. Unlike the preceding offline mechanism studies, each policy chooses using its own current model state.

| Policy | Definition |
|---|---|
| loss_hard | Four largest current candidate losses |
| gradnov | Existing greedy hardness-plus-gradient-direction selector, novelty weight0.6 |
| anchor_random | Largest-loss candidate plus3 uniformly selected candidates among the other15 |
| sham_gradnov | Same gradnov rule with independently permuted row-and-column correspondence of the gradient Gram matrix to candidate losses |

All methods evaluate16candidate environments and compute normalized head-gradient signatures;4environments contribute to each update. Minibatch and candidate sequences and model initialization are paired within a training block. Every method in a block is trained on the same CPU runner. No claim of equal wall-clock efficiency follows from these statistical comparisons.

Canonical Digits:988training and809nontraining images. Existing SmallCNN, AdamW lr0.005/wd0.001, 10epochs, batch128/last92, 80updates. Exactly60fresh blocks5000-5059, split into three prospectively fixed environment pools of20blocks. Training pools start200000/202000/204000 (64environments each); heldout pools201000/203000/205000 (80each). All240final states were sealed globally before constructing heldout environments. Images are reused; this is not a fresh-dataset experiment.

## D: primary results

All effects are **gradnov minus comparator** in percentage points. Each confidence interval below is a marginal two-sided95% t59 interval; these are not simultaneous intervals. The registered tests were one-sided in the positive direction, with Holm adjustment across all three comparisons.

| Comparator | Mean difference (pp) | Paired SE (pp) | Two-sided95% CI (pp) | Holm-adjusted one-sided p | Positive blocks | Frozen result |
|---|---:|---:|---|---:|---:|---|
| loss_hard | +2.671688 | 0.325786 | [+2.019792,+3.323585] | 3.759003e-11 | 53/60 | Supported |
| anchor_random | +0.677019 | 0.365919 | [-0.055184,+1.409222] | 0.034648 | 39/60 | Supported under the registered one-sided rule |
| sham_gradnov | +2.070277 | 0.317844 | [+1.434274,+2.706281] | 1.791759e-8 | 51/60 | Supported |

The anchor_random comparison is materially less decisive than the other two: its two-sided interval crosses zero. Passing the preregistered one-sided test is not equivalent to a strictly positive two-sided95% interval. No decision threshold was changed after seeing this difference.

The [tracked primary summary](../results/online_controls_summary60.csv) preserves the exact statistics. [Independent audit](../results/online_controls_independent_audit.json) reproduced them without importing the experimental estimator or Holm implementation.

## What was distinguished

The correctly associated online gradient selector outperformed the gradient-shuffled policy in the registered setting. Thus, the benefit is not reproduced merely by retaining a Gram matrix's entry distribution/spectrum while randomizing its association with the actual candidate environments. The comparison with anchored random selection also meets its registered directional criterion, but is smaller and should not be represented as overwhelming or uniform across pools.

The contrast is a total policy effect. The policies can choose different losses, loss ranks, transformations and later update trajectories. Those differences may be mediating pathways; they are not eliminated by this control. In particular, the result does not establish centered spectral effective rank, maximum opposition, raw gradient magnitude or task alignment as the unique cause. Earlier failed calibration/performance decisions remain unchanged.

## Exploration: an accuracy tradeoff, not the primary claim

Against anchor_random, gradnov had the following exploratory differences:

| Endpoint | Mean difference (pp) | Two-sided95% CI (pp) |
|---|---:|---|
| Clean-image accuracy | -7.148743 | [-8.6207,-5.6768] |
| Minimum accuracy over80heldout environments | +3.753605 | [+2.6534,+4.8538] |
| Across-environment accuracy SD | -2.028841 | [-2.3817,-1.6760] |

These suggest a clean-versus-shift/tail tradeoff: uniform relaxation may favor clean accuracy, while informed selection may favor performance on difficult transformations. These endpoints were secondary and unadjusted; the same data are not an independent confirmation of this new explanation. Minimum over80environments is not a worst-case guarantee over all transformations.

Pool-specific mean gradnov-minus-anchor_random estimates were -0.2501pp, +0.8251pp and +1.4560pp. These are descriptive, and no between-pool interaction was registered. Do not claim superiority in every pool. All60blocks use only three fixed environment pools, so population-level uncertainty over arbitrary pools is not measured by the pooled block-level interval.

Descriptive mean selected ranks were3.5319 for gradnov,3.3556 for sham_gradnov,7.0122 for anchor_random and2.5001 for loss_hard. Mean candidate losses and selected physical/gradient properties also evolve differently. These summaries do not establish causal mediation.

## Independent audit and implementation conditions

Independent recomputation verified440source/artifact hashes,240checkpoint tensor digests,19,200per-environment rows and19,200training-selection records. It independently reconstructed the registered candidate/minibatch sequence, checked adopted environment mappings and counts, reconstructed accuracy from integer correct counts over809images, and separately calculated paired t statistics and Holm adjustment. Maximum environment-reaggregation and primary-statistic discrepancies were0.0. The archived experiment source matched committed Git blob `d94e2e8cf47f618751260bdd6438a94ff5349360`.

This audit did not rerun raw-image inference or training, reconstruct raw gradients, or verify each selector's choice from archived full candidate gradients (those raw matrices are not archived). Initialization pairing was checked through common metadata/digests, not by rerunning initialization. Source/runtime records, not successful hashes alone, document the scientific procedure.

Training/evaluation used Python3.12.14, torch2.10.0+cpu, numpy2.3.5, pandas2.2.3, scipy1.17.0, scikit-learn1.8.0, one deterministic CPU thread and MKLDNN enabled. CPU models included AMD EPYC7763/9V45/9V74 and Intel Xeon8573C/6973P-C. Training clock snapshots ranged2.300-4.081GHz and evaluation snapshots2.445-4.284GHz; these were not controlled clocks and are not throughput benchmarks. Training was float32, counts were integers, aggregation float64; no quantization/zero-point assumption. The independent audit used Python3.13.5 with the same listed NumPy/Pandas/SciPy/Torch versions.

## C / U and variable table

Remaining uncertainty includes initialization/training randomness conditional on the fixed pools, reused images, pool-specific effects and cross-backend drift. Paired SE is not a total combined uncertainty over new data, hardware or environmental distributions. Such combined uncertainty is not identifiable from this design and is not fabricated. No GPU speed, universal robustness, adversarial guarantee or unique-mediator claim follows.

| Symbol | Meaning (Japanese) | SI unit | Definition | Domain / assumptions | Type |
|---|---|---|---|---|---|
| n | 学習反復数 | 1 | fixed60 training blocks | three fixed pools | integer scalar |
| K | 候補環境数 | 1 | 16 per update | distinct candidates | integer scalar |
| Q | 採用環境数 | 1 | 4 per update | selected from16 | integer scalar |
| B | 平均正答率のペア差 | 1 | gradnov minus comparator | [-1,1] per block | real scalar |
| SE | ペア平均差の標準誤差 | 1 | sample SD of60block differences divided by square root of60 | fixed design | nonnegative scalar |
| k | 95%区間の被覆係数 | 1 | t59 97.5th percentile,2.0009953781 | marginal two-sided interval | real scalar |
| p | 片側検定p値 | 1 | registered positive-direction paired t probability | [0,1] | probability scalar |

Unit check: normalized/standardized scores and accuracy fractions are dimensionless. Reported percentage-point differences and their SE/intervals are fraction values multiplied by100, not relative percentage improvement.

## Next question

Confirm the exploratory clean/tail tradeoff on new prospectively fixed pools, and separately assess a loss/rank-matched online control before treating the policy-level information advantage as a unique geometric mechanism. Keep X announcement paused until the owner chooses the scope of the public claim.
