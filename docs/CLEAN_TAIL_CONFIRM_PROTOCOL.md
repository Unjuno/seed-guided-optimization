# Prospective confirmation: clean-versus-tail tradeoff of online gradnov

Registered2026-09-25 before computing outcomes for any repetition/environment below. This is a NEW experiment motivated by secondary Issue121 observations. It does not upgrade those observations or rerun Issue115. The first valid complete execution after registration is authoritative; preserve invalid attempts separately.

## H: falsifiable compound claim

Relative to the hardest-plus-random-three online policy, online gradnov increases minimum accuracy over a specified80-environment heldout pool but decreases clean-image accuracy. The proposed tradeoff is a statement about these two policies and the given task/generator, not a proof of a unique mediator or a worst-case guarantee over all transformations.

## T: fixed task, policies and independence unit

Use unchanged canonical Digits images988train/809nontraining; existing SmallCNN; AdamW learning rate0.005, weight decay0.001; 10epochs, batch128/last92,80updates;16distinct candidate environments,4adopted; one deterministic CPU thread. Training and gradient signatures float32, counts integer, statistical aggregation float64. No quantization.

Two policies: existing common.select_hard_gradient_novel with novelty_weight0.6, and anchor_random retaining the current highest-loss candidate plus3uniform other candidates without replacement. Each policy uses its own current model state. Both evaluate16candidates and compute the same form of head-gradient signatures. Sort adopted local indices before backward/update. Within each block share initialization, candidate/minibatch sequence, CPU runner and optimizer specification. The anchor control uses a separate RNG with seed base_seed+102021, matching Issue121's anchor RNG offset. There is no matching claim for realized arm-state loss/rank/physical summaries.

Exactly60fresh training blocks, reps6000-6059, base_seed7500000000+4099*rep. Exactly6fixed fresh environment pools with10blocks each; pool index is integer floor((rep-6000)/10). For pool index0..5, training seeds start300000+2000*pool and contain64consecutive keys; heldout seeds start301000+2000*pool and contain80keys. These pools and blocks do not reuse Issue121's environment keys or repetitions. Images are reused and must not be described as independent new-image validation.

Construct/train only training environments until all120final policy states are globally sealed with file/tensor, source, split, initialization and schedule hashes. Only after verifying the global seal construct heldout environments. No performance-based exclusions, early stopping, extensions, alternate pools or changing the primary endpoints. No performance peeking between shards.

## Primary analysis deliberately uses six pool-level contrasts

For each training block, calculate gradnov-minus-anchor_random differences for (a) minimum accuracy over its pool's80heldout environments and (b) clean accuracy over809images. Average each of these differences across the10training blocks in its pool. These produce6pool-level paired contrasts per endpoint.

The PRIMARY tests use the six pool-level contrasts (t distribution with5degrees of freedom), not60blocks or4800environment evaluations. This conservative aggregation avoids presenting60model repetitions as60independent environment pools. Pools are six prospectively fixed draws from the implemented generator; extrapolation to arbitrary new generators/datasets remains outside scope. Report the6raw pool contrasts, pool-level SE and two-sided95% t5 intervals. Report all60raw block differences too, but block-level intervals are secondary and conditional on these pools.

## D: frozen decisions

Both conditions are required for CLEAN-TAIL POLICY TRADEOFF SUPPORT:
1. mean pool-level minimum-accuracy contrast is positive AND its one-sided paired t p in the positive direction is below0.05;
2. mean pool-level clean-accuracy contrast is negative AND its one-sided paired t p in the negative direction is below0.05.

This is a conjunctive claim: both component nulls must be rejected. Individual component results must be clearly labeled; no separate unadjusted familywise claim is made for any collection of secondary endpoints.

Labels:
- both components pass: CLEAN-TAIL POLICY TRADEOFF SUPPORT;
- only minimum component passes: TAIL ADVANTAGE ONLY / CLEAN COST NOT CONFIRMED;
- only clean component passes: CLEAN COST ONLY / TAIL ADVANTAGE NOT CONFIRMED;
- neither passes: NO CLEAN-TAIL POLICY TRADEOFF CONFIRMATION;
- malformed/nonfinite/state/source/hash/grid/boundary failure: INVALID / EXECUTION FAILURE.

A zero standard-error endpoint is a declared degenerate execution/analysis condition to inspect, not a reason to fabricate a p-value. Non-significance is not equality or safety. The mean heldout-accuracy contrast, p10, across-environment SD, training-loss/rank diagnostics and all within-pool/block-level inferential tests are SECONDARY and cannot change the frozen decision. No performance-equivalence or noninferiority margin is registered.

## C: opposing explanations

A confirmed tradeoff can still reflect different hardness exposure, transformation allocation, gradient-task correspondence, later trajectory divergence or higher-order interactions. It does not separately identify span, opposition or raw update magnitude. Failure of a component weakens this registered compound explanation without proving the corresponding true effect is zero. The minimum depends strongly on the80environment sample and is not a universal adversarial endpoint.

## U: measurement and uncertainty

There are60training blocks but only6environment pools for primary inference. Six is still a small number of pools; report t5 intervals and all raw pool contrasts transparently. The fixed reused image set, generator family and backend are additional uncertainty sources not represented by that interval. Total combined uncertainty across new datasets/hardware is not estimable from this design; do not substitute the paired SE for it.

Use pinned scientific versions matching Issue121: Python3.12, torch2.10.0+cpu, numpy2.3.5, pandas2.2.3, scipy1.17.0, sklearn1.8.0. Archive actual CPU, uncontrolled clock snapshot, thread flags, MKLDNN status and source hashes. No GPU/equal-wall-clock claim. Independent audit must check120state tensor digests,9600environment-count rows,60paired block contrasts,6pool contrasts and the two frozen tests; raw-image inference/retraining is a separate audit and not implied.

## Variables and unit check

| Symbol | Meaning (Japanese) | SI unit | Definition | Domain / assumptions | Type |
|---|---|---|---|---|---|
| rep | 学習ブロック識別番号 | 1 | 6000..6059 | unique integer | integer scalar |
| pool | 変換環境群の識別番号 | 1 | floor((rep-6000)/10) | integer0..5 | integer scalar |
| base_seed | 初期化・順序の再現キー | 1 | 7500000000+4099*rep | nonnegative integer, not quality score | integer scalar |
| W | 最低正答率のペア差 | 1 | gradnov minimum minus anchor minimum over80environments | [-1,1] per block; averaged per pool | real scalar |
| C | clean正答率のペア差 | 1 | gradnov clean minus anchor clean | [-1,1] per block; averaged per pool | real scalar |
| n_pool | 主解析の環境群数 | 1 | fixed6 | not60training blocks | integer scalar |
| SE | 環境群平均差の標準誤差 | 1 | sample SD of6pool contrasts divided by square root of6 | primary pool-level design | nonnegative scalar |
| k | 95%区間の被覆係数 | 1 | t5 97.5th percentile | two-sided95% interval | real scalar |
| p | 主検定の片側p値 | 1 | positive tail for W, negative tail for C | [0,1] | probability scalar |

Unit check: all normalized/standardized quantities and accuracy fractions are dimensionless. Percentage-point reporting multiplies a fraction difference and its interval/SE by100, not by a baseline-dependent relative-improvement factor.
