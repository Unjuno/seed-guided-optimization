# Prospective online gradient-information control experiment

Registration date: 2026-09-25. No outcome from the repetitions or environment pools below has been computed before this file is committed. This is a new experiment, not a relabeling, extension or rescue of Issues 100, 104 or 115.

## Motivation / H

Issue100's centered-span intervention did not support its registered performance hypothesis. Issue104 supported a rank/hardness-relaxation schedule effect, but Issue115's fresh n=60 mean/tail replication did not meet either primary significance criterion. Offline schedules and original online gradnov are different treatments. We now compare online policies directly instead of treating reference-trajectory scalar matching as unique causal identification.

Hypothesis: original online gradnov improves heldout mean accuracy relative to loss-hard and to each of two simple controls. The controls test alternatives to the usefulness of correctly associated gradient directions. A positive policy contrast is not unique-mediator identification: selected hardness, rank and actual update trajectories may differ.

## Fixed task and implementation

Canonical Digits images, existing 988 training / 809 nontraining union, existing SmallCNN, unchanged common.py and environment generator. AdamW lr 0.005, weight_decay 0.001; 10 epochs, batch128 (last batch92), 80 updates. K=16 distinct candidates per step; Q=4 adopted. All policies evaluate the same K images and compute normalized head-gradient signatures, even when unused. Candidate sequence, minibatches and initial model are paired within a repetition. Every policy of a repetition is trained on the same one-thread CPU runner. No GPU, equal-wall-clock or general hardware efficiency claim.

Policies, evaluated on EACH policy's current model state:
1. loss_hard: existing common.select_loss_hard, top4 current loss.
2. gradnov: existing common.select_hard_gradient_novel, novelty_weight=0.6.
3. anchor_random: retain the largest-loss candidate; select 3 of the other15 uniformly without replacement with a separate deterministic RNG.
4. sham_gradnov: use the existing gradnov rule, but independently permute rows AND columns of the 16x16 gradient Gram matrix before associating it with candidate losses. This preserves the matrix's spectrum and multiset of entries, not its correspondence with the actual environments. A separate RNG is used; no data-dependent permutation rejection.

All selected indices are sorted before identical candidate-tensor backward/update. Matching of arm-state loss/rank summaries is NOT assumed or required; those summaries are descriptive.

## T: samples, pools and stopping

Exactly 60 fresh training blocks: reps5000-5059. Base seed = 6500000000 +4099*rep. Pool index is floor((rep-5000)/20), giving three fixed pools of20 blocks.

- Pool0: train seeds200000-200063; heldout seeds201000-201079.
- Pool1: train seeds202000-202063; heldout seeds203000-203079.
- Pool2: train seeds204000-204063; heldout seeds205000-205079.

Train all240 final model states and seal file/tensor/source/split/schedule digests before constructing any heldout environment. Only after the global seal evaluate mean, SD, p10, min over80 environments and clean accuracy. No interim performance inspection, optional stopping, additional blocks or outcome-dependent exclusions. Infrastructure repairs are allowed only with explicit invalid-attempt records and no changes to scientific rules; retain unsuccessful attempts.

The first fully valid complete GitHub Actions run after registration is authoritative. Stage boundaries and complete grids are mandatory. Local pre-run checks use synthetic data only.

## D: frozen decision

Three confirmatory contrasts per block: gradnov minus loss_hard, gradnov minus anchor_random, gradnov minus sham_gradnov, all for heldout mean accuracy. Calculate one-sided paired t tests in the positive direction using60 independent training blocks conditional on the fixed image/pool design. Apply Holm step-down familywise adjustment across these THREE tests at alpha0.05. Report raw and adjusted p, mean, paired SE, two-sided95% t59 CI, positive count and the paired raw fractions. Do not count environments/steps as independent blocks.

A contrast is supported only when its mean is positive and Holm-adjusted p<0.05.
- all three supported: ONLINE GRADIENT-INFORMATION CONTROL SUPPORT;
- gradnov versus loss_hard supported but either extra control not supported: ONLINE EFFECT WITHOUT FULL CONTROL SEPARATION;
- no loss_hard superiority: ONLINE BASELINE SUPERIORITY NOT CONFIRMED; still report the other adjusted contrasts;
- malformed grids, source/hash/state/nonfinite mismatch or boundary violation: INVALID / EXECUTION FAILURE, not a negative scientific result.

Pool-specific results, clean/p10/min/SD, anchor_random versus loss_hard, selected-loss rank and gradient summaries are descriptive only. They cannot upgrade the frozen primary decision. No performance equivalence margin is registered: nonsignificance is not equality or proof of safety.

## C: alternative explanations

Any benefit over anchor_random could involve residual differences in chosen hardness rather than gradient novelty alone. Sham changes the loss/geometry correspondence and can change selected hardness too. Training-state divergence, class alignment, raw full-network gradients and optimizer moments remain unisolated. The experiment measures the policy's practical statistical contribution in the original online setting, not a single causal mediator.

## U: uncertainty and audit

Three pools address one particular environment-pool dependence but do not establish generalization to all possible pools, datasets or architectures. n=60 intervals are conditional on the three specified pools and reused images. Paired SE is not combined uncertainty across datasets/backends. No meaningful total combined standard uncertainty is identifiable from this design; report that limitation, rather than inventing u_c. CI coverage factor is the t59 97.5th percentile. CPU model, observed clock snapshot, backend, dtypes, package versions and thread counts must be archived; clocks are not controlled. Training uses float32 and statistical aggregation float64; there is no quantization or zero-point assumption.

Independent verification must reconstruct all240 model-state tensor digests, 19200 per-environment rows, 60 paired contrasts and Holm decisions from the archive. Raw-image retraining is a distinct audit and must not be claimed unless run.

## Variable table

| Symbol | Japanese meaning | SI unit | Definition | Domain / assumptions | Type |
|---|---|---|---|---|---|
| K | 候補数 | 1 | 16 candidates per update | positive integer | integer scalar |
| Q | 採用数 | 1 | 4 selected candidates | 1..K | integer scalar |
| n | 学習反復数 | 1 | 60 training blocks | fixed; conditional pools | integer scalar |
| rep | 学習反復の識別番号 | 1 | 5000..5059 | identifier, not seed quality | integer scalar |
| G | 勾配Gram行列 | 1 | dot products of normalized head signatures | finite 16x16; float32 | real matrix |
| B | ペア正答率差 | 1 | gradnov accuracy minus comparator accuracy | [-1,1]; average over80 environments | real scalar per block |
| p | 片側検定p値 | 1 | upper tail of t statistic under zero mean | [0,1] | probability scalar |
| SE | ペア平均差の標準誤差 | 1 | sample SD of60 differences divided by square root of60 | conditional design | real scalar |
| k | 区間の被覆係数 | 1 | t59 97.5th percentile | 95% two-sided CI | real scalar |

Dimensional check: losses, normalized gradient dot products and accuracy fractions are dimensionless. All comparisons and intervals compare like units. Displayed pp values are accuracy fractions multiplied by100; relative percent improvement is different.
