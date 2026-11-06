# Optimizer switching-time boundary — 2026-11-07

**First valid prospectively registered fresh-seed study:** [Issue #131](https://github.com/Unjuno/seed-guided-optimization/issues/131). Original conditions and source SHA256 preceded training/outcomes. A v1 tuple-versus-JSON-list preflight failure yielded **zero trained states / zero heldout observations**. The v2 serialization-only correction SHA was added to the Issue **before the first valid run**, with the old failure archived. This result writeup is made after outcomes, NOT a new prospective registration.

## H/T: policy experiment, not mathematical state-space optima

Historical Digits MLP, **800 training and 809 REUSED test images**, 12 NEW independent transformation pools ×4 NEW initial-weight keys each ×4 fixed switching policies = **192 final trajectories**, 200 applied optimizer steps each. Same initial weights, minibatch order and transformations for all policies per block. Hidden 128×64 weight: M always Muon; T80/T100/T120 Muon through step80/100/120 then AdamW. Head and biases AdamW throughout. Every policy maintains and computes **both hidden shadow momentum histories from the same realized gradient**; only one proposal updates real weights, so no optimizer-state reset, but no single-optimizer cost claim. All 192 states hashed/sealed **before** constructing heldout transforms. Exactly 9,216 environment CE/count rows and 192 clean rows.

## D: three PREDEFINED positive T100 contrasts, Holm3 one-sided t11

CE difference = CE(baseline) minus CE(T100), positive means T100 reduces loss. Registered SUPPORT: Holm-adjusted p<.05 and point mean gain ≥.01CE; t-test null is zero, not .01.

| Contrast | Mean CE gain | 95% pointwise CI | Holm3 one-sided p | Frozen result |
|---|---:|---|---:|---|
| T100 vs M | **+0.025832** | [+0.001760,+0.049905] | 0.037690 | **SUPPORT** |
| T100 vs T80 | **−0.010962** | [−0.015430,−0.006494] | 0.999892 | **NOT SUPPORTED** |
| T100 vs T120 | **+0.013302** | [+0.008738,+0.017867] | 0.00007473 | **SUPPORT** |

**The registered T100 peak among 80/100/120 FAILS** because T100 is not superior to T80. The reverse CE difference T80 versus T100 was positive in **12/12 pools**; do not retroactively re-register that direction. An exact best switching time remains unmeasured, as no times outside 80/100/120 were tested.

[Three primary CSVs](../results/switch_time_20261107_primary.csv), [36 paired pool differences](../results/switch_time_20261107_pool_differences.csv) and [means of four arms](../results/switch_time_20261107_policy_means.csv). Read-only statistics-only checker: scripts/check_switch_time_boundary_20261107.py. Summary CSV check is NOT raw model/audit validation.

## Two OBJECTIVES, different preferred switches (POST-OUTCOME secondary)

| Policy | Shifted CE (48 transforms) | Shifted accuracy | Clean CE | Clean accuracy |
|---|---:|---:|---:|---:|
| **T80** | **1.177579** | 60.78% | 0.855915 | 68.03% |
| T100 | 1.188541 | 60.41% | 0.837724 | 68.70% |
| T120 | 1.201843 | 59.98% | 0.826227 | 69.21% |
| **M** | 1.214373 | 59.92% | **0.804520** | **70.07%** |

For these sample means, improving performance over transformed images has a cost on clean images. No single policy dominates both CE criteria.

Let w in [0,1] be the fraction or fixed objective weight for transformed CE, and 1−w for clean CE. Weighted objective J_j(w)=w S_j+(1−w) C_j. **The sample-mean-only posthoc lower envelope** is M when w<**0.562438**, T100 when **0.562438<w<0.623979**, T80 when w>**0.623979** (ties at boundaries). T120 is Pareto nondominated but not on this linear lower envelope.

**Algebra:** For a,b, solve (1−w)(C_a−C_b)+w(S_a−S_b)=0 for w=(C_a−C_b)/[(C_a−C_b)−(S_a−S_b)] (nonzero denominator). Substitute observed mean M and T100→0.5624375319, T100 and T80→0.6239793030, compare remaining pairwise lines. This algebra is exact for the observed means, **NOT a general optimal switching law or validation of a dynamic controller**.

12-pool bootstrap, 10,000 resamples, w=.6: winner M 36.80%, T100 35.63%, T80 22.68%, T120 4.89%. Thresholds are highly uncertain even conditional on the reused Digits image split. At w=.8 T80 wins 96.88% of resamples, but still no new-image generalization.

## Distinct old trial: short-horizon versus future long-horizon boundary

Historical 2026-10-31 matched step100 states: final test CE favors AdamW switch in **44/48**, a 5-step threshold chose switch only **17/48**, missing **27** states. At the SAME long horizon, control images favor A in33/48 whereas final test favors A in44/48: **11 sign disagreements**. This was post-outcome diagnostic, not a registered new trial. It establishes that a scalar five-step improvement is not a reliable proxy for the current study's long-horizon objective under this protocol.

## C/U — exact claim boundary

**Established only conditionally:** registered policy outcome for fixed Muon/AdamW settings and 12 new transform pools, reused historic Digits test images. The experiment does **not** isolate polar orthogonalization (learning rates and momentum/decay differ), prove an optimizer-favorable parameter-space region, identify the best time before80/after120, show learned per-seed adaptive switching, or demonstrate single-optimizer/GPU efficiency. Both hidden shadow proposals are evaluated every step.

The paired SE/CI captures observed between-pool variation, not combined uncertainty across new images/tasks/backends. Sources: [prospective Issue131](https://github.com/Unjuno/seed-guided-optimization/issues/131) with all source hashes and failure records. Owner review/full evidence ZIPs are kept in the originating ChatGPT conversation and are **not yet mirrored into GitHub**.

## ERROR CHECK and implementation conditions

Python3.13.5, torch2.10.0+cpu, numpy2.3.5, scipy1.17.0, sklearn1.8.0, CPU Intel Xeon Platinum8573C, 1 CPU thread/worker, clock uncontrolled, float32 weights/gradients, Muon NS bfloat16. Source v2 SHA256 ad93e470d293b762246ed5229bccb2548bbd50485a170ee2d20da6d7c1b806d5, frozen protocol SHA256 00e318caf80c5f1afde8560c1d9529e94808c158dc61e7c0743ded68e3110aaf, freeze file SHA256 677a1277c13582959cf929ef74ce4c34f9e89fcdd6134614c5e852577d8134ea, global seal SHA256 785c40eb9d1ac04e9de91520d152810ff3b106df125b09bff0d2b0dce1e64fd4.

Independent CPU audit: **192 tensor/file digests, 38,400 applied updates, 9,216 heldout count+CE rows, 192 clean counts, 3 t11/Holm3 tests, 26,880 exact shared-prefix comparisons**, three bitwise state+optimizer-history retrain reproductions and two separate raw-image re-inferences PASS. Original v1 preflight failure retained, zero outcomes. Full evidence ZIP SHA256 2b6ab7d799a9b8f81541703cfe48c33f0339f9d2241822be61425517b6dfdcf8; review ZIP 16a4e14bb15685b40141f1467d065956851128d42a0db881f368d717b74dacdc.

## Variable table and unit check

| Symbol | Japanese meaning | SI | Definition | Range/assumptions | Type |
|---|---|---|---|---|---|
| t,s | 現在更新数と固定切替番号 | 1 | s∈{80,100,120,none}, t=1..200 | integers, fixed | integer scalars |
| S_j,C_j | 変換環境CE、clean CE | 1 | heldout CE means by policy j | nonnegative real | real scalars |
| w | 変換CEの目的重み | 1 | transformed weight, clean weight1−w | [0,1] | real scalar |
| J_j | 混合CE目的 | 1 | w S_j+(1−w)C_j | nonnegative | real scalar |
| Δ | 対照−対象CE | 1 | positive is loss improvement | real | real scalar |
| n_pool | 独立変換環境群数 | 1 | 12 | integer | integer scalar |
| SE | 群差の標準誤差 | 1 | 12 pool differences sample SD/√12 | finite nonnegative | real scalar |
| u_c | データ/装置も含めた合成不確かさ | 1 | not estimable here | unavailable | undefined |

CE, differences, weights and standard errors are dimensionless (SI1). A 0.01 CE change is NOT a 1-percentage-point accuracy change. Seconds/FLOPs/energy require separate units and cannot be conflated with CE.

**Next:** test a small number of earlier switching times (for example60) on fresh environments and genuinely new test images before claiming a universal time boundary; predefine a performance criterion and clean-performance guardrail. Then test a state/history-aware long-horizon predictor against M/T80 and fixed-cost controls. No X publicity.