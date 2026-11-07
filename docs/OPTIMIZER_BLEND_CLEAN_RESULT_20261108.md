# Frozen optimizer-update blending after 80 Muon updates (2026-11-08)

**Status: prospectively registered before training in [Issue135](https://github.com/Unjuno/seed-guided-optimization/issues/135).** Four tests and their Holm4 family, exact data/initialization keys, code SHA256 and clean noninferiority margin were fixed before any new outcome. A human transcription typo in the old unchanged `legacy_study.py` SHA was **corrected by comment at zero trained states**, before freeze. Source/protocol were frozen and 192 trained states were globally SHA256 sealed **before heldout evaluation**. This report was authored after results; its commit is not a new preregistration.

## H / T: what was tested

Following Issue133's failed clean-CE noninferiority for full Muon→AdamW switching, we tested **hidden-weight proposal interpolation**, not a learned per-seed policy. All four models in each (pool,init) block share 80 initial Muon updates. Then for 120 remaining steps:
- **M** remains pure Muon (0% AdamW).
- **B25** applies 75% Muon and 25% AdamW hidden *weight proposal*.
- **B50** applies 50% Muon and 50% AdamW hidden proposal.
- **T80** applies 100% AdamW (hard-switch reference).

Each step computes a **single true gradient**, evolves both hidden AdamW and Muon optimizer shadow moments from that realized gradient and combines the two proposed hidden weights. Head/bias always use AdamW. Alpha0 and alpha1 are bitwise identical to the tested original pure-rules engine under the unit tests. This tests a convex combination of *updates* in fixed model coordinates; it does **not** interpolate the raw gradients, identify loss-basin boundaries or isolate the true mechanism.

Exactly **12 new independent transformation pools ×4 new per-pool initialization keys ×4 policies=192 final states, 38,400 applied gradient updates**. Digits MLP128 hidden neurons; historical **800 fit and 809 REUSED evaluation images**, new initialization/transform key schedules. Four training environments per update, batch100,200 applied steps. All hidden proposals computed for each policy every step; no extra lookahead. Fixed CPU per-worker one thread, PyTorch2.10.0+cpu (Muon NS internal bfloat16), weight/grad float32, statistics float64. No equal-production-FLOP or GPU efficiency claim. All192 final model+optimizer state files sealed before construction of48 fresh heldout transforms/pool. 9,216 heldout CE/count rows plus192 clean count rows. Inferential unit **12 pool-mean differences**, not48 initialization models or9,216 image-environments.

## D: four registered components, Holm4 one-sided t11 family

**Shifted CE improvement** = CE(M)−CE(blend), positive good, requires Holm4 p<.05 and sample point gain≥.010 CE. **Clean noninferiority** = CE_clean(blend)−CE_clean(M), requires Holm4 one-sided test H0:true penalty≥**+.05 CE**, alternative penalty<+.05, and point mean penalty<.05. Each blend's **joint clean-safe shift SUPPORT** requires both its shifted and clean components PASS. Confidence intervals are pointwise two-sided95% t11, not familywise simultaneous.

| Registered contrast | 12-pool mean CE diff | 95% pointwise CI | Holm4 p | Frozen decision |
|---|---:|---|---:|---|
| B25−M shifted benefit | **+0.0310067** | [+0.0258464,+0.0361670] | **8.51e−8** | **SUPPORT** |
| B25−M clean penalty | **−0.0147804** | [−0.0478314,+0.0182705] | **0.00122698** | **NONINFERIORITY SUPPORT** |
| B50−M shifted benefit | **+0.0431939** | [+0.0346045,+0.0517833] | **3.99e−7** | **SUPPORT** |
| B50−M clean penalty | **−0.0120903** | [−0.0658162,+0.0416357] | **0.0136511** | **NONINFERIORITY SUPPORT** |

**Both registered conjunctive clean-safe shifted-CE improvement claims PASS, for the average over the 12 sampled environment pools only.** This is NOT proof of strictly lower true clean CE: the two clean confidence intervals cross zero. It is also NOT a per-environment/per-seed safety guarantee. Individual clean penalty **exceeded the +.05 margin in 1/12 B25 pools and 3/12 B50 pools**; the largest B50 pool penalty was **+.156238 CE**. The causal interpretation is tightly conditional.

[Published four tests](../results/optimizer_blend_20261108_primary.csv) · [48 pool effects](../results/optimizer_blend_20261108_pool_effects.csv) · [four-method means](../results/optimizer_blend_20261108_policy_means.csv). `python scripts/check_optimizer_blend_20261108.py` independently recomputes the **statistical summaries only**. It does NOT re-train or verify checkpoint SHA256. A post-hoc 2^12 sign-flip/Holm4 sensitivity analysis also passes all four under a symmetry assumption, but is *not* a second registered analysis.

## Secondary observations — not a new confirmatory family

| Policy | Mean transformed CE | Transformed accuracy | Mean clean CE | Clean accuracy |
|---|---:|---:|---:|---:|
| M | 1.177445 | 61.856% | 0.873523 | 67.540% |
| **B25** | 1.146439 | 62.703% | **0.858742** | **67.944%** |
| **B50** | **1.134251** | **62.889%** | 0.861432 | 67.903% |
| T80 hard switch | 1.147066 | 62.009% | 0.896167 | 66.553% |

In this *sample*, B50 has lower CE than T80 on both shifted and clean sets. **B50−T80 was not a registered primary test; do not promote these descriptive means to a general superiority claim.** Predeclared descriptive objective w=.8 shifted CE and .2 clean CE yields means M1.116661, B25 1.088899, B50 1.079688, T80 1.096887; no confirmatory mixed-objective test was registered.

### Post-hoc update norm / competing mechanism

Using saved actual Muon/AdamW shadow proposal norm and cosine fields, reconstruct the norm of the applied blended hidden update for the 120 post-80 updates per model, average over 5,760 model-steps/policy:

| Policy | Mean applied hidden Frobenius update norm | Mean Muon proposal norm | Mean AdamW proposal norm | Mean proposal cosine |
|---|---:|---:|---:|---:|
| M | 0.08248 | 0.08248 | 0.16042 | 0.36767 |
| B25 | 0.08290 | 0.08285 | 0.14457 | 0.37484 |
| B50 | 0.09256 | 0.08297 | 0.13677 | 0.37574 |
| T80 | 0.12836 | 0.08300 | 0.12836 | 0.37372 |

**An equally plausible explanation is update-norm rescaling/smoothing**, not the existence of a special Muon-vs-AdamW position region. These are observational comparisons of trajectories that diverge after80. Do not treat a post-treatment norm correlation as a unique causal mediator. Next preregistered experiment should norm-match proposals using **development-only** signals, then verify on fresh heldout environments/images.

**Algebra:** if both candidate updates are computed from the same current weight/gradient, the applied update at alpha is their affine mixture; by expanding its Frobenius inner product with itself, squared norm is (1−alpha)^2 normM^2+alpha^2 normA^2+2alpha(1−alpha) normM normA cosine. This identity is exact in the implementation's float32 coordinates except rounding; empirical comparisons across subsequent training states do not hold the gradients fixed. Variable definitions, dimensions and numerical audit are in the owner's full local [study report](https://github.com/Unjuno/seed-guided-optimization/issues/135) and complete evidence ZIP.

## ERROR CHECK — incident and verification

- Pre-outcome issue contains fixed source hashes and **0 trained / 0 heldout** freeze checkpoint. `freeze.json` SHA256 `4b72eb746cf9d353456f37619dc256d9f0593dfe019012a1946811e84ae9db12`; global seal (created before heldout) SHA256 `1d548918c581b701bfdc4887f0040cee2e6a7a0dc5cfd286b1d4272df7e2e25b`. Frozen runner SHA256 `de2d3608c6b3cc3d1485646c860d00a32f462ab6c00d78508e47d9904d503f62`.
- First **analysis output** command raised `NameError: statistics is not defined` while generating the final JSON. Training/evaluation had completed and were unaffected. Preserved original frozen runner source (not silently edited); used a separate independent auditor to compute the final summary. Original partial four-primary CSV agrees with independent auditor to within numerical tolerance. No change to H/T/D, no extra models, no data-driven exclusion or thresholds.
- Independent evidence audit: **192 model checkpoint file+tensor SHA, 38,400 exactly correct alpha updates, 23,040 identical first-80-prefix pair-step comparisons, 9,216 integer correct/CE rows +192 clean rows; four frozen one-sided t11/Holm4 and pointwise CIs** PASS.
- Three exact 200-update retraining replays, including both AdamW and Muon optimizer moments, were **bitwise identical** to stored states. Two independent raw-image heldout reinferences matched stored correct counts and CE sums. All four pretraining unit tests passed.
- Owner review evidence ZIP SHA256 **`e43291911262821b5039fc3dc35e50c068945bd8f5b4c004beab1e0e6be8a61d`** (1,684,141 bytes); full 192-state evidence SHA256 **`90700562732bcf2057a3d82db94d124decc2a143c8ad942c6e271a26d0d4609a`** (28,798,283 bytes). They are current-chat attachments **not yet mirrored into the GitHub repository**.
- Hardware AMD EPYC 9V74, actual clock uncontrolled, Python3.13.5, torch2.10.0+cpu, NumPy2.3.5, SciPy1.17.0, sklearn1.8.0, CPU4 workers ×1 torch thread/worker, batch100, MKLDNN enabled. No quantization/zero-point.

## H/T/D/C/U and provenance limits

**H:** two shifted-CE superiority tests at 0-null plus two clean noninferiority tests at +.05-null, in Holm4 family. **T:** exactly12 pools,48 initializer-blocks,192 final models and globally sealed train/test barrier. **D:** all four component SUPPORT, both joint SUPPORT conditionally on historical Digits images and sampled generator. **C:** different actual applied update norms, optimizer histories, early trajectory divergence and hyperparameters; alternative interpolation could be mostly a step-size modification. **U:** reported t11 confidence intervals reflect among-pool variation only; genuine new image/task/backend/hardware combined uncertainty cannot be estimated.

The research remains open on a unique mechanism, online optimizer-aware **seed selection**, truly new-image/task replication, per-pool safety, and equal-GPU-cost improvement. No X announcement.

## Variable table / SI and dimensions

| Symbol | 意味 | SI unit | Definition | Domain/type |
|---|---|---|---|---|
| α | AdamW提案の採用混合率 | 1 | 0/.25/.5/1 after step80, zero before | [0,1], real scalar |
| W, W_M, W_A | 現在・Muon/AdamW提案hidden重み | 1 | current/optimizer-proposed 128×64 weights | finite matrix |
| D_M, D_A, D_α | 各提案更新と実採用更新 | 1 | W_M−W, W_A−W, (1−α)D_M+αD_A | real 128×64 matrix |
| r_M,r_A,r_α | 各更新のFrobeniusノルム | 1 | ||D||_F | nonnegative scalar |
| c | 二提案の方向cosine | 1 | inner product/(r_M r_A), nonzero denominator | [−1,1] scalar |
| S_q,C_q | 変換画像/clean CE | 1 | 48 heldout env/809 clean image means | nonnegative scalar |
| Δ_q,G_q | CE改善量/cleanペナルティ | 1 | S_M−S_q, C_q−C_M | real scalar |
| n | 変換プール数 | 1 | 12 | integer scalar |
| SE | 12群差の標準誤差 | 1 | sample SD of12 pool contrasts/√12 | nonnegative scalar |
| u_c | 別画像/装置に渡る合成不確かさ | 1 | not identified in this study | unavailable |

**Unit check:** all representation-coordinates and CE quantities are dimensionless SI1. The Frobenius norm-squared identity adds only norm-squared terms (SI1 in this normalized coordinate premise); CE differences, noninferiority margin and confidence limits are not percentage-point accuracy differences. Time in seconds/FLOPs/energy were not matched across production optimizers.
