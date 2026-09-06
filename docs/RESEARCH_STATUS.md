# Research status

Updated 2026-09-06. A finding is **supported** only within its comparison, task and statistical rule. Negative results and preregistered decisions are retained; a PASS label is not universal or causal proof.

## Latest completed evidence

| Experiment | Frozen decision | Evidence and scope |
|---|---|---|
| Dual-evaluation transfer specificity, #59, 30 pairs | **NO SHARED REPLICATION** | Shared benefit +0.78165 pp, one-sided p0.099776; specificity +0.00674 pp, p0.425828. Difficulty matched, but selected mixtures were near clean. |
| Fixed-dose MLP audit, #61, 30 pairs | **DOSE-DEPENDENT BENEFIT PASS** | Full benefit +2.84223 pp; full-minus-clean +1.62125 pp, one-sided p0.048974. Interaction was borderline. |
| Reserve-image MLP replication, #64, 30 pairs | **DOSE-DEPENDENT BENEFIT REPLICATES ON RESERVE IMAGES** | On the disjoint 364-image reserve subset, full benefit +2.27370 pp (p1.75e-8) and full-minus-clean +2.90556 pp (p0.001726). Intermediate strengths were negative, so no monotone dose law. |
| SmallCNN regime audit, #67, 30 pairs | **CNN FULL EFFECT REPLICATES / CLEAN INTERACTION DOES NOT** | Full benefit +2.41698 pp (p2.34e-5), but clean benefit +3.43634 pp and full-minus-clean -1.01937 pp (p0.792). Full-specific interaction is not architecture-general. |
| SmallCNN Q-scaling, #70, 30 pairs | **CNN FINITE-BUDGET COVERAGE REPLICATES** | Low-Q minus high-Q benefit attenuation +1.26494 pp, 95% CI [+0.46325,+2.06663], p0.001548; Q=16 gives exact model-state and metric identity. |

The full-strength positive effect now survives MLP and SmallCNN. However, the MLP's full-vs-clean interaction does not survive SmallCNN, so `stronger shift -> larger SGO benefit` is not a universal mechanism law.

The stronger cross-architecture result is upstream: **finite-budget subset allocation**. See [CNN_BUDGET_SCALING_RESULT.md](CNN_BUDGET_SCALING_RESULT.md), [RESERVE_DOSE_REPLICATION.md](RESERVE_DOSE_REPLICATION.md), [CNN_REGIME_INTERACTION_RESULT.md](CNN_REGIME_INTERACTION_RESULT.md), [DUAL_TRANSFER_RESULT.md](DUAL_TRANSFER_RESULT.md), and [FIXED_DOSE_RESPONSE.md](FIXED_DOSE_RESPONSE.md).

## Established results within tested regimes

| Topic | Evidence | Current conclusion |
|---|---|---|
| Structured Digits geometric shifts | Paired MLP and SmallCNN experiments | Hardness plus gradient novelty can improve held-out mean performance over loss-hard under tested structured shifts |
| Model-conditioned diversity | Gradient novelty versus transformation-parameter novelty | Model-conditioned signatures contain useful information beyond physical parameter distances in the tested MLP |
| CNN replication | Original 20 pairs plus fresh fixed-dose/Q-scaling blocks | Positive full-strength mean effects survive the tested SmallCNN parameterization |
| Optimizer replication | Independently tuned AdamW and SGD+momentum | Effect is not explained by AdamW alone in the tested MLP |
| RNG candidate compression | Prefix/compression sweeps | Moderate prefiltering can reduce signature evaluations; aggressive compression loses tail coverage |
| Learned RNG relevance | Original and shifted-coordinate generators | Relevant RNG coordinates can be learned using training gradients; stale fingerprint transfer fails |
| Relative redundancy | Digits and independent synthetic task | Within-step normalization transfers better than fixed absolute cosine targets, without a safety guarantee |
| CIFAR-10 / ResNet-20 primary | 40 paired runs | Mean +0.1206 pp, Holm(5) p0.01336; CIFAR tail robustness remains unconfirmed |
| Gradient mechanism audit | Mean-gradient and one-step controls | Final gains are not explained by superior mean-gradient estimation or maximal immediate loss decrease alone |
| Finite-budget Q-scaling | Three preregistered n=30 blocks: MLP, fresh MLP, SmallCNN | Low-vs-high attenuation +2.034 pp (p1.06e-7), +2.086 pp (p5.16e-7), then +1.265 pp (p0.00155); exact method/model identity when Q=K in all three |
| Function-preserving rank intervention | 20 reps, two methods | Raw hidden rank changes without changing the learned function; raw rank is not intrinsic causal evidence |

Thus the best-supported mechanism statement is now:

```text
binding subset-update budget
    + hard, model-conditioned non-redundant candidate gradients
    -> different optimization trajectory / learned function
    -> performance benefit in tested regimes.
```

The **finite-budget dependence is architecture-robust within the Digits/geometric family**. The downstream map from trajectory change to clean versus shifted performance is architecture/optimization dependent and remains unidentified.

## Raw-rank predictor versus mediator

The recorded prospective raw-rank direction rule matched nine registered condition-level tests across six datasets: Digits (three generators), Wine, Iris, Diabetes regression, FashionMNIST (Tiny and Medium Transformer) and CIFAR-10. These are not nine independent datasets or independent Bernoulli trials. The record is a fixed-parameterization condition-average marker, not a calibrated per-run gate.

Function-preserving intervention changed raw effective rank while predictions stayed identical. Channel-standardized rank did not rescue the mediator theory: in fresh #38, benefit attenuation replicated but standardized-rank attenuation was -0.00271, one-sided p0.5063, yielding **COVERAGE REPLICATION ONLY**. Raw rank remains a trajectory marker, not a causal quantity.

See [THEORETICAL_FRAMEWORK.md](THEORETICAL_FRAMEWORK.md), [FUNCTION_PRESERVING_RANK_INTERVENTION.md](FUNCTION_PRESERVING_RANK_INTERVENTION.md), [STANDARDIZED_RANK_BUDGET_RESULT.md](STANDARDIZED_RANK_BUDGET_RESULT.md) and [PROSPECTIVE_REPRESENTATION_RANK.md](PROSPECTIVE_REPRESENTATION_RANK.md).

## Negative results and mechanism boundaries

More candidate seeds are not monotonically better; worst-only selection can damage mean/clean performance; pure diversity without hardness is weak. Full-network gradient signatures did not beat the cheaper head proxy in the tested MLP. Learned selectors/fingerprints do not transfer universally, and long irrelevant RNG fingerprints can hurt. Absolute cosine targets can be infeasible on another task. Relative control is not a task-level safety guarantee.

Gradient novelty did not outperform random sampling as a mean-gradient estimator; loss-hard can produce larger one-step decrease. Higher accumulated gradient effective rank is insufficient. Raw representation rank is coordinate-dependent and non-causal; standardized rank failed the fresh mediator-coupling test.

The structured-versus-nuisance matching program (#41/#44/#47/#50/#53/#56) repeatedly failed calibration or support overlap. #59 eventually matched near the clean limit but returned **NO SHARED REPLICATION**. These attempts have not identified reusable-factor causality.

The MLP full-minus-clean effect replicated on reserve images, but SmallCNN directly failed that interaction while retaining a positive full effect. Therefore a universal shift-strength/dose conversion law is rejected. Q curves are also not strictly monotone: in the fresh SmallCNN Q-scaling block Q=12 benefit exceeded Q=8. The supported budget claim is the preregistered low-vs-high attenuation plus exact Q=K identity.

CIFAR primary p10/minimum differences were positive but Holm p0.05294/0.08815; they are not confirmed tail benefits.

## Execution and statistical uncertainty

The CIFAR single-thread hosted-CPU audit returned **DRIFT PERSISTS**: max rank drift0.427255 and accuracy drift0.014667. Hardware-dependent numerical paths remain a candidate cause, not an isolated sole cause.

The SmallCNN Q-scaling run used Python3.12, PyTorch2.10.0+cpu, one thread and deterministic algorithms. Hosted shards used AMD EPYC7763, AMD EPYC9V74 and Intel Xeon Platinum8573C. All scientific comparisons are paired within replicate; no speed or cross-hardware bitwise claim is made.

For Issue #70, all six artifact ZIP digests matched GitHub metadata; 300/300 checkpoint file hashes and 300/300 canonical state-tensor digests matched; all Q=16 parameter tensors were bitwise equal between methods; 24,000 environment rows independently reproduced aggregate heldout metrics to maximum numerical error 1.11e-16.

The training pairs—not heldout environments—are the statistical replicate units. Confidence intervals quantify run-to-run uncertainty conditional on the fixed Digits images/environment samples, not combined uncertainty across datasets or hardware.

## Current open work and public wording

Safe current wording:

> Gradient-aware selection of stochastic training environments can improve held-out mean performance in some structured regimes. Three preregistered n=30 Q-scaling experiments within the Digits/geometric family—two MLP blocks and one SmallCNN block—support a finite-budget subset-allocation explanation: the advantage is larger in the frozen low-Q versus high-Q contrast and becomes exactly zero when all K candidates contribute. This establishes architecture robustness within that family, not cross-dataset universality. The downstream function-space mediator remains unidentified.

The highest-value next falsification is **not another Digits replication**. Repeat the frozen Q-scaling contrast on a materially different task/architecture where the environment generator and compute budget can be held fixed cleanly. FashionMNIST/Transformer or CIFAR/ResNet are candidates, with protocol choices fixed before outcomes.

Other priorities: early-trajectory training-only diagnostics, a new tail-safety theory, pinned-hardware numerical studies, and GPU comparisons with genuinely matched costs. Do not claim universal seed quality, a universal controller, causal rank law, a universal monotone Q curve, reliable per-run gating, confirmed CIFAR tail robustness, general large-Transformer validity, or GPU efficiency.
