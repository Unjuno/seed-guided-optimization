# Research status

Updated 2026-09-06. A finding is **supported** only within its comparison, task and statistical rule. Negative results and preregistered decisions are retained; a PASS label is not universal or causal proof.

## Latest completed evidence

| Experiment | Frozen decision | Evidence and scope |
|---|---|---|
| Dual-evaluation transfer specificity, #59, 30 pairs | **NO SHARED REPLICATION** | Shared benefit +0.78165 pp, one-sided p0.099776; specificity +0.00674 pp, p0.425828. Difficulty matched, but selected mixtures were near clean. |
| Fixed-dose functional-response audit, #61, fresh 30 pairs | **DOSE-DEPENDENT BENEFIT PASS** | Full geometric benefit +2.84223 pp, 95% CI [2.23536,3.44910]; full-minus-clean +1.62125 pp, 95% CI [-0.31784,3.56035], one-sided p0.048974. Interaction is borderline. |

The new audit does not overturn #59. It changes the evaluation question to a fixed-dose profile with clean and full-strength endpoints. No new calibration was used. The same trained states were evaluated across strengths; all 60 checkpoint hashes and 38,460 environment-level rows were verified. It is not a difficulty-matched specificity or causal-mediation experiment. Intermediate-dose benefits were not monotone.

See [DUAL_TRANSFER_RESULT.md](DUAL_TRANSFER_RESULT.md) and [FIXED_DOSE_RESPONSE.md](FIXED_DOSE_RESPONSE.md) for the protocol, complete primary statistics, artifact IDs, runtime conditions and interpretation boundaries.

## Established results within tested regimes

| Topic | Evidence | Current conclusion |
|---|---|---|
| Structured Digits geometric shifts | Paired MLP experiments | Hardness plus gradient novelty can improve held-out mean/tail metrics over loss-hard |
| Model-conditioned diversity | Gradient novelty versus transformation-parameter novelty | Model-conditioned signatures contain useful information beyond physical parameter distances in the tested MLP |
| CNN replication | 20 paired runs | Mean and minimum gains versus loss-hard survive Holm correction |
| Optimizer replication | Independently tuned AdamW and SGD+momentum | Effect is not explained by AdamW alone in the tested MLP |
| RNG candidate compression | Prefix/compression sweeps | Moderate prefiltering can reduce signature evaluations; aggressive compression loses tail coverage |
| Learned RNG relevance | Original and shifted-coordinate generators | Relevant RNG coordinates can be learned using training gradients; stale fingerprint transfer fails |
| Soft relevance | Cross-generator weighted-top12 control | Soft weighting is more robust than forcing an exact hard set in the tested generator |
| Relative redundancy | Digits and independent synthetic task | Within-step normalization transfers better than fixed absolute cosine targets, without a safety guarantee |
| CIFAR-10 / ResNet-20 primary | 40 paired runs | Mean +0.1206 pp, Holm(5) p0.01336; CIFAR tail robustness remains unconfirmed |
| Gradient mechanism audit | Mean-gradient and one-step controls | Final gains are not explained by superior mean-gradient estimation or maximal immediate loss decrease alone |
| Finite-budget Q-scaling | Two preregistered n=30 blocks | Frozen attenuation +2.034 pp (p1.06e-7) then +2.086 pp (p5.16e-7); exact method identity when all candidates contribute |
| Function-preserving rank intervention | 20 reps, two methods | Raw hidden rank changes without changing the learned function; raw rank is not intrinsic causal evidence |

Primary detailed evidence remains in [RESULTS.md](RESULTS.md) and the individual experiment documents. The budget findings support the importance of subset allocation within Digits/geometric conditions; they do not identify a unique causal coverage mechanism.

## Raw-rank predictor versus mediator

The recorded prospective raw-rank direction rule matched nine registered condition-level tests across six datasets: Digits (three generators), Wine, Iris, Diabetes regression, FashionMNIST (Tiny and Medium Transformer) and CIFAR-10. These are not nine independent datasets or independent Bernoulli trials. The record is a fixed-parameterization condition-average marker, not a calibrated per-run gate.

FashionMNIST/Tiny Transformer had initial n=10 PASS and independent n=20 directional replication: rank +0.07853 and held-out mean +0.4377 pp. The combined n=30 estimate is secondary precision analysis. FashionMNIST/Medium Transformer n=10 gave raw rank +0.10293 and mean +0.9579 pp. CIFAR/ResNet independent reps40-49 gave raw rank +0.03205 and mean +0.1703 pp; rank's descriptive p0.1464 does not show significant rank expansion.

Function-preserving intervention changed raw effective rank while predictions stayed identical. Channel-standardized rank did not rescue the mediator theory: in fresh #38, benefit attenuation replicated but standardized-rank attenuation was -0.00271, one-sided p0.5063, yielding **COVERAGE REPLICATION ONLY**. Raw rank was secondary and had attenuation +0.1780, p0.0180. The first Q-scaling raw-rank mediator threshold also failed (p0.1095).

See [THEORETICAL_FRAMEWORK.md](THEORETICAL_FRAMEWORK.md), [FUNCTION_PRESERVING_RANK_INTERVENTION.md](FUNCTION_PRESERVING_RANK_INTERVENTION.md), [STANDARDIZED_RANK_BUDGET_RESULT.md](STANDARDIZED_RANK_BUDGET_RESULT.md) and [PROSPECTIVE_REPRESENTATION_RANK.md](PROSPECTIVE_REPRESENTATION_RANK.md).

## Negative results and failed mechanism designs

More candidate seeds are not monotonically better; worst-only selection can damage mean/clean performance; pure diversity without hardness is weak. Full-network gradient signatures did not beat the cheaper head proxy in the tested MLP. Learned selectors/fingerprints do not transfer universally, and long irrelevant RNG fingerprints can hurt. Absolute cosine targets can be infeasible on another task. Relative control is not a task-level safety guarantee.

Gradient novelty did not outperform random sampling as a mean-gradient estimator; loss-hard can produce larger one-step decrease. Higher accumulated gradient effective rank is insufficient (Breast-high counterexample). The representation-rank-SD tail rule failed on Wine and remains retired. Raw-rank signs are unreliable per replicate. Optimizer under-tuning can create a false selector failure.

The structured-versus-IID-nuisance test #41 failed its matching prerequisites, so its large held-out contrast remained **INCONCLUSIVE**. Gated high-dimensional nuisance #44, two-axis #47 and common-contrast #50 all failed training-only calibration. Those gates prevented confirmatory evaluation. Evaluation-only nuisance difficulty calibrations #53/#56 also failed; #59 eventually matched difficulty near the clean-input limit but returned the negative result above. These attempts have not identified a reusable-factor causal mediator.

CIFAR primary p10/minimum differences were positive but Holm p0.05294/0.08815; they are not confirmed tail benefits. A null significance result is not exact-zero evidence.

## Execution and statistical uncertainty

The CIFAR single-thread hosted-CPU audit returned **DRIFT PERSISTS**: max rank drift0.427255 and accuracy drift0.014667, with positive aggregate directions in both repeats. Rep45 was bitwise equal across AMD EPYC7763/9V74; rep46 differed across AMD EPYC7763/Intel Xeon6973P-C. Hardware-dependent numerical paths are a candidate cause, not an isolated sole cause. See [CIFAR_CPU_REPRO_AUDIT.md](CIFAR_CPU_REPRO_AUDIT.md).

The latest fixed-dose run used Python3.12.14, PyTorch2.10.0+cpu and pinned scientific dependencies, one thread and deterministic algorithms. CPU families still varied across shards; methods within each pair ran on the same runner. All states were retained and source/checkpoint hashes audited. No controlled-clock, GPU-efficiency or bitwise cross-hardware claim is made.

The 30 training pairs—not the environments—are the statistical replicate units. Confidence intervals use paired SE and t29 multiplier2.04523. They are conditional on the fixed images and environment samples, not combined uncertainty across datasets/hardware. The reused Digits split contains 988 training/445 test examples. New seed IDs are not new image data. Earlier loss-hard-only evaluation calibration reused test-image labels and was not training-only validation.

## Current open work and public wording

The causal internal mediator remains unidentified. The current evidence supports gradient-aware environment selection in some structured finite-budget regimes and a replicated full-strength effect, with only tentative evidence for the latest full-minus-clean interaction.

Next statistical priority: independently challenge that borderline interaction at fixed strengths. Next mechanistic priority: intervene on reusable structure without confounding it with difficulty, margins or other function-space changes. Do not keep extending the present sample or selecting strengths until significance appears.

Other priorities remain materially different tasks/modalities, early-trajectory training-only diagnostics, a new tail-safety theory, pinned-hardware numerical studies, and GPU comparisons with truly matched costs. Do not claim universal seed quality, a universal controller, causal rank law, reliable per-run gating, confirmed CIFAR tail robustness, general large-Transformer validity, or GPU efficiency.
