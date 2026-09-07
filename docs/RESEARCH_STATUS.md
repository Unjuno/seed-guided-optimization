# Research status

Updated 2026-09-07. A finding is **supported** only within its comparison, task and statistical rule. Negative results and preregistered decisions are retained; a PASS label is not universal or causal proof.

## Latest completed evidence

| Experiment | Frozen decision | Evidence and scope |
|---|---|---|
| Reserve-image MLP replication, #64, 30 pairs | **DOSE-DEPENDENT BENEFIT REPLICATES ON RESERVE IMAGES** | Full +2.27370 pp, p1.75e-8; full-minus-clean +2.90556 pp, p0.001726. Intermediate strengths negative. |
| SmallCNN regime audit, #67, 30 pairs | **CNN FULL EFFECT REPLICATES / CLEAN INTERACTION DOES NOT** | Full +2.41698 pp, p2.34e-5; clean +3.43634 pp; full-minus-clean -1.01937 pp, p0.792. |
| SmallCNN Q-scaling, #70, 30 pairs | **CNN FINITE-BUDGET COVERAGE REPLICATES** | attenuation +1.26494 pp, 95% CI [+0.46325,+2.06663], p0.001548; Q16 exact identity. |
| SmallCNN endpoint sensitivity, post-hoc | descriptive only | Excluding Q16: low mean(Q2,Q4)-Q12 +0.47608 pp, 95% CI [-0.53741,+1.48956], p0.1723. The primary PASS does not establish non-full-Q attenuation by itself. |
| FashionMNIST Tiny Transformer Q-scaling, #73, 30 pairs | **FASHION TRANSFORMER FINITE-BUDGET COVERAGE REPLICATES** | attenuation +0.31375 pp, 95% CI [-0.00651,+0.63401], preregistered one-sided p0.027264; Q8 exact. Borderline cross-task support. |
| CIFAR-10 ResNet-20 Q-scaling, #76, 30 pairs | **CIFAR RESNET FINITE-BUDGET COVERAGE REPLICATES** | attenuation **+0.11679 pp**, 95% CI **[+0.03353,+0.20005]**, p**0.003804**; Q8 exact identity. Strong larger-task cross-task replication. |
| CIFAR endpoint sensitivity, post-hoc | descriptive only | Excluding Q8: low mean(Q2,Q4)-Q6 **+0.09682 pp**, 95% CI **[+0.00324,+0.19040]**, p0.02152. Unlike SmallCNN, attenuation remains positive without the forced Q=K endpoint. |

The full-strength positive effect survives MLP and SmallCNN, but the full-vs-clean interaction is not architecture-general. The strongest common upstream evidence is now finite-budget subset allocation.

## Finite-budget mechanism status

Five preregistered n=30 Q-scaling blocks are complete:

| Task / model | Frozen attenuation | One-sided p | Q=K identity | Interpretation |
|---|---:|---:|---|---|
| Digits geometric / MLP #1 | +2.034 pp | 1.06e-7 | exact | strong |
| Digits geometric / MLP #2 fresh | +2.086 pp | 5.16e-7 | exact | strong replication |
| Digits geometric / SmallCNN | +1.265 pp | 0.001548 | exact | architecture replication within Digits |
| FashionMNIST / Tiny Transformer | +0.314 pp | 0.027264 | exact | cross-task support, but two-sided CI crosses zero |
| CIFAR-10 / ResNet-20 | **+0.1168 pp** | **0.003804** | exact | strong larger-task cross-task replication |

The common frozen statistic compares low subset coverage with high subset coverage. **Strict monotonicity is not a universal law.** SmallCNN had Q12 > Q8 and Fashion had Q6 > Q4. CIFAR happened to show a monotone mean curve Q2>Q4>Q6>Q8 in its fresh block.

Q=K identity is a necessary implementation control: when every candidate contributes in the same order, the two selectors must collapse to the same update. It should not be treated as causal-mediation proof by itself. The CIFAR post-hoc sensitivity is important because the low-Q advantage remains positive relative to Q6 even after Q8 is removed; the analogous SmallCNN sensitivity does not.

Current mechanism statement:

```text
binding subset-update budget
    + hard, model-conditioned non-redundant candidate gradients
    -> different allocation across unresolved learning directions
    -> changed optimization trajectory / learned function
    -> task/architecture-dependent performance expression
```

The first arrow is increasingly well supported as a finite-budget phenomenon, but the phrase “coverage of unresolved directions” remains a working mechanistic interpretation rather than a directly identified causal mediator.

See [CIFAR_BUDGET_SCALING_RESULT.md](CIFAR_BUDGET_SCALING_RESULT.md), [CNN_BUDGET_SCALING_RESULT.md](CNN_BUDGET_SCALING_RESULT.md), [FASHION_BUDGET_SCALING_RESULT.md](FASHION_BUDGET_SCALING_RESULT.md), and [THEORETICAL_FRAMEWORK.md](THEORETICAL_FRAMEWORK.md).

## Other established results within tested regimes

| Topic | Evidence | Current conclusion |
|---|---|---|
| Structured Digits geometric shifts | MLP and SmallCNN paired experiments | Hardness plus gradient novelty can improve held-out mean performance over loss-hard under tested structured shifts |
| Model-conditioned diversity | Gradient novelty versus transformation-parameter novelty | Model-conditioned signatures contain useful information beyond physical parameter distances in the tested MLP |
| Optimizer replication | Independently tuned AdamW and SGD+momentum | Effect is not explained by AdamW alone in the tested MLP |
| RNG candidate compression | Prefix/compression sweeps | Moderate prefiltering can reduce signature evaluations; aggressive compression loses tail coverage |
| Learned RNG relevance | Original and shifted-coordinate generators | Relevant RNG coordinates can be learned using training gradients; stale fingerprint transfer fails |
| Relative redundancy | Digits and independent synthetic task | Within-step normalization transfers better than fixed absolute cosine targets, without a safety guarantee |
| CIFAR-10 / ResNet-20 primary | 40 paired runs | Mean +0.1206 pp, Holm(5) p0.01336; CIFAR tail robustness remains unconfirmed |
| Gradient mechanism audit | Mean-gradient and one-step controls | Final gains are not explained by superior mean-gradient estimation or maximal immediate loss decrease alone |
| Function-preserving rank intervention | 20 reps, two methods | Raw hidden rank changes without changing the learned function; raw rank is not intrinsic causal evidence |

## Raw-rank predictor versus mediator

The prospective raw-rank direction record remains a fixed-parameterization condition-average marker, not a calibrated per-run gate. Function-preserving intervention changed raw effective rank while predictions stayed identical. Channel-standardized rank did not rescue the mediator theory: in fresh #38, benefit attenuation replicated but standardized-rank attenuation was -0.00271, one-sided p0.5063.

Thus raw/standardized representation rank should not be treated as the causal state variable. The next mediator needs to be functionally intrinsic, prospective, training-only where possible, and coupled to the budget effect across tasks.

## Negative results and mechanism boundaries

More candidate seeds are not monotonically better; worst-only selection can damage mean/clean performance; pure diversity without hardness is weak. Gradient novelty did not outperform random sampling as a mean-gradient estimator; loss-hard can produce larger one-step decrease. Higher accumulated gradient effective rank is insufficient.

The structured-versus-nuisance matching program repeatedly failed calibration/support overlap, and the matched near-clean #59 test returned **NO SHARED REPLICATION**. Reusable-factor causality remains unproven.

The MLP full-minus-clean effect replicated on reserve images but failed in SmallCNN while full performance stayed positive. Therefore a universal shift-strength conversion law is rejected. Q curves are also not universally monotone.

Q=K disappearance establishes that selector freedom vanishes at full candidate coverage, but not that gradient non-redundancy is itself the causal ingredient. The next direct test should hold Q and hardness structure fixed while actively reversing gradient redundancy.

CIFAR primary and Q-scaling tail differences are secondary/multiplicity-sensitive; confirmed tail safety remains open.

## Execution and uncertainty

The CIFAR Q-scaling recovery is documented because the original five-replicate shard 60-64 exceeded a 180-minute workflow limit before held-out evaluation. The recovery preserved exactly 25 completed sealed reps, rejected the entire unsealed partial shard, retrained only reps60-64 in individual jobs under unchanged scientific code, then required a global 240-state seal before evaluation.

Recovery validation verified 240 checkpoints, 30 Q8 training identity pairs and 7680 environment rows with maximum environment reaggregation error 0.0. Protocol hash: `e388d13d5890e8b60e939f085403763d5065360bd3cdaecc7aa05de510f144d1`.

The earlier CIFAR single-thread hosted-CPU audit returned **DRIFT PERSISTS**; bitwise cross-hardware hosted-CPU reproducibility remains unestablished. Q-scaling comparisons are paired within replicate and do not constitute a cross-hardware bitwise claim.

The paired training runs—not held-out environments—are the statistical replicate units. Confidence intervals are conditional on the fixed dataset subsets and environment samples.

## Public claim boundary and next work

Safe current wording:

> Gradient-aware selection of stochastic training environments can improve held-out mean performance in some structured regimes. Five preregistered Q-scaling blocks support a finite-budget subset-allocation explanation: three strong Digits/geometric blocks, a borderline FashionMNIST/Tiny Transformer block, and a strong CIFAR-10/ResNet-20 block. In all blocks the methods become exactly identical when all K candidates contribute. CIFAR additionally shows positive exploratory attenuation after removing the exact-zero Q=K endpoint. These results support finite selector freedom as an important upstream condition while leaving the causal role of gradient non-redundancy and the downstream learned-function mediator unresolved.

Do not claim universal task validity, a universal monotone Q curve, universal seed quality/controller settings, causal rank laws, reliable per-run gating, confirmed CIFAR tails, large-Transformer generality, cross-hardware bitwise reproducibility, or GPU efficiency.

The highest-value next mechanism test is **loss-stratified gradient-redundancy reversal**: keep Q and loss-rank strata fixed, then prospectively choose maximally non-redundant versus maximally redundant subsets. This directly tests non-redundancy without relying on Q=K identity. In parallel, downstream function-space diagnostics should replace further rank-normalization searches.
