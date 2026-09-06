# Research status

Updated 2026-09-06. A finding is **supported** only within its comparison, task and statistical rule. Negative results and preregistered decisions are retained; a PASS label is not universal or causal proof.

## Latest completed evidence

| Experiment | Frozen decision | Evidence and scope |
|---|---|---|
| Dual-evaluation transfer specificity, #59, 30 pairs | **NO SHARED REPLICATION** | Shared benefit +0.78165 pp, p0.099776; specificity +0.00674 pp, p0.425828. |
| Fixed-dose MLP audit, #61, 30 pairs | **DOSE-DEPENDENT BENEFIT PASS** | Full +2.84223 pp; full-minus-clean +1.62125 pp, one-sided p0.048974; interaction initially borderline. |
| Reserve-image MLP replication, #64, 30 pairs | **DOSE-DEPENDENT BENEFIT REPLICATES ON RESERVE IMAGES** | Full +2.27370 pp, p1.75e-8; full-minus-clean +2.90556 pp, p0.001726. Intermediate strengths were negative. |
| SmallCNN regime audit, #67, 30 pairs | **CNN FULL EFFECT REPLICATES / CLEAN INTERACTION DOES NOT** | Full +2.41698 pp, p2.34e-5; clean +3.43634 pp; full-minus-clean -1.01937 pp, p0.792. |
| SmallCNN Q-scaling, #70, 30 pairs | **CNN FINITE-BUDGET COVERAGE REPLICATES** | Low-Q minus high-Q attenuation +1.26494 pp, 95% CI [+0.46325,+2.06663], p0.001548; Q16 exact state/metric identity. |
| FashionMNIST Tiny Transformer Q-scaling, #73, 30 pairs | **FASHION TRANSFORMER FINITE-BUDGET COVERAGE REPLICATES** | Low-Q minus high-Q attenuation +0.31375 pp, 95% CI [-0.00651,+0.63401], preregistered one-sided p0.027264; Q8 exact identity. First direct cross-task Q-scaling support, but statistically borderline. |

The full-strength positive effect survives MLP and SmallCNN, but full-vs-clean interaction is not architecture-general. The stronger upstream mechanism is finite-budget subset allocation.

## Finite-budget mechanism status

Four preregistered n=30 Q-scaling blocks now exist:

| Task / model | Frozen attenuation | One-sided p | Q=K identity | Interpretation |
|---|---:|---:|---|---|
| Digits geometric / MLP #1 | +2.034 pp | 1.06e-7 | exact | strong |
| Digits geometric / MLP #2 fresh | +2.086 pp | 5.16e-7 | exact | strong replication |
| Digits geometric / SmallCNN | +1.265 pp | 0.001548 | exact | architecture replication within Digits |
| FashionMNIST / Tiny Transformer | +0.314 pp | 0.027264 | exact | first cross-task directional replication; two-sided CI crosses zero |

The common frozen statistic compares low subset coverage with high subset coverage; **strict monotonicity was never required and is not observed**. SmallCNN had Q12 > Q8, and Fashion had Q6 > Q4.

Exact Q=K identity is especially important: in all four Q-scaling experiments, when every candidate contributes, loss-hard and gradnov produce exactly the same learned model under the controlled deterministic execution. This supports the necessity of selector freedom under a binding subset budget.

Current mechanism statement:

```text
binding subset-update budget
    + hard, model-conditioned non-redundant candidate gradients
    -> different coverage of unresolved learning directions
    -> changed optimization trajectory / learned function
    -> task/architecture-dependent performance expression
```

Within Digits/geometric the finite-budget dependence is strongly architecture-robust. FashionMNIST/Tiny Transformer adds preregistered cross-task support, but the attenuation estimate is weak enough that it should be described as **borderline cross-task evidence**, not universal confirmation.

See [CNN_BUDGET_SCALING_RESULT.md](CNN_BUDGET_SCALING_RESULT.md), [FASHION_BUDGET_SCALING_RESULT.md](FASHION_BUDGET_SCALING_RESULT.md), [THEORETICAL_FRAMEWORK.md](THEORETICAL_FRAMEWORK.md), and the preceding experiment documents.

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

The MLP full-minus-clean effect replicated on reserve images but failed in SmallCNN while full performance stayed positive. Therefore a universal shift-strength conversion law is rejected. Q curves are also not universally monotone. The supported budget claim is the preregistered low-vs-high contrast plus exact Q=K disappearance.

CIFAR primary p10/minimum differences were positive but not Holm-significant; confirmed tail safety remains open.

## Execution and uncertainty

The CIFAR single-thread hosted-CPU audit returned **DRIFT PERSISTS**; bitwise cross-hardware hosted-CPU reproducibility is not established.

Issue #70 SmallCNN verification matched six ZIP digests, 300 checkpoint hashes/state digests, Q16 parameter tensors and 24,000 environment rows. Issue #73 Fashion verification matched six ZIP digests, all 240 checkpoint file hashes, all 240 canonical state-tensor digests, archived source hashes, Q8 tensors, aggregate rows, and independently recomputed paired statistics.

Fashion hosted shards used AMD EPYC 9V74, AMD EPYC 7763 and Intel Xeon Platinum 8573C with PyTorch2.10.0+cpu and four threads. No wall-clock or cross-hardware bitwise claim is made. The Fashion workflow did not archive per-environment heldout rows, so its heldout aggregation cannot be independently reconstructed from archived CSVs alone without rerunning the fixed data/generator; this is an evidence-archive limitation.

The paired training runs—not heldout environments—are the statistical replicate units. Confidence intervals are conditional on the fixed dataset subsets and environment samples.

## Public claim boundary and next work

Safe current wording:

> Gradient-aware selection of stochastic training environments can improve held-out mean performance in some structured regimes. Three strong preregistered Q-scaling blocks within Digits/geometric and one preregistered but borderline FashionMNIST/Tiny Transformer block support a finite-budget subset-allocation explanation: the low-Q versus high-Q contrast is positive under the frozen rules, and the methods become exactly identical when all K candidates contribute. This supports selector freedom as a necessary component of the observed effect while leaving the downstream function-space mediator unresolved.

Do not claim universal task validity, a universal monotone Q curve, universal seed quality/controller settings, causal rank laws, reliable per-run gating, confirmed CIFAR tails, large-Transformer generality, cross-hardware bitwise reproducibility, or GPU efficiency.

The highest-value next budget falsification is **CIFAR-10 / ResNet-20 Q-scaling**, using the existing CIFAR protocol with K/Q and the attenuation statistic frozen before outcomes. In parallel, the downstream mediator should be tested with training-only function-space diagnostics rather than new rank normalizations.
