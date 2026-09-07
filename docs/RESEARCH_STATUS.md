# Research status

Updated 2026-09-07. A finding is **supported** only within its comparison, task and statistical rule. Negative results and preregistered decisions are retained; a PASS label is not universal or proof of a unique mediator.

## Latest completed evidence

| Experiment | Frozen decision | Evidence and scope |
|---|---|---|
| Loss-stratified shared-reference intervention, #80, 30 pairs | **LOSS-STRATIFIED NONREDUNDANCY SUPPORT** | novelty contrast +0.23163, 30/30 positive; standardized reference hardness difference -0.00910 SD with 90% CI [-0.01328,-0.00492] inside frozen ±0.10 TOST margin; MAXNOV-MINNOV heldout **+4.669 pp**, 95% CI [+3.187,+6.152], p2.39e-7, 26/30 positive. Post-hoc physical parameter diversity also increases strongly, so gradient-specific uniqueness remains unresolved. |
| Reserve-image MLP replication, #64, 30 pairs | **DOSE-DEPENDENT BENEFIT REPLICATES ON RESERVE IMAGES** | Full +2.27370 pp, p1.75e-8; full-minus-clean +2.90556 pp, p0.001726. Intermediate strengths negative. |
| SmallCNN regime audit, #67, 30 pairs | **CNN FULL EFFECT REPLICATES / CLEAN INTERACTION DOES NOT** | Full +2.41698 pp, p2.34e-5; clean +3.43634 pp; full-minus-clean -1.01937 pp, p0.792. |
| SmallCNN Q-scaling, #70, 30 pairs | **CNN FINITE-BUDGET COVERAGE REPLICATES** | attenuation +1.26494 pp, 95% CI [+0.46325,+2.06663], p0.001548; Q16 exact identity. |
| FashionMNIST Tiny Transformer Q-scaling, #73, 30 pairs | **FASHION TRANSFORMER FINITE-BUDGET COVERAGE REPLICATES** | attenuation +0.31375 pp, 95% CI [-0.00651,+0.63401], preregistered one-sided p0.027264; Q8 exact. Borderline cross-task support. |
| CIFAR-10 ResNet-20 Q-scaling, #76, 30 pairs | **CIFAR RESNET FINITE-BUDGET COVERAGE REPLICATES** | attenuation **+0.11679 pp**, 95% CI **[+0.03353,+0.20005]**, p**0.003804**; Q8 exact identity. Strong larger-task cross-task replication. |
| CIFAR endpoint sensitivity, post-hoc | descriptive only | Excluding Q8: low mean(Q2,Q4)-Q6 **+0.09682 pp**, 95% CI **[+0.00324,+0.19040]**, p0.02152. Attenuation remains positive without forced Q=K zero. |

The strongest common upstream evidence is finite-budget subset allocation. Issue #80 adds a fixed-Q direct intervention: within tightly loss-stratified feasible subsets defined on the same reference states, high reference gradient novelty produces much better shifted heldout mean than low novelty. The main remaining competing explanation is that the high-novelty schedules also select more diverse known physical transformations.

## Direct nonredundancy intervention status

Issue #80 removes the largest limitation of Q-scaling: Q is fixed at 4 in both arms and no Q=K endpoint enters the comparison.

Shared-reference construction:

```text
same model state + same K=16 candidates
 -> retain loss rank 1
 -> choose one from each identical pair (2,3), (4,5), (6,7)
 -> among the same 8 feasible Q4 subsets:
      MAXNOV = maximum pairwise gradient novelty
      MINNOV = minimum pairwise gradient novelty
 -> seal both schedules
 -> reset two identical SmallCNNs and train on the fixed schedules
```

Frozen results:

- novelty manipulation mean: **+0.231634**, SE0.002623, 95% CI [+0.226269,+0.236999], p4.12e-37, 30/30 positive;
- standardized reference hardness: **-0.009100 SD**, 90% CI [-0.013276,-0.004924], both TOST p<3e-26, frozen ±0.10 margin passed;
- reference raw selected-loss difference: -0.001248 CE;
- heldout mean MAXNOV-MINNOV: **+4.66929 pp**, SE0.72486 pp, 95% CI **[+3.18679,+6.15180]**, p2.389e-7, 26/30 positive;
- clean effect secondary: **-1.75525 pp**;
- p10/minimum secondary: +5.48908/+3.79481 pp, not multiplicity-controlled tail claims.

Independent archive audit verified six evaluation ZIP digests, 30 schedule hashes, 60 checkpoint file/tensor digests, 2,400 reference-step rows and 4,800 environment rows. Environment reaggregation maximum error was 1.25e-16.

### Remaining competing explanation

Post-hoc, the seven known geometric environment parameters (rotation, x/y translation, blur, contrast, brightness, noise) were standardized across the 64 training environments. Mean selected-set pairwise parameter distance was:

- MAXNOV: 3.97085;
- MINNOV: 3.47815;
- difference: **+0.492697**, SE0.009559, 95% CI [+0.473147,+0.512246], 30/30 positive.

Thus the present treatment changes both model-conditioned gradient novelty and physical transformation diversity. The frozen PASS is valid for the preregistered schedule intervention, but the strongest gradient-specific causal wording is not yet justified. The earlier finding that gradient-novel selection outperformed parameter-novel selection weakens a pure parameter-distance account, but does not remove this within-intervention covariate shift.

See [LOSS_STRATIFIED_NONREDUNDANCY_RESULT.md](LOSS_STRATIFIED_NONREDUNDANCY_RESULT.md).

## Finite-budget mechanism status

Five preregistered n=30 Q-scaling blocks are complete:

| Task / model | Frozen attenuation | One-sided p | Q=K identity | Interpretation |
|---|---:|---:|---|---|
| Digits geometric / MLP #1 | +2.034 pp | 1.06e-7 | exact | strong |
| Digits geometric / MLP #2 fresh | +2.086 pp | 5.16e-7 | exact | strong replication |
| Digits geometric / SmallCNN | +1.265 pp | 0.001548 | exact | architecture replication within Digits |
| FashionMNIST / Tiny Transformer | +0.314 pp | 0.027264 | exact | cross-task support, two-sided CI crosses zero |
| CIFAR-10 / ResNet-20 | **+0.1168 pp** | **0.003804** | exact | strong larger-task cross-task replication |

Strict monotonicity is not a universal law. Q=K identity is a necessary implementation control, not causal proof. CIFAR is stronger than SmallCNN under a post-hoc non-Q=K sensitivity because the low-Q versus Q6 contrast remains positive with a two-sided interval just above zero.

Current mechanism picture:

```text
binding subset-update budget
    -> selector freedom among hard candidates
    -> model-conditioned allocation differs
    -> optimization trajectory / learned function differs
    -> architecture/task-dependent performance expression
```

Issue #80 further supports:

```text
fixed Q + matched reference hardness strata
    + schedule chosen for high versus low reference gradient nonredundancy
    -> large shifted heldout difference
```

but the high-novelty schedule also changes physical transformation diversity. The next test must orthogonalize these two quantities.

## Other established results within tested regimes

| Topic | Evidence | Current conclusion |
|---|---|---|
| Structured Digits geometric shifts | MLP and SmallCNN paired experiments | Hardness plus gradient novelty can improve held-out mean performance over loss-hard under tested structured shifts |
| Model-conditioned versus parameter diversity | Gradient novelty versus transformation-parameter novelty | Model-conditioned signatures contain useful information beyond physical parameter distances in the tested MLP, but Issue #80 shows the two can still covary strongly |
| Optimizer replication | Independently tuned AdamW and SGD+momentum | Effect is not explained by AdamW alone in the tested MLP |
| RNG candidate compression | Prefix/compression sweeps | Moderate prefiltering can reduce signature evaluations; aggressive compression loses tail coverage |
| Learned RNG relevance | Original and shifted-coordinate generators | Relevant RNG coordinates can be learned using training gradients; stale fingerprint transfer fails |
| Relative redundancy | Digits and independent synthetic task | Within-step normalization transfers better than fixed absolute cosine targets, without a safety guarantee |
| CIFAR-10 / ResNet-20 primary | 40 paired runs | Mean +0.1206 pp, Holm(5) p0.01336; CIFAR tail robustness remains unconfirmed |
| Gradient mechanism audit | Mean-gradient and one-step controls | Final gains are not explained by superior mean-gradient estimation or maximal immediate loss decrease alone |
| Function-preserving rank intervention | 20 reps, two methods | Raw hidden rank changes without changing the learned function; raw rank is not intrinsic causal evidence |

## Raw-rank predictor versus mediator

The prospective raw-rank direction record remains a fixed-parameterization condition-average marker, not a calibrated per-run gate. Function-preserving intervention changed raw effective rank while predictions stayed identical. Channel-standardized rank did not rescue the mediator theory: in fresh #38, benefit attenuation replicated but standardized-rank attenuation was -0.00271, one-sided p0.5063.

Thus raw/standardized representation rank should not be treated as the causal state variable. The next mediator needs to be functionally intrinsic, prospective, training-only where possible, and coupled to the budget/direct-intervention effect across tasks.

## Negative results and mechanism boundaries

More candidate seeds are not monotonically better; worst-only selection can damage mean/clean performance; pure diversity without hardness is weak. Gradient novelty did not outperform random sampling as a mean-gradient estimator; loss-hard can produce larger one-step decrease. Higher accumulated gradient effective rank is insufficient.

The structured-versus-nuisance matching program repeatedly failed calibration/support overlap, and the matched near-clean #59 test returned **NO SHARED REPLICATION**. Reusable-factor causality remains unproven.

The MLP full-minus-clean effect replicated on reserve images but failed in SmallCNN while full performance stayed positive. A universal shift-strength conversion law is rejected. Q curves are not universally monotone.

Issue #80 provides strong fixed-Q intervention evidence but does **not** isolate gradient novelty from known physical transformation diversity. Therefore “gradient novelty is the unique causal mediator” remains too strong.

CIFAR primary and Q-scaling tail differences, and Issue #80 p10/minimum differences, are secondary/multiplicity-sensitive; confirmed tail safety remains open.

## Execution and uncertainty

The CIFAR Q-scaling recovery preserved 25 completed sealed reps, rejected the unsealed partial shard, retrained only reps60-64 under unchanged code and required a global240-state seal before evaluation. Validation verified240 checkpoints,30 Q8 identity pairs and7680 environment rows with max reaggregation error0.0.

Issue #80 enforced two separate global barriers: all 30 reference schedules were sealed before any intervention arm trained, and all60 intervention states were sealed before any heldout environment was constructed.

The earlier CIFAR hosted-CPU audit returned **DRIFT PERSISTS**; bitwise cross-hardware reproducibility remains unestablished. Paired training runs—not heldout environments—are the statistical replicate units. Confidence intervals are conditional on fixed dataset subsets and environment samples.

## Public claim boundary and next work

Safe current wording:

> Gradient-aware selection of stochastic training environments can improve held-out mean performance in several tested structured regimes. Five preregistered Q-scaling blocks support a finite-budget subset-allocation explanation. A separate fixed-Q shared-reference intervention shows that, within tightly matched loss-rank strata, schedules maximizing reference gradient novelty substantially outperform schedules minimizing it. This is direct intervention support for the usefulness of a nonredundancy-selected schedule. However those schedules also differ strongly in known physical transformation diversity, so gradient novelty has not yet been isolated as the unique causal variable; the downstream learned-function mediator also remains unresolved.

Do not claim universal task validity, a universal monotone Q curve, universal seed quality/controller settings, causal rank laws, gradient-specific uniqueness from Issue #80 alone, reliable per-run gating, confirmed CIFAR/tail safety, large-Transformer generality, cross-hardware bitwise reproducibility, or GPU efficiency.

The highest-value next test is **physical-parameter-diversity-matched gradient-novelty reversal** under the same shared-reference, fixed-Q framework. It should match both reference hardness and the known seven-dimensional transformation diversity before maximizing versus minimizing gradient novelty. In parallel, downstream function-space diagnostics should replace further rank-normalization searches.
