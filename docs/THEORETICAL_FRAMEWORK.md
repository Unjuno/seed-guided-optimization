# Theoretical framework: budgeted coverage of unresolved gradient subspaces

## Status

This document states the current **working theory**. It is not a theorem.

The evidence now separates the mechanism into a strongly supported outer component and an unresolved inner component:

```text
binding subset budget
    + hard, non-redundant environment-induced gradients
    -> broader coverage of unresolved learning directions
    -> trajectory / learned-function change
    -> held-out mean benefit in reusable structured regimes
```

The **finite-budget coverage component has now passed two preregistered n=30 Q-scaling tests on fresh replicate/held-out blocks**. In contrast, two representation-rank mediator candidates have failed increasingly direct tests:

1. raw hidden effective rank failed the first frozen Q-scaling attenuation test and is coordinate-dependent under a function-preserving reparameterization;
2. channel-standardized effective rank is invariant to that trivial rescaling, but failed the fresh preregistered Q-scaling mediator test.

The shortest current claim is therefore:

> **SGO is a finite-budget stochastic-environment subset-selection method. When only a subset of candidate environments can contribute to an update, retaining hard examples while reducing model-conditioned gradient redundancy can improve held-out mean performance in tested structured regimes. The internal functionally meaningful mediator remains unidentified.**

## 1. Seed is an environment index

Let `s` index a stochastic environment `e_s`. The relevant chain is

```text
s -> e_s -> g(theta, B, e_s) -> optimization trajectory -> learned function -> held-out behavior
```

with

```text
g(theta, B, e) = grad_theta L(theta; B, e).
```

SGO does not optimize integer seed values. It allocates finite learning budget across **model-conditioned stochastic environments**.

## 2. Finite-budget selection problem

Suppose each step exposes `K` candidate environments but only `Q < K` can contribute to the expensive backward update. Loss-hard selection prefers high current loss. Gradient-novel selection keeps a hard anchor while preferring additional candidates whose gradient signatures are less redundant with the already selected set.

An idealized novelty quantity is residual energy outside the selected gradient span:

```text
novelty(e | S) = ||g_e - P_S g_e||^2 / (||g_e||^2 + eps).
```

The implemented selector uses cheaper cosine/signature proxies. Its practical role is approximately

```text
error relevance x directional non-redundancy.
```

Hardness limits wasted budget on arbitrary unusual directions; novelty limits repeated expenditure on nearly identical errors.

## 3. Replicated direct evidence for a binding-budget mechanism

### First preregistered Q-scaling audit

With `K=16`, Q=`{2,4,8,12,16}`, n=30 fresh pairs:

| Q | gradnov - loss-hard held-out mean |
|---:|---:|
| 2 | +2.151 pp |
| 4 | +2.590 pp |
| 8 | +2.076 pp |
| 12 | +0.673 pp |
| 16 | 0.000 pp exactly |

Frozen low-Q minus high-Q attenuation:

- `A_B = +2.034 pp`;
- one-sided `p=1.06e-7`;
- Q=16 scientific identity: exact in all 30 pairs.

### Independent fresh Q-scaling replication

Issue #38 used fresh reps `300-329` and fresh held-out seeds `17000-17079` under the same K/Q/selector structure:

| Q | gradnov - loss-hard held-out mean |
|---:|---:|
| 2 | +2.613 pp |
| 4 | +2.443 pp |
| 8 | +2.156 pp |
| 12 | +0.885 pp |
| 16 | 0.000 pp exactly |

Frozen attenuation:

- `A_B = +2.086 pp`;
- SE `0.339 pp`;
- approximate 95% interval `+1.393 to +2.778 pp`;
- one-sided `p=5.16e-7`;
- Q=16 exact scientific mismatches: `0`.

Selected-gradient novelty again decayed toward zero with Q and became exactly zero at Q=16.

The replicated conclusion is:

> **The selector advantage depends strongly on subset-selection freedom. When the subset constraint disappears at Q=K, the two methods become exactly identical.**

This is currently the strongest direct mechanism evidence in the repository.

## 4. Why local descent does not explain the effect

Loss-hard can produce a larger immediate one-step loss decrease. Gradient novelty is therefore not best understood as a greedier optimizer of the next update.

If multiple hard environments induce approximately the same correction,

```text
g_1 ~= g_2 ~= g_3,
```

then several backward contributions can consume budget while adding little new directional coverage. SGO instead trades some immediate descent magnitude for broader coverage of unresolved directions.

## 5. Latent-factor interpretation

A useful approximation is

```text
g_e ~= sum_k a[e,k] v_k + eta_e,
```

where `v_k` are task-relevant correction directions, `a[e,k]` indicates which latent factors are active, and `eta_e` contains idiosyncratic/noisy components.

When Q is small relative to K, loss-only selection can repeatedly spend budget on the same dominant direction. Hard-plus-novelty selection can cover a broader set of unresolved directions. If train and held-out environments reuse latent factors in different combinations, this can improve held-out mean behavior.

This motivates the central abstraction:

> **SGO is approximately a budgeted coverage problem over unresolved, task-relevant gradient subspaces.**

The repository does not prove submodularity, global optimality, or a universal approximation guarantee.

## 6. Gradient diversity is not sufficient

The stronger rule

```text
more gradient effective rank -> more held-out benefit
```

is false. Breast-high is a direct counterexample: accumulated gradient effective rank increased strongly while held-out mean did not improve.

Thus useful selection is not generic diversity maximization. Directional novelty must remain task-relevant and must affect the learned function in a reusable way.

## 7. Raw representation effective rank: predictive marker, not causal quantity

Across separate prospective conditions, the frozen fixed-parameterization rule

```text
sign(delta raw hidden effective rank)
    -> sign(delta condition-average held-out mean benefit)
```

has repeatedly matched, including FashionMNIST Transformers and CIFAR/ResNet. This empirical record remains valid within its registered protocols.

However three later results sharply limit interpretation.

### 7.1 First Q-scaling mediator test

Benefit attenuation passed strongly, but raw-rank attenuation did not pass its frozen threshold (`+0.0735`, one-sided `p=0.1095`).

### 7.2 Function-preserving intervention

For the one-hidden-layer MLP, positive diagonal reparameterization

```text
A' = D A
b' = D b
W' = W D^-1
```

preserves the represented function while rescaling hidden coordinates.

Issue #35 changed raw effective rank by about `+2.1 to +2.3` under a spread intervention and `-5.8 to -6.1` under a concentration intervention, yet across all 80 intervention rows:

- predicted classes were exactly unchanged;
- all five held-out/clean accuracy metrics were exactly unchanged;
- maximum observed logit difference was `0.0`.

Therefore raw hidden effective rank is **coordinate-dependent and cannot itself be a functionally intrinsic causal quantity**.

### 7.3 Fresh Q-scaling replication

Raw rank was preregistered as secondary only. Its attenuation was nevertheless positive again (`+0.1780`, one-sided `p=0.0180`).

This is consistent with raw rank being a repeatable **trajectory marker under a fixed parameterization**, not a causal state variable.

## 8. Channel-standardized effective rank: scale-invariant but not the mediator

Issue #35 found that independently z-scaling nonconstant hidden channels before SVD made effective rank exactly invariant to the positive diagonal intervention.

Issue #38 therefore preregistered channel-standardized effective rank as a new mediator candidate in a fresh Q-scaling replication.

Mean gradnov-minus-loss-hard standardized-rank differences were:

| Q | standardized-rank delta |
|---:|---:|
| 2 | -0.0006 |
| 4 | +0.0815 |
| 8 | +0.0850 |
| 12 | +0.0863 |
| 16 | 0.0000 |

Frozen low-minus-high attenuation:

- `A_Z = -0.00271`;
- SE `0.1695`;
- approximate 95% interval `-0.3493 to +0.3439`;
- one-sided `p=0.5063`;
- standardized-mediator decision: **FAIL**.

Benefit attenuation in the same fresh run was strongly positive. Therefore removing trivial channel-scale dependence does **not** rescue effective rank as the quantitative mediator of the budget effect.

The next mechanism search should not simply invent another rank normalization.

## 9. Current causal picture

The evidence currently supports:

```text
binding subset budget
    -> non-redundant hard-gradient selection
    -> broader unresolved-direction coverage
    -> trajectory / learned-function difference
    -> held-out mean difference
```

The unresolved object is the middle mapping from trajectory coverage to a **functionally meaningful reusable change**.

A viable mediator should ideally satisfy all of the following:

- invariant to trivial function-preserving reparameterizations;
- measured using training-only information before held-out construction;
- vary with the finite-budget effect rather than only with training convention;
- prospectively predict or mediate held-out mean behavior;
- survive a materially different task or architecture.

## 10. Conditions under which SGO should help

The theory predicts benefit when most of the following hold:

- Q is substantially smaller than K;
- candidate environments induce materially different model-conditioned gradients;
- those differences contain reusable task structure rather than mostly noise;
- train and held-out environments share latent factors;
- the model can convert coverage into a reusable learned-function change;
- the selector retains hardness while reducing redundancy.

The binding-budget condition now has two preregistered n=30 confirmations on fresh blocks.

## 11. Predicted failure regimes

Weak, null, or negative effects are expected when:

- Q approaches K;
- candidate gradients are already redundant;
- novelty is dominated by nuisance/noise;
- train/held-out factors do not overlap;
- selected directions are mutually incompatible rather than reusable;
- novelty displaces necessary hard examples;
- the target is extreme tail safety rather than average coverage.

At Q=K=16, exact method identity has now been observed in both preregistered Q-scaling runs.

## 12. What remains empirically useful from representation rank

Raw representation rank should now be described narrowly:

> **A fixed-parameterization, condition-average directional marker with a strong prospective record, but not a coordinate-free causal law or validated quantitative mediator.**

The function-preserving intervention does not erase that predictive record. It changes what the record means.

Channel-standardized rank should not currently be promoted as a replacement: it passed scale invariance but failed the fresh budget-coupling test.

## 13. Highest-value next falsification program

### A. Function-space mediator

Measure a diagnostic defined by model behavior rather than hidden-coordinate scale. Candidate classes include training-only cross-environment probe transfer, class-conditional margin structure, or predictive disagreement across controlled environment combinations.

### B. Matched-coverage / different-reusability experiment

Construct two environment families with comparable selected-gradient non-redundancy but different latent-factor reuse. If held-out benefit differs, this isolates the conversion/reusability step after coverage.

### C. Structured-vs-unstructured matched novelty

Match gradient-novelty magnitude while changing whether the stochastic variation reflects reusable structure or unstructured nuisance.

### D. Cross-task budget scaling

Repeat the frozen low-vs-high Q contrast on a materially different task/architecture. This is now more valuable than a third same-task replication.

### E. Early-trajectory functional diagnostics

Freeze candidate function-space diagnostics early in training and test them prospectively on new conditions without calibrating on final held-out outcomes.

## Working claim

> **Under a binding finite update budget, selecting stochastic environments that are both currently hard and non-redundant in model-conditioned gradient space can improve coverage of unresolved learning directions and thereby improve held-out mean performance in tested structured regimes. The budget dependence has independently replicated. Raw representation effective rank is a useful but coordinate-dependent marker; channel-standardized effective rank is scale-invariant but failed as a quantitative mediator. The functionally meaningful internal mediator remains open.**
