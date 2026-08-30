# Budget-scaling falsification of the finite-gradient-coverage theory

## Status

Issue #32 preregistered a direct mechanism test of the working theory in `THEORETICAL_FRAMEWORK.md`: if gradient novelty primarily repairs incomplete coverage under a binding update budget, its held-out advantage should shrink as more of the same candidate set is allowed into each backward update.

Actions run `33312898169` completed all 30 paired replicates and the frozen aggregate.

**Frozen decision: PARTIAL PASS.**

The finite-budget benefit prediction passed strongly. The preregistered representation-effective-rank attenuation test did not pass.

## Frozen design

- Digits / geometric stochastic environments;
- MLP `64 -> 128 ReLU -> 10`;
- 64 training environments, seeds `13000-13063`;
- fresh held-out environments, seeds `15000-15079`;
- AdamW lr `1e-2`, weight decay `1e-3`;
- 10 epochs, batch 128;
- candidate `K=16` fixed;
- backward/update `Q in {2,4,8,12,16}`;
- loss-hard versus gradient novelty, novelty weight `0.6`;
- paired replicate IDs `100-129`, n=30;
- one-thread deterministic CPU execution;
- shared initialization, minibatch order, candidate schedule and training-only probe within each replicate.

Training-only representation effective rank was computed on a fixed 192-example probe concatenated across eight fixed training environments. Held-out environments were constructed only after training states and training-only diagnostics had been sealed.

## Frozen primary contrast

For each replicate and Q:

```text
B[r,Q] = heldout_mean(gradnov,Q) - heldout_mean(loss_hard,Q)
R[r,Q] = rep_rank(gradnov,Q) - rep_rank(loss_hard,Q)
```

Low and high budget contrasts were frozen as:

```text
B_low  = mean(B[2], B[4])
B_high = mean(B[12], B[16])
A_B    = B_low - B_high

R_low  = mean(R[2], R[4])
R_high = mean(R[12], R[16])
A_R    = R_low - R_high
```

Q=8 was excluded from the primary test and retained only as an intermediate descriptive condition.

## Identity control

At Q=16 every candidate environment enters the backward update, so loss-hard and gradient-novel selection must reduce to the same update set. Selected indices were explicitly sorted before backward.

Across all 30 paired replicates:

- scientific-output mismatches at Q=16: **0**;
- maximum absolute scientific difference: **0.0**;
- **IDENTITY PASS**.

This is an important internal control: the selector advantage disappears exactly when selection no longer removes any candidate from the update.

## Held-out benefit attenuation

Condition-average gradient-novel minus loss-hard held-out mean accuracy:

| Q | coverage Q/K | mean benefit | SE |
|---:|---:|---:|---:|
| 2 | 0.125 | **+2.151 pp** | 0.479 pp |
| 4 | 0.250 | **+2.590 pp** | 0.305 pp |
| 8 | 0.500 | **+2.076 pp** | 0.183 pp |
| 12 | 0.750 | **+0.673 pp** | 0.163 pp |
| 16 | 1.000 | **0.000 pp** | 0.000 pp |

The preregistered low-vs-high contrast was:

- mean `B_low`: **+2.370 pp**;
- mean `B_high`: **+0.337 pp**;
- mean attenuation `A_B`: **+2.034 pp**;
- SE: **0.302 pp**;
- approximate two-sided 95% t interval: **+1.417 to +2.651 pp**;
- t = **6.743**;
- one-sided p = **1.06e-7**;
- **COVERAGE PASS**.

The Q-specific sequence is not strictly monotone at low/intermediate Q, and strict monotonicity was not the frozen criterion. The preregistered low-vs-high attenuation is large and strongly positive.

## Manipulation check: selected-gradient non-redundancy

The mean gradient-novel minus loss-hard selected pairwise novelty difference decreases as the update covers more candidates:

| Q | delta selected pairwise novelty |
|---:|---:|
| 2 | +0.2802 |
| 4 | +0.1335 |
| 8 | +0.0583 |
| 12 | +0.0230 |
| 16 | 0.0000 |

This verifies that increasing Q progressively removes the selector's ability to choose a more non-redundant subset. At Q=16 there is no subset-selection freedom and the manipulation vanishes exactly.

## Representation-rank attenuation

The corresponding training-only representation-effective-rank differences were:

| Q | mean delta representation effective rank | SE |
|---:|---:|---:|
| 2 | +0.0622 | 0.1016 |
| 4 | +0.1971 | 0.0825 |
| 8 | +0.2196 | 0.0467 |
| 12 | +0.1122 | 0.0409 |
| 16 | 0.0000 | 0.0000 |

Frozen low-vs-high representation contrast:

- mean `R_low`: **+0.1296**;
- mean `R_high`: **+0.0561**;
- mean attenuation `A_R`: **+0.0735**;
- SE: **0.0585**;
- approximate two-sided 95% t interval: **-0.0462 to +0.1933**;
- t = **1.257**;
- one-sided p = **0.1095**;
- **REPRESENTATION-COUPLING FAIL under the preregistered threshold**.

The direction is compatible with attenuation but is too noisy to satisfy the frozen test. In addition, the largest mean rank difference occurs at Q=8 rather than the lowest budgets. Therefore representation effective rank should not be promoted as the direct quantitative mediator of budget dependence.

## Secondary relation

Across the 30 replicates, benefit attenuation and representation-rank attenuation had descriptive Pearson `r=0.365`, two-sided `p=0.0474`. This is secondary and does not upgrade the frozen representation decision.

## Frozen decision

The preregistered decision logic was:

- STRONG THEORY PASS = identity + benefit attenuation + representation attenuation;
- PARTIAL PASS = identity + benefit attenuation, representation attenuation fails;
- THEORY FAIL = identity passes, benefit attenuation fails;
- INVALID / REPRO FAILURE = identity fails.

Observed:

```text
IDENTITY PASS
COVERAGE PASS
REPRESENTATION-COUPLING FAIL
=> PARTIAL PASS
```

## Interpretation

This is strong evidence for the **finite-budget coverage component** of the theory. Gradient novelty provides large mean gains when only a small subset of K=16 candidate environments can contribute to each update, becomes much less valuable when Q is high, and becomes exactly identical to loss-hard when Q=K.

The same experiment weakens the stronger mediation story in which representation effective rank itself is expected to track the budget effect tightly. Effective rank remains useful as a condition-average directional marker in the separate prospective tests, but this audit does not support treating it as the sole or direct causal mediator.

The updated mechanism should therefore separate two claims:

1. **Supported mechanism component:** SGO gains depend strongly on a binding subset-selection budget and disappear when the full candidate set is used.
2. **Open mediator:** the internal representation quantity that converts better gradient-subspace coverage into generalization benefit is not yet identified causally; effective rank is an informative but incomplete proxy.

## Evidence

- Issue #32: preregistration.
- PR #33: implementation and frozen workflow.
- Actions run `33312898169`: six 5-replicate shards plus successful aggregate.
- `results/budget_coverage_decision30.csv`
- `results/budget_coverage_q_summary30.csv`
- `results/budget_coverage_attenuation30.csv`
- `results/budget_coverage_q16_identity30.csv`
