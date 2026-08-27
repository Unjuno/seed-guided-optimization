# Trajectory-level mechanism pilot

## Question

Why does gradient-novel seed/environment selection produce large held-out gains under structured geometric shifts, but only tiny or negative changes on some stochastic tabular regimes?

This experiment tests whether the answer is visible in **training-only trajectory statistics**, rather than in one-step loss reduction or mean-gradient estimation.

## Design

Four previously used task families were rerun with 20 paired replicates each:

- Digits geometric shift
- independent synthetic classification
- Breast Cancer Wisconsin, low heterogeneity
- Breast Cancer Wisconsin, high heterogeneity

Within each replicate, initialization, minibatch order, and candidate-environment schedule are shared between:

- `loss_hard`: beta = 0
- `gradnov`: one hard anchor plus gradient novelty with beta = 1.5

Candidate count is K=16 and Q=4 environments contribute to the backward update.

All mechanism diagnostics are computed from the **training pool only**. Held-out environments are used only for final evaluation.

Recorded diagnostics include:

- effective rank of accumulated selected head-gradient directions;
- effective rank of hidden representations across fixed training environments;
- classification margins on training environments;
- cross-environment hidden-feature invariance;
- parameter displacement;
- approximate head-gradient path length and consecutive gradient cosine.

## Main result

The amount of gradient-space diversity itself does **not** explain held-out benefit.

| task | held-out mean delta | p10 delta | gradient effective-rank delta | representation effective-rank delta | rep/gradient conversion |
|---|---:|---:|---:|---:|---:|
| Digits geometric | +4.394 pp | +3.927 pp | +0.823 | +0.454 | +0.552 |
| Synthetic | +0.089 pp | +0.193 pp | +6.231 | +0.113 | +0.018 |
| Breast low | +0.040 pp | +0.056 pp | +3.362 | -0.165 | -0.049 |
| Breast high | -0.034 pp | -0.021 pp | +7.148 | -0.361 | -0.051 |

The strongest counterexample is Breast-high: gradient novelty increases accumulated gradient effective rank by about 7.15, much more than in Digits, while held-out mean does not improve and representation effective rank decreases.

Conversely, Digits shows the largest held-out gain while requiring only a small increase in gradient effective rank, but it produces the largest positive change in representation effective rank.

Across these four task means, the rank ordering of `delta representation effective rank` matches the rank ordering of held-out mean benefit. This four-task correlation is **descriptive only** because the candidate statistic was identified after inspecting these tasks.

At the replicate level, after centering deltas within task, `delta representation effective rank` has Spearman rho about 0.419 with `delta held-out mean` (p about 1.1e-4 over 80 paired replicates). A task-fixed-effect exploratory OLS with HC3 errors gives a positive representation-rank coefficient (p about .023). The corresponding accumulated-gradient-rank signal is substantially weaker and is not significant under the same exploratory fixed-effect comparison.

## What did not separate helpful from harmful regimes

- Larger accumulated gradient effective rank: **fails**; Breast-high increases it the most.
- Larger training margin: **fails**; margins rise strongly on Breast-high too.
- Greater feature invariance: **fails**; Breast-high improves this diagnostic despite no held-out mean benefit.
- Larger parameter displacement: **not a consistent separator**.

## Current mechanism hypothesis

Gradient novelty is useful when extra optimization-direction diversity is converted into a richer, reusable representation rather than merely producing more diverse local updates.

A useful diagnostic candidate is therefore not raw gradient diversity but a conversion quantity such as

`delta representation effective rank / delta accumulated gradient effective rank`.

This should be treated as a **frozen exploratory hypothesis**, not as a validated gate.

## Next falsification test

Before using this statistic for selector tuning, test it prospectively on an environment not used to propose it:

1. CIFAR-10 / ResNet trajectory audit, preferably on the already separated held-out primary protocol;
2. a fresh structured generator with different stochastic parameters;
3. optionally a larger architecture.

PASS requires the predeclared representation-rank direction to predict the sign or relative magnitude of novelty benefit without fitting to held-out performance. Failure on the next task should be retained as a negative result.
