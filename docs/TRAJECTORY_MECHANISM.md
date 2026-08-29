# Trajectory-level mechanism

## Question

Why does gradient-novel seed/environment selection produce large held-out gains under structured geometric shifts, but only tiny or negative changes on some stochastic regimes?

The mechanism program tests whether the answer is visible in **training-only trajectory and representation statistics**, rather than in one-step loss reduction or mean-gradient estimation.

## Discovery design

Four previously used task families were rerun with 20 paired replicates each:

- Digits geometric shift;
- independent synthetic classification;
- Breast Cancer Wisconsin, low heterogeneity;
- Breast Cancer Wisconsin, high heterogeneity.

Within each replicate, initialization, minibatch order, and candidate-environment schedule were shared between:

- `loss_hard`: beta = 0;
- `gradnov`: one hard anchor plus gradient novelty with beta = 1.5.

Candidate count was K=16 and Q=4 environments contributed to the backward update. All mechanism diagnostics were computed from the **training pool only**.

Recorded diagnostics included:

- effective rank of accumulated selected head-gradient directions;
- effective rank of hidden representations across fixed training environments;
- classification margins;
- cross-environment hidden-feature invariance;
- parameter displacement;
- approximate head-gradient path length and consecutive gradient cosine.

## Main discovery result

The amount of gradient-space diversity itself does **not** explain held-out benefit.

| task | held-out mean delta | p10 delta | gradient effective-rank delta | representation effective-rank delta | rep/gradient conversion |
|---|---:|---:|---:|---:|---:|
| Digits geometric | +4.394 pp | +3.927 pp | +0.823 | +0.454 | +0.552 |
| Synthetic | +0.089 pp | +0.193 pp | +6.231 | +0.113 | +0.018 |
| Breast low | +0.040 pp | +0.056 pp | +3.362 | -0.165 | -0.049 |
| Breast high | -0.034 pp | -0.021 pp | +7.148 | -0.361 | -0.051 |

The strongest counterexample is Breast-high: gradient novelty increases accumulated gradient effective rank by about 7.15, much more than in Digits, while held-out mean does not improve and representation effective rank decreases.

Conversely, Digits shows the largest held-out gain while requiring only a small gradient-rank increase, but it produces the largest positive representation-rank change.

Across these four task means, the rank ordering of `delta representation effective rank` matched the rank ordering of held-out mean benefit. This discovery association is descriptive because the statistic was identified after inspecting these tasks.

At the replicate level, after centering deltas within task, `delta representation effective rank` had Spearman rho about 0.419 with `delta held-out mean` (`p` about `1.1e-4` over 80 paired replicates). A task-fixed-effect exploratory OLS with HC3 errors gave a positive representation-rank coefficient (`p` about `.023`). The corresponding accumulated-gradient-rank signal was substantially weaker under the same exploratory comparison.

## Controls that failed to explain the effect

- **Better mean-gradient estimation:** not supported by the gradient-direction audit.
- **Largest one-step improvement:** not supported; loss-hard can produce larger immediate one-step loss decrease.
- **Larger accumulated gradient effective rank:** fails; Breast-high increases it the most.
- **Larger training margin:** fails; margins also rise strongly in Breast-high.
- **Greater feature invariance:** fails as a sufficient explanation.
- **Larger parameter displacement:** not a consistent separator.

## Updated mechanism hypothesis

The original pilot phrased the candidate as "gradient diversity converted into richer representation." The current theory is more precise:

> Under a finite update/evaluation budget, hard-plus-novelty selection can improve **coverage of unresolved task-relevant gradient directions**. Benefit occurs when that additional coverage is converted by training into **reusable representation structure**, rather than remaining as diverse local updates or nuisance/noise responses.

An idealized novelty quantity is residual gradient energy outside the selected span,

```text
||g_e - P_S g_e||^2 / ||g_e||^2,
```

while the implemented selector uses cheaper gradient-signature/cosine proxies.

This theory explains why hardness remains necessary: novelty without task relevance can spend budget on unusual but useless directions.

See [`THEORETICAL_FRAMEWORK.md`](THEORETICAL_FRAMEWORK.md) for the full formalization, predicted success/failure regimes, and causal falsification program.

## Prospective falsification sequence

The discovery statistic was frozen into the directional rule

> `sign(delta representation effective rank) -> sign(condition-average mean held-out benefit)`.

It was then tested without fitting to held-out outcomes.

### Original six tests

Three new Digits generators plus Wine, Iris, and Diabetes all matched the predeclared sign. Diabetes was especially informative: gradient rank rose strongly, representation rank fell, and held-out benefit was negative as predicted.

### FashionMNIST / Tiny Transformer

The architecture/dataset/generator-shift test produced:

- delta representation rank `+0.05415`;
- held-out mean benefit `+0.7363 pp`;
- frozen decision **PASS**.

An independent exact-protocol 20-pair extension then gave:

- delta representation rank `+0.07853`;
- held-out mean benefit `+0.4377 pp`;
- frozen decision **REPLICATES**.

Per-run sign matches were imperfect, supporting a condition-average interpretation rather than a run-level gate.

### CIFAR-10 / ResNet-20

A separately preregistered independent 10-pair audit produced:

- training-only delta representation rank `+0.03205`, above the `0.01` tolerance;
- sealed direction = positive;
- held-out mean benefit `+0.1703 pp`;
- frozen decision **PASS**.

The rank estimate itself was noisy (descriptive `p=0.1464`), and 3 of 9 non-uncertain individual reps disagreed with the condition-level rule. Again, the evidence is for aggregate direction only.

## Current interpretation

The registered condition-level record is now **8/8 across six datasets**. Three tests share Digits, so the count is not eight independent datasets.

The strongest evidence currently supports this narrower chain:

```text
selector
  -> changes which unresolved gradient directions enter the finite-budget trajectory
  -> sometimes changes representation geometry in a reusable direction
  -> condition-average held-out mean changes with the same sign.
```

What remains unresolved is **causality**. Representation effective rank may be a marker of another useful geometry rather than the causal mediator itself.

## Next mechanism tests

The highest-value tests now attack the theory directly:

1. **Mediation:** test whether selector-induced benefit statistically/experimentally passes through representation changes after controlling for gradient coverage.
2. **Representation intervention:** modify representation geometry without SGO and test whether held-out benefit follows.
3. **Matched structured-vs-unstructured novelty:** hold novelty magnitude approximately fixed while changing whether the novel directions correspond to reusable latent factors.
4. **Budget scaling:** test whether the SGO advantage shrinks as ordinary sampling receives enough budget to cover the relevant subspace.
5. **Early-trajectory proxy:** search for a diagnostic available before fully training both selectors, freeze it on existing development tasks, then prospectively test on a new architecture/dataset.

The failed representation-rank-SD tail rule remains retired and must not be recycled without a new theory.
