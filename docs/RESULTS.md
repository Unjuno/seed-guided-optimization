# Results summary

The repository separates **supported**, **suggestive**, and **negative/null** findings. Headline comparisons are paired, use disjoint held-out environment seeds, and apply Holm correction when five outcome metrics are treated as one family.

## Supported findings

### 1. Seed/environment choice changes optimization trajectory

With the underlying training dataset held fixed, stochastic environments induced by different seeds produce different gradients and different final held-out performance. This is the basic empirical premise of Seed-Guided Optimization.

### 2. Hardness and gradient novelty are not the same signal

Loss-hard selection concentrates on immediately difficult environments. Gradient-novel selection retains a hard anchor while adding environments whose final-layer gradient signatures are less redundant.

On the structured Digits geometric-shift benchmark, gradient novelty outperformed transformation-parameter novelty even when the selected batches had nearly the same physical transformation-space diversity. The paired held-out mean advantage was about **+1.45 percentage points**.

### 3. Small-CNN architecture replication

Across 20 paired runs on the same structured geometric environment family, gradient-novel vs. loss-hard improved:

- held-out mean by **+2.25 pp** (`Holm p=0.0171`);
- minimum-environment accuracy by **+2.57 pp** (`Holm p=0.00573`).

The p10 improvement was positive but did not survive the five-metric correction. Parameter novelty also helped the CNN, so the stronger MLP result that gradient novelty clearly dominates physical parameter novelty is not architecture-general.

### 4. Optimizer replication after independent tuning

The initial SGD+momentum comparison was confounded by under-tuning. A separate loss-hard-only learning-rate sweep was therefore completed before the final selector comparison.

Gradient-novel then improved held-out performance under both tested optimizers:

- AdamW: mean **+2.69 pp** (`Holm p=5.1e-5`), p10 **+2.94 pp** (`Holm p=0.00142`);
- SGD+momentum: mean **+1.88 pp** (`Holm p=0.0362`), p10 **+3.02 pp** (`Holm p=0.00250`).

This is evidence against an AdamW-only explanation, not proof of optimizer universality.

### 5. RNG prefiltering reduces expensive candidate evaluation only when the fingerprint is relevant

A seed integer, its finite RNG outputs, the stochastic environment generated from them, and the resulting model gradient are treated as distinct objects.

In the geometric benchmark, using RNG outputs that actually drive the environment permits moderate candidate compression before gradient evaluation. Adding unrelated RNG outputs degrades the distance metric. Compression also has a real failure boundary: reducing 16 candidates directly to 4 sharply damages lower-tail performance.

A conservative tested point is **16 initial candidates → 12 cheaply prefiltered candidates → 4 backward environments**.

### 6. RNG relevance can be learned without generator-coordinate labels

A training-only ridge relevance model was fit from a 64-value RNG window to gradient-derived targets. In the original generator, learned top-coordinate filtering materially outperformed raw64 distance and approached the oracle relevant-coordinate representation.

A second generator moved the relevant coordinates to non-contiguous positions. Reusing the old fingerprint failed, while re-learning relevance on the new generator recovered the strong coordinates. A soft relevance-weighted top-12 fingerprint improved over raw64 by approximately:

- held-out mean **+1.37 pp**;
- p10 **+3.02 pp**;
- worst environment **+3.40 pp**;

with the tested differences surviving Holm correction. The weighted representation was not significantly different from the oracle-seven representation on the five tested held-out metrics.

Interpretation: useful RNG relevance can be discovered, but it is **generator/model-conditioned** rather than a universal property of seed values.

### 7. Relative gradient-redundancy control transfers better than an absolute cosine target

A controller that attempted to maintain an absolute selected-gradient cosine target of `0.15` transferred poorly from Digits to Synthetic because Synthetic's feasible gradient-cosine range was much higher; the controller saturated at the maximum novelty weight.

The revised controller defines a target within the step-specific feasible range:

```text
c_target = c_strong_novelty + rho * (c_hardness - c_strong_novelty)
```

Using the same `rho=0.15` without held-out tuning:

- on Digits, relative control improved held-out mean by **+4.63 pp**, p10 by **+3.58 pp**, and worst by **+3.92 pp** versus beta=0, all significant after Holm correction;
- on Synthetic, the controller no longer saturated and was statistically indistinguishable from the tested fixed beta=1.5/3 operating points on the five held-out metrics.

This supports normalization across a task's feasible gradient geometry. It does not establish a universal value of `rho`.

### 8. Mechanism audit: the effect is not simple mean-gradient estimation or greedy one-step improvement

Direct gradient audits produced important negative controls:

- gradient-novel selection did **not** significantly beat random selection as an estimator of the mean expected gradient;
- it aligned better with the tested tail/robust gradient direction;
- loss-hard selection produced larger immediate one-step average loss reduction than gradient novelty.

Therefore the long-run held-out benefit is not adequately explained as either superior unbiased mean-gradient estimation or maximal next-step loss reduction. The current interpretation is trajectory-oriented: environment selection changes which stochastic directions are repeatedly represented in the optimization path.

## Suggestive / inconclusive result

### CIFAR-10 / ResNet-20 primary validation

The first 20 paired primary replicates completed successfully under a fixed protocol using 6,000 training images, 3,000 test images, ResNet-20, 64 training environment seeds, `K=8` candidates and `Q=4` backward environments.

Gradient-novel minus loss-hard:

- held-out mean **+0.1380 pp**, raw `p=.0224`, Holm `p=.1120`;
- p10 **+0.1433 pp**, raw `p=.0324`, Holm `p=.1298`;
- minimum +0.1900 pp, Holm `p=.6268`;
- clean +0.0733 pp, Holm `p=.9432`;
- environment SD essentially unchanged.

This is **positive-direction but not confirmatory** after correction. The exact protocol is being extended to 40 paired replicates in PR #11 without changing selector, optimizer, data size, K/Q budget, or held-out environment definition.

## Negative and null results retained

- Worst-only / very narrow tail objectives can over-focus on outlier environments.
- Pure gradient diversity without hardness can hurt tail performance.
- Full-network gradient signatures did not improve selection over the cheaper final-layer signature in the tested MLP.
- A fixed selector learned on one task did not transfer universally.
- Long raw RNG fingerprints containing irrelevant coordinates are worse than compact relevant fingerprints.
- An old learned RNG fingerprint did not transfer after the generator's relevant coordinates were moved.
- Directly predicting a compact gradient embedding from raw RNG was weaker than learning coordinate relevance in the tested setup.
- Low-heterogeneity tasks can show little benefit from active seed selection.
- Absolute gradient-cosine feedback can be infeasible across tasks.
- An under-tuned optimizer can create a false selector failure; optimizer tuning must be separated from selector evaluation.
- One-step mean-gradient quality and one-step loss reduction are insufficient mechanism explanations.

## Public claim boundary

The current experiments justify saying:

> Gradient-aware stochastic-environment selection can improve held-out optimization/generalization relative to loss-only selection under some structured stochastic shifts in the tested settings. Training-only information can also identify useful RNG relevance for candidate prefiltering in the tested generators.

They do **not** justify claiming a universally optimal seed family, semantic seed-number classes, a universal selector/controller, modern large-scale validation, or a GPU wall-clock advantage.

For a compact matrix, see [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md). For current gaps, see [`LIMITATIONS.md`](LIMITATIONS.md).
