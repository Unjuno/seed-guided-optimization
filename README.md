# Seed-Guided Optimization

**Gradient-aware selection of stochastic training environments under a finite compute budget.**

Random seeds are usually treated as reproducibility controls. This project studies a narrower optimization question: when a seed indexes a stochastic training environment, it also changes the gradient trajectory. Instead of consuming candidate environments uniformly or selecting only the highest-loss ones, Seed-Guided Optimization (SGO) tests whether updates improve when training keeps **hard** environments while reducing **gradient-direction redundancy**.

> **Status:** experimental research code. Strong results exist on structured small-scale benchmarks; a CIFAR-10 / ResNet-20 validation is currently suggestive but not confirmatory after multiple-outcome correction. This repository intentionally keeps negative and null results.

## Research claim

For model parameters `θ`, minibatch `B`, and stochastic environment seed `s`, let

```text
s -> stochastic environment -> gradient g(θ, B, s) -> optimization trajectory
```

SGO does **not** assume that seed integers have intrinsic semantic classes or that a universally "good seed" exists. The working claim is:

> Under some structured stochastic shifts, finite update budgets can be allocated more effectively by selecting hard but non-redundant environment-induced gradient directions rather than using loss-only selection.

The method is best understood as **model-conditioned stochastic-environment selection**, not seed-number optimization.

## Evidence at a glance

All headline comparisons use paired replicates and disjoint held-out environment seeds. Holm correction is used when five outcome metrics are tested as one family.

| Evidence | Result | Status |
|---|---|---|
| Digits MLP, geometric shifts | gradient-novel > parameter-novel by **+1.45 pp** held-out mean | supported |
| Small CNN replication | gradient-novel > loss-hard by **+2.25 pp** mean and **+2.57 pp** minimum; both Holm-significant | supported |
| Optimizer replication | gains survive tuned AdamW and tuned SGD+momentum | supported |
| RNG candidate compression | moderate compression can reduce gradient evaluations; 16→4 damages tail coverage | supported |
| Learned RNG relevance | training-only learned relevance approaches oracle filtering in two tested handcrafted generators; old-generator fingerprint transfer fails | supported with scope limits |
| Relative redundancy control | absolute cosine targets fail cross-task; normalized relative targets transfer better across Digits and Synthetic | supported with scope limits |
| CIFAR-10 / ResNet-20, first 20 pairs | mean **+0.138 pp**, p10 **+0.143 pp**; raw p<.05 but Holm p=.112/.130 | suggestive, not confirmatory |

See [`docs/RESULTS.md`](docs/RESULTS.md) for the complete result summary and [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md) for supported, negative, and in-progress experiments.

## Mechanistic findings

Several simple explanations have been ruled out:

- Gradient novelty is **not** merely a better estimator of the mean expected gradient than random sampling.
- Loss-hard selection produces larger immediate one-step loss reduction in the tested audit, yet can generalize worse over the full trajectory.
- Gradient novelty better covers the tested tail/robust gradient direction than random selection.
- Pure diversity without a hard anchor is weak.
- Full-network gradient signatures were slower and did not beat the cheaper final-layer signature in the tested MLP.

The current interpretation is therefore trajectory-oriented: SGO shapes which stochastic directions receive updates, rather than greedily maximizing the next loss decrease.

## Main results

### Gradient novelty vs. physical environment diversity

On the structured Digits geometric-shift benchmark:

| Selector | Held-out mean | p10 |
|---|---:|---:|
| Loss-hard | 50.21% | 39.15% |
| Parameter-novel | 51.66% | 40.98% |
| **Gradient-novel** | **53.11%** | **41.78%** |

Gradient-novel and parameter-novel selections had nearly the same physical transformation-space diversity, but gradient novelty produced the higher held-out mean. This supports a **model-conditioned** notion of environment diversity.

### CNN and optimizer replication

A small CNN on the same structured shifts replicated the main effect: gradient novelty improved held-out mean by **+2.25 pp** and minimum-environment accuracy by **+2.57 pp** versus loss-hard across 20 paired runs; both survived five-metric Holm correction.

After independent optimizer tuning, the MLP effect also survived:

| Optimizer | Mean gain vs. loss-hard | p10 gain |
|---|---:|---:|
| AdamW | **+2.69 pp** | **+2.94 pp** |
| SGD+momentum | **+1.88 pp** | **+3.02 pp** |

### RNG prefiltering and learned fingerprints

The project separates four objects:

1. seed integer;
2. finite RNG output sequence;
3. stochastic environment generated from that sequence;
4. model-dependent gradient signature induced by the environment.

Long raw RNG fingerprints can be actively harmful when they include coordinates unrelated to the environment. Moderate candidate prefiltering works when the fingerprint tracks relevant stochastic variation. Training-only gradient information can also learn useful RNG relevance; however, that relevance is **generator/model-conditioned**, not a universal property of seed values.

### Relative redundancy control

A fixed absolute selected-gradient cosine target did not transfer from Digits to Synthetic because the feasible cosine ranges differed substantially. A normalized controller instead uses

```text
c_target = c_strong_novelty + ρ (c_hardness - c_strong_novelty)
```

within each step's feasible range. Reusing `ρ = 0.15` without held-out tuning worked across the tested Digits and Synthetic tasks, while the absolute target saturated on Synthetic. This is not evidence that `ρ = 0.15` is universal.

### CIFAR-10 / ResNet-20

The first 20 paired primary replicates are intentionally reported as **inconclusive but positive-direction evidence**:

- held-out mean: **+0.1380 pp**, raw `p=.0224`, Holm `p=.1120`;
- p10: **+0.1433 pp**, raw `p=.0324`, Holm `p=.1298`;
- minimum: +0.1900 pp, Holm `p=.6268`;
- clean: +0.0733 pp, Holm `p=.9432`.

The exact protocol is being extended to 40 paired replicates in [PR #11](https://github.com/Unjuno/seed-guided-optimization/pull/11). No selector or evaluation rule is being changed for the extension.

## Negative results are part of the repository

The evidence does **not** support the following stronger claims:

- more seeds are always better;
- worst-only training is robustly best;
- pure gradient diversity is sufficient;
- seed integers themselves form meaningful semantic clusters;
- a selector or RNG fingerprint learned on one task/generator transfers universally;
- longer RNG fingerprints are automatically better;
- mean-gradient estimation or one-step loss reduction alone explains the final gains;
- CPU wall-clock optima determine GPU optima.

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) and [`docs/GRADIENT_DIRECTION_AUDIT.md`](docs/GRADIENT_DIRECTION_AUDIT.md).

## Repository map

```text
.
├── README.md                 # public overview and claim boundary
├── CITATION.cff
├── LICENSE                   # Apache-2.0
├── requirements.txt
├── docs/
│   ├── README.md             # documentation index
│   ├── RESEARCH_STATUS.md    # supported / negative / in-progress matrix
│   ├── METHODS.md            # experimental protocol
│   ├── RESULTS.md            # result narrative
│   ├── LIMITATIONS.md        # claims to avoid and open gaps
│   ├── GRADIENT_DIRECTION_AUDIT.md
│   ├── RNG_CROSS_GENERATOR.md
│   ├── ADAPTIVE_BETA.md
│   ├── RELATIVE_REDUNDANCY_CONTROL.md
│   └── CIFAR_RESNET_PRIMARY.md
├── experiments/
│   ├── README.md             # script map and reproduction order
│   ├── common.py
│   └── *.py                  # committed reproduction scripts
├── results/
│   ├── README.md             # evidence-file index and conventions
│   └── *.csv                 # committed result snapshots
└── .github/workflows/        # CI-backed long-running validations
```

## Reproduction

Tested public CPU stack includes PyTorch 2.10, NumPy 2.3.x, pandas 2.2.x, SciPy 1.17, and scikit-learn 1.8. Install from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start with the primary structured benchmark:

```bash
python experiments/mlp_geometric.py --start 0 --end 20 --output mlp_geometric.csv
```

Then follow [`experiments/README.md`](experiments/README.md). Accuracy-like CSV fields are stored on `[0, 1]`; `0.01` equals one percentage point.

## Current roadmap

- [Issue #1](https://github.com/Unjuno/seed-guided-optimization/issues/1): finish the 40-pair CIFAR-10 / ResNet validation.
- [Issue #2](https://github.com/Unjuno/seed-guided-optimization/issues/2): GPU-vectorized wall-clock benchmark.
- [Issue #12](https://github.com/Unjuno/seed-guided-optimization/issues/12): identify trajectory-level conditions that predict when novelty helps.

Completed roadmap item: RNG fingerprint discovery without generator-coordinate labels is tracked in closed [Issue #3](https://github.com/Unjuno/seed-guided-optimization/issues/3).

## Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff).

Licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE).
