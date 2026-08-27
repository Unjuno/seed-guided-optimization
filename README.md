# Seed-Guided Optimization

**Gradient-aware selection of stochastic training environments under a finite compute budget.**

Random seeds are usually treated as reproducibility controls. This project studies a narrower optimization question: when a seed indexes a stochastic training environment, it changes the gradient trajectory. Seed-Guided Optimization (SGO) tests whether a fixed update budget can be allocated more effectively by retaining hard environments while reducing redundancy among model-conditioned gradient directions.

> **Status:** experimental research code. The strongest replicated results are on structured small-scale benchmarks. A fixed-protocol CIFAR-10 / ResNet-20 validation has 40 paired replicates with a corrected-significant held-out mean improvement, while tail metrics remain unconfirmed. A frozen training-only representation-rank direction rule has also passed seven prospective tests across five datasets, including a small FashionMNIST Transformer test. Negative and null results are retained.

## Research claim

For model parameters `θ`, minibatch/data state `B`, and stochastic environment seed `s`:

```text
s -> stochastic environment e_s -> gradient g(θ, B, s) -> optimization trajectory
```

SGO does **not** assume that seed integers have intrinsic semantic classes or that a universally "good seed" exists. The working claim is:

> Under some structured stochastic shifts, finite update budgets can be allocated more effectively by selecting hard but non-redundant environment-induced gradient directions rather than using loss-only selection.

The method is best understood as **model-conditioned stochastic-environment selection / trajectory shaping**, not seed-number optimization.

## Evidence at a glance

Headline comparisons use paired replicates and disjoint held-out environment seeds. Holm correction is used when five outcome metrics are tested as one family.

| Evidence | Result | Status |
|---|---|---|
| Digits MLP, geometric shifts | gradient-novel > parameter-novel by **+1.45 pp** held-out mean | supported |
| Small CNN replication | gradient-novel > loss-hard by **+2.25 pp** mean and **+2.57 pp** minimum; both Holm-significant | supported |
| Optimizer replication | gains survive independently tuned AdamW and SGD+momentum | supported |
| RNG candidate compression | moderate compression reduces gradient evaluations; 16→4 damages tail coverage | supported |
| Learned RNG relevance | training-only gradient information recovers useful generator/model-conditioned RNG dimensions | supported with scope limits |
| Relative redundancy control | absolute cosine targets fail cross-task; normalized relative targets transfer better in tested tasks | supported with scope limits |
| CIFAR-10 / ResNet-20, 40 pairs | mean **+0.1206 pp**, Holm(5) `p=0.01336`; p10/min positive but not corrected-significant | supported for mean only |
| Trajectory mechanism | raw gradient-rank expansion does not predict benefit; representation effective-rank change is a better candidate | exploratory mechanism evidence |
| Prospective representation-rank rule | **7/7** registered direction matches across five datasets, including negative Diabetes and FashionMNIST/Tiny Transformer architecture-shift tests | prospectively supported, not universal proof |

See [`docs/RESULTS.md`](docs/RESULTS.md), [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md), and [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Core selector

The strongest practical selector keeps a hard anchor and adds candidates with non-redundant head-gradient directions. A generic score is

```text
score_i = z(loss_i) + beta * z(novelty_i)
```

where novelty is computed from the distance/cosine separation between the candidate gradient signature and already selected directions.

A fixed absolute redundancy target does not transfer reliably across tasks. The tested relative controller instead places the target within each step's feasible cosine range:

```text
c_target = c_strong_novelty + rho * (c_hardness - c_strong_novelty)
```

The tested `rho=0.15` is **not** claimed to be universal.

## External validation: CIFAR-10 / ResNet-20

The pre-committed primary protocol was extended from 20 to **40 paired replicates** without changing selector, optimizer, data size, K/Q budget, or held-out environment definition.

Gradient-novel minus loss-hard:

- held-out mean: **+0.1206 pp**, raw `p=0.002672`, Holm(5) `p=0.01336`;
- p10: **+0.1213 pp**, Holm `p=0.05294`;
- minimum: **+0.1875 pp**, Holm `p=0.08815`;
- clean: **+0.0658 pp**, Holm `p=0.63998`;
- environment SD: -0.0142 pp, Holm `p=0.63998`.

The safe claim is therefore narrow: **the held-out mean improvement is corrected-significant under this fixed protocol; CIFAR tail robustness is not confirmed.**

See [`docs/CIFAR_RESNET_PRIMARY.md`](docs/CIFAR_RESNET_PRIMARY.md).

## Mechanism: trajectory shaping, not one-step greediness

Several simpler explanations are ruled out or weakened:

- Gradient novelty is **not** simply a better estimator of the mean expected gradient than random sampling.
- Loss-hard can produce a larger immediate one-step loss decrease, so greedy next-step improvement does not explain the final gains.
- Larger accumulated selected-gradient effective rank is **not sufficient**: Breast-high increases it strongly while held-out mean does not improve.
- Pure diversity without a hard anchor can hurt.
- Full-network gradient signatures were slower without benefit over the tested final-layer proxy.

Across four discovery tasks, representation effective-rank change tracked the sign/magnitude of mean held-out benefit better than gradient-rank change. That led to a frozen candidate rule:

> **sign(delta hidden-representation effective rank) predicts sign(mean held-out benefit).**

The rule has now matched seven prospective direction tests across five datasets. The record includes a negative Diabetes regression prediction and a positive FashionMNIST/Tiny Transformer test that changed dataset, stochastic generator, and representation architecture. Three tests still share Digits, and the Transformer is deliberately small, so this is not a calibrated universal gate.

In the FashionMNIST test, mean representation-rank difference was **+0.0542** and mean held-out accuracy benefit was **+0.736 pp** across 10 paired runs, yielding a preregistered **PASS**.

See [`docs/TRAJECTORY_MECHANISM.md`](docs/TRAJECTORY_MECHANISM.md), [`docs/PROSPECTIVE_REPRESENTATION_RANK.md`](docs/PROSPECTIVE_REPRESENTATION_RANK.md), and [`docs/FASHION_TRANSFORMER_REP_RANK.md`](docs/FASHION_TRANSFORMER_REP_RANK.md).

## Negative results are part of the repository

The evidence does **not** support these stronger claims:

- seed integers form semantic clusters;
- a universally good seed exists;
- more candidates/seeds are always better;
- gradient diversity alone guarantees generalization gains;
- worst-only training is robustly best;
- learned RNG fingerprints transfer universally;
- a fixed absolute gradient-cosine target transfers universally;
- the representation-rank rule is already a universal/calibrated gate;
- the attempted per-environment representation-rank SD tail rule works;
- CIFAR p10/worst robustness is confirmed;
- one small Transformer establishes general Transformer or large-model validity;
- CPU wall-clock results determine GPU-optimal settings.

## Repository map

```text
.
├── README.md
├── CITATION.cff
├── LICENSE                  # Apache-2.0
├── requirements.txt
├── docs/
│   ├── README.md
│   ├── RESEARCH_STATUS.md
│   ├── METHODS.md
│   ├── RESULTS.md
│   ├── LIMITATIONS.md
│   ├── TRAJECTORY_MECHANISM.md
│   ├── PROSPECTIVE_REPRESENTATION_RANK.md
│   ├── FASHION_TRANSFORMER_REP_RANK.md
│   └── ...
├── experiments/
│   ├── README.md
│   ├── common.py
│   └── *.py
└── results/
    ├── README.md
    └── *.csv
```

## Reproduction

Tested public CPU stack includes PyTorch 2.10, NumPy 2.3.x, pandas 2.2.x, SciPy 1.17, and scikit-learn 1.8. CIFAR/FashionMNIST workflows additionally pin torchvision 0.25.0.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start with the primary structured benchmark:

```bash
python experiments/mlp_geometric.py --start 0 --end 20 --output mlp_geometric.csv
```

Then follow [`experiments/README.md`](experiments/README.md). Accuracy-like CSV fields are stored on `[0,1]`; `0.01` equals one percentage point.

## Current roadmap

1. Complete the independently preregistered CIFAR-10 / ResNet-20 representation-rank falsification audit.
2. Continue prospective rank-rule falsification on a larger/different architecture beyond the Tiny Transformer.
3. Find a new training-only tail-safety diagnostic; the previous rep-rank-SD rule is retired.
4. Run GPU-vectorized wall-clock comparisons at matched update and candidate-evaluation budgets.
5. Extend beyond handcrafted stochastic corruption generators.

## Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). Licensed under the **Apache License 2.0**.
