# Seed-Guided Optimization

**Gradient-aware selection of stochastic training environments under a finite compute budget.**

Random seeds are usually treated as reproducibility controls. This project studies a narrower optimization question: when a seed indexes a stochastic training environment, it changes the gradient trajectory. Seed-Guided Optimization (SGO) tests whether a fixed update budget can be allocated more effectively by retaining hard environments while reducing redundancy among model-conditioned gradient directions.

> **Status:** experimental research code. Structured small-scale benchmarks show the strongest effects. A fixed-protocol CIFAR-10 / ResNet-20 study with 40 paired replicates supports a small corrected-significant held-out mean improvement, while CIFAR tail metrics remain unconfirmed. A frozen training-only representation-rank direction rule has now matched **8 registered prospective conditions across 6 datasets**, including FashionMNIST/Tiny Transformer and CIFAR-10/ResNet-20. The Fashion condition also replicated in 20 independent new pairs. The rule is supported as a **condition-average directional predictor**, not a per-run gate or causal law.

## Research claim

For model parameters `theta`, minibatch/data state `B`, and stochastic environment seed `s`:

```text
s -> stochastic environment e_s -> gradient g(theta, B, e_s)
  -> optimization trajectory -> representation -> held-out behavior
```

SGO does **not** assume that seed integers have intrinsic semantic classes or that a universally "good seed" exists. The working claim is:

> Under some structured stochastic shifts and a finite learning budget, selecting environments that are both currently hard and non-redundant in model-conditioned gradient space can improve coverage of unresolved task-relevant directions. When that coverage is converted into a richer reusable hidden representation, held-out **mean** performance tends to improve in the tested regimes.

The method is best understood as **model-conditioned stochastic-environment selection / trajectory shaping**, not seed-number optimization.

See [`docs/THEORETICAL_FRAMEWORK.md`](docs/THEORETICAL_FRAMEWORK.md) for the current theory and explicit falsification predictions.

## Evidence at a glance

Headline comparisons use paired replicates and disjoint held-out environment seeds. Holm correction is used when five outcome metrics are tested as one family.

| Evidence | Result | Status |
|---|---|---|
| Digits MLP, geometric shifts | gradient-novel > parameter-novel by **+1.45 pp** held-out mean | supported |
| Small CNN replication | gradient-novel > loss-hard by **+2.25 pp** mean and **+2.57 pp** minimum; both Holm-significant | supported |
| Optimizer replication | gains survive independently tuned AdamW and SGD+momentum | supported |
| RNG candidate compression | moderate compression reduces gradient evaluations; 16->4 damages tail coverage | supported |
| Learned RNG relevance | training-only gradient information recovers useful generator/model-conditioned RNG dimensions | supported with scope limits |
| Relative redundancy control | absolute cosine targets fail cross-task; normalized relative targets transfer better in tested tasks | supported with scope limits |
| CIFAR-10 / ResNet-20 primary, 40 pairs | mean **+0.1206 pp**, Holm(5) `p=0.01336`; p10/min positive but not corrected-significant | supported for mean only |
| Trajectory mechanism | raw gradient-rank expansion does not predict benefit; representation effective-rank change is the stronger candidate | exploratory mechanism evidence |
| Prospective representation-rank rule | **8/8 registered condition-level direction matches across 6 datasets**; CIFAR/ResNet test also PASS | prospectively supported, not universal proof |
| FashionMNIST / Tiny Transformer independent extension | new reps 10-29: rank `+0.07853`, mean `+0.4377 pp`; frozen direction **REPLICATES** | independent replication |
| Hosted-CPU reproducibility audit | one thread did **not** remove cross-run drift; frozen decision **DRIFT PERSISTS**; aggregate directions stayed positive | execution limitation |

See [`docs/RESULTS.md`](docs/RESULTS.md), [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md), and [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Core selector

The strongest practical selector keeps a hard anchor and adds candidates with non-redundant head-gradient directions. A generic score is

```text
score_i = z(loss_i) + beta * z(novelty_i)
```

where novelty is computed from distance/cosine separation between the candidate gradient signature and already selected directions.

A fixed absolute redundancy target does not transfer reliably across tasks. The tested relative controller instead places the target within each step's feasible cosine range:

```text
c_target = c_strong_novelty + rho * (c_hardness - c_strong_novelty)
```

The tested `rho=0.15` is **not** claimed to be universal.

## External validation: CIFAR-10 / ResNet-20

### Primary 40-pair benefit result

Gradient-novel minus loss-hard under the unchanged primary protocol:

- held-out mean: **+0.1206 pp**, raw `p=0.002672`, Holm(5) `p=0.01336`;
- p10: **+0.1213 pp**, Holm `p=0.05294`;
- minimum: **+0.1875 pp**, Holm `p=0.08815`;
- clean: **+0.0658 pp**, Holm `p=0.63998`;
- environment SD: `-0.0142 pp`, Holm `p=0.63998`.

The safe claim is narrow: **the held-out mean improvement is corrected-significant under this fixed protocol; CIFAR tail robustness is not confirmed.**

### Independent prospective representation-rank audit

A separately preregistered 10-pair audit (reps 40-49) froze the training-only rule before held-out evaluation:

- mean delta representation effective rank: **+0.03205**;
- frozen tolerance: `0.01`, therefore prediction = positive;
- held-out mean accuracy delta: **+0.1703 pp**;
- decision: **PASS**;
- descriptive rank p-value: `0.1464`;
- descriptive held-out mean p-value: `0.0331`.

This supports the **condition-average direction** of the frozen rule. It does not establish a calibrated per-run gate; using the same tolerance descriptively gave 6 matches, 3 mismatches, and 1 uncertain replicate.

See [`docs/CIFAR_RESNET_PRIMARY.md`](docs/CIFAR_RESNET_PRIMARY.md) and [`docs/CIFAR_RESNET_REP_RANK.md`](docs/CIFAR_RESNET_REP_RANK.md).

## Mechanism: finite-budget coverage and representation conversion

Several simpler explanations are ruled out or weakened:

- gradient novelty is **not** simply a better estimator of the mean expected gradient than random sampling;
- loss-hard can produce a larger immediate one-step loss decrease;
- larger accumulated selected-gradient effective rank is **not sufficient**: Breast-high increases it strongly while held-out mean does not improve;
- pure diversity without a hard anchor can hurt;
- full-network gradient signatures were slower without benefit over the tested final-layer proxy.

The current working mechanism is therefore not "maximize gradient diversity." It is:

```text
hard + non-redundant gradient coverage
    -> trajectory change
    -> conversion into reusable representation structure
    -> held-out mean benefit.
```

Across the discovery tasks, hidden representation effective-rank change tracked benefit better than raw gradient-rank expansion. This produced the frozen directional rule:

> **sign(delta hidden-representation effective rank) predicts sign(mean held-out benefit).**

The registered condition-level record is now 8/8 across six datasets. Three tests still share Digits, so these are not eight independent datasets. The evidence supports a condition-average predictor, not a universal magnitude map or causal theorem.

See [`docs/THEORETICAL_FRAMEWORK.md`](docs/THEORETICAL_FRAMEWORK.md), [`docs/TRAJECTORY_MECHANISM.md`](docs/TRAJECTORY_MECHANISM.md), and [`docs/PROSPECTIVE_REPRESENTATION_RANK.md`](docs/PROSPECTIVE_REPRESENTATION_RANK.md).

## FashionMNIST / Tiny Transformer replication

The initial 10-pair prospective test gave:

- mean delta representation rank **+0.05415**;
- mean held-out accuracy benefit **+0.7363 pp**;
- frozen directional decision **PASS**.

A preregistered independent extension added 20 new pairs (reps 10-29) under the unchanged protocol:

- delta representation rank **+0.07853**, descriptive `p=0.003414`;
- held-out mean benefit **+0.4377 pp**, descriptive `p=0.03420`;
- frozen directional decision **REPLICATES**.

After that decision, all 30 pairs were combined for precision only: rank delta `+0.07041`, mean benefit `+0.5372 pp`.

Per-run signs remain imperfect, so this strengthens a condition-average claim only. See [`docs/FASHION_TRANSFORMER_EXTENSION.md`](docs/FASHION_TRANSFORMER_EXTENSION.md).

## Execution reproducibility boundary

The CIFAR hosted-CPU audit reran reps 45 and 46 twice with OMP/MKL/OpenBLAS/PyTorch threads forced to one. The preregistered decision was **DRIFT PERSISTS**:

- max representation-rank drift: `0.427255`;
- max accuracy-metric drift: `0.014667`;
- aggregate rank and held-out mean directions remained positive in both repeats.

Rep45 was bitwise-identical across AMD EPYC 7763 and EPYC 9V74 runners. Rep46 drifted between AMD EPYC 7763 and Intel Xeon 6973P-C under the same software versions and thread controls. This makes hardware/microarchitecture-dependent numerical paths a strong candidate, but does not isolate a sole cause.

**Do not mix scientific rows from separate hosted-runner executions.** Bitwise cross-hardware reproducibility is not established.

See [`docs/CIFAR_CPU_REPRO_AUDIT.md`](docs/CIFAR_CPU_REPRO_AUDIT.md).

## Negative results are part of the repository

The evidence does **not** support these stronger claims:

- seed integers form semantic clusters;
- a universally good seed exists;
- more candidates/seeds are always better;
- gradient diversity alone guarantees generalization gains;
- worst-only training is robustly best;
- learned RNG fingerprints transfer universally;
- a fixed absolute gradient-cosine target transfers universally;
- the representation-rank rule is a universal/calibrated or per-run gate;
- the attempted per-environment representation-rank-SD tail rule works;
- CIFAR p10/worst robustness is confirmed;
- one small Transformer establishes general Transformer or large-model validity;
- representation effective rank is already proven causal;
- hosted-CPU runs are bitwise reproducible across heterogeneous hardware;
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
│   ├── THEORETICAL_FRAMEWORK.md
│   ├── TRAJECTORY_MECHANISM.md
│   ├── PROSPECTIVE_REPRESENTATION_RANK.md
│   ├── CIFAR_CPU_REPRO_AUDIT.md
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

1. Test the proposed mechanism causally: mediation, representation intervention, and matched structured-vs-unstructured novelty experiments.
2. Prospectively falsify the frozen condition-average rank rule on a materially larger/different architecture or modality.
3. Derive an actionable **early-trajectory** diagnostic; final representation rank currently requires training both methods.
4. Develop a new training-only tail-safety theory; the previous rep-rank-SD rule remains retired.
5. Run exact reproducibility tests on pinned hardware and GPU-vectorized wall-clock comparisons at matched budgets.
6. Extend beyond handcrafted stochastic corruption generators to naturally stochastic simulators/processes.

## Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). Licensed under the **Apache License 2.0**.
