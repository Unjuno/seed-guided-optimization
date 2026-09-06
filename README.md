# Seed-Guided Optimization

**Gradient-aware selection of stochastic training environments under a finite update budget.**

Random seeds index stochastic environments, which induce different model-conditioned gradients. Seed-Guided Optimization (SGO) studies whether a fixed subset-update budget can be allocated more effectively by retaining hard environments while reducing redundancy among those gradient directions. Seed integers are not assumed to have intrinsic semantic classes or universal quality.

> **Status — 2026-09-06:** experimental research code. The strongest mechanism evidence is finite-budget subset allocation. Three preregistered n=30 Q-scaling experiments within Digits/geometric—two MLP blocks and one SmallCNN block—give strong low-Q versus high-Q attenuation and exact method identity when Q=K. A new preregistered n=30 FashionMNIST/Tiny Transformer test also passes the frozen directional contrast and reproduces exact Q=K identity, providing the first cross-task support, but its attenuation is statistically borderline because the two-sided 95% CI narrowly crosses zero. The downstream function-space mediator remains unidentified.

## Latest completed experiments

| Experiment | Result | Interpretation |
|---|---|---|
| MLP reserve-image replication, Issue #64, 30 pairs | full +2.274 pp, p=1.75e-8; full-minus-clean +2.906 pp, p=0.00173 | MLP interaction replicates on disjoint reserve images, but intermediate strengths are non-monotone |
| SmallCNN regime audit, Issue #67, 30 pairs | full +2.417 pp, p=2.34e-5; clean +3.436 pp; full-minus-clean -1.019 pp, p=0.792 | full effect replicates; full-specific interaction does not |
| SmallCNN Q-scaling, Issue #70, 30 pairs | attenuation **+1.265 pp**, 95% CI [+0.463,+2.067], p=0.00155; Q=16 exact identity | strong architecture-robust finite-budget evidence within Digits/geometric |
| FashionMNIST Tiny Transformer Q-scaling, Issue #73, 30 pairs | attenuation **+0.314 pp**, 95% CI [-0.0065,+0.6340], preregistered one-sided p=0.0273; Q=8 exact identity | **first cross-task directional support**, but borderline rather than strong confirmation |

See [Fashion budget result](docs/FASHION_BUDGET_SCALING_RESULT.md), [CNN budget result](docs/CNN_BUDGET_SCALING_RESULT.md), and [research status](docs/RESEARCH_STATUS.md).

## Research claim

For parameters `theta`, minibatch/data state `B`, and stochastic environment `e_s` indexed by seed `s`:

```text
s -> e_s -> g(theta, B, e_s) -> optimization trajectory -> learned function -> held-out behavior
```

SGO is **model-conditioned stochastic-environment selection / trajectory shaping**, not seed-number optimization. The practical selector keeps a hard anchor and adds candidates whose head-gradient signatures are less redundant with those already selected.

The current supported mechanism statement is:

> Under a binding subset-update budget, hard + gradient-nonredundant environment selection can outperform hardness-only allocation in tested structured regimes. The dependence is strongly replicated across MLP and SmallCNN within Digits/geometric and receives preregistered but borderline cross-task support on FashionMNIST/Tiny Transformer. In every Q-scaling test, the methods become exactly identical when all K candidates contribute. The downstream learned-function mediator remains open.

Do **not** read this as a universal gradient-diversity law. Pure diversity can be weak, larger accumulated gradient rank is not sufficient, and the Q-response shape is task dependent.

## Evidence at a glance

| Evidence | Result | Scope |
|---|---|---|
| Digits MLP geometric shifts | gradient-novel > parameter-novel by +1.45 pp held-out mean | structured small-scale regime |
| Original SmallCNN replication | +2.25 pp mean and +2.57 pp minimum vs loss-hard, both Holm-significant | tested CNN protocol |
| Optimizer replication | gains survive tuned AdamW and SGD+momentum | not explained by AdamW alone in tested MLP |
| CIFAR-10 / ResNet-20 primary, 40 pairs | mean +0.1206 pp, Holm(5) p=0.01336 | mean supported; tails unconfirmed |
| Digits finite-budget Q-scaling | MLP attenuation +2.034 and +2.086 pp; SmallCNN +1.265 pp; Q=K exact in all three | strong architecture robustness within one dataset/generator family |
| FashionMNIST/Tiny Transformer Q-scaling | +0.314 pp attenuation; one-sided p=0.0273; Q=8 exact | first cross-task support; two-sided CI crosses zero |
| Raw representation-rank prospective record | registered condition-average direction matches across multiple datasets/architectures | fixed-parameterization marker, not causal law |
| Function-preserving rank intervention | raw effective rank changes while predictions remain identical | raw rank is not functionally intrinsic |
| Standardized-rank budget test | mediator criterion failed while benefit attenuation replicated | normalization did not rescue the mediator hypothesis |
| Hosted-CPU reproducibility audit | one thread did not remove CIFAR cross-run drift | bitwise cross-hardware reproducibility not established |

## Q-scaling evidence

Latest SmallCNN/Digits held-out mean benefits, K=16:

| Q | gradnov - loss-hard |
|---:|---:|
| 2 | +2.034 pp |
| 4 | +2.073 pp |
| 8 | +1.195 pp |
| 12 | +1.578 pp |
| 16 | 0 exactly |

FashionMNIST/Tiny Transformer, K=8:

| Q | gradnov - loss-hard |
|---:|---:|
| 2 | +0.846 pp |
| 4 | +0.026 pp |
| 6 | +0.245 pp |
| 8 | 0 exactly |

Neither curve is strictly monotonic. The supported quantity is the preregistered low-Q versus high-Q contrast plus exact disappearance at Q=K, not a smooth universal dose law.

At Q=K, loss-hard and gradnov have the same candidate set and update order. Exact model-state identity was verified in every paired run of all four preregistered Q-scaling blocks.

Recompute the published paired statistics without retraining:

```bash
python experiments/check_cnn_budget_paired.py --input-dir results
python experiments/check_fashion_budget_paired.py --input-dir results
```

## Mechanism boundaries

The evidence does **not** support:

- universally good seed families;
- a universal selector/controller setting;
- “more gradient diversity is always better”;
- a universal monotone Q curve;
- a universal `stronger shift -> larger SGO benefit` law;
- raw or standardized representation effective rank as a validated causal mediator;
- calibrated per-run gating;
- confirmed CIFAR p10/worst-case robustness;
- general large-Transformer validity;
- bitwise cross-hardware hosted-CPU reproducibility;
- GPU efficiency claims from CPU experiments.

The downstream map from broader finite-budget coverage to clean/shifted performance remains architecture/optimization dependent. The structured-vs-nuisance matching program also has not established reusable-factor causality.

## External validation: CIFAR-10 / ResNet-20

The primary 40-pair gradient-novel minus loss-hard mean difference was +0.1206 pp, raw p=0.002672 and Holm(5) p=0.01336. p10 and minimum were positive but not corrected-significant. See [CIFAR primary](docs/CIFAR_RESNET_PRIMARY.md) and [CPU reproducibility audit](docs/CIFAR_CPU_REPRO_AUDIT.md).

## Reproducibility

Accuracy-like CSV fields use fractions: `0.01` is one percentage point. The public CPU dependency baseline is pinned in `requirements.txt`; individual workflows additionally pin CPU/GPU-specific packages where required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use [experiments/README.md](experiments/README.md) and [docs/README.md](docs/README.md) for exact workflows and evidence files.

## Current research priority

The next highest-value budget falsification is **CIFAR-10 / ResNet-20 Q-scaling**, because it would move the mechanism to a substantially larger convolutional task. That experiment is more expensive and should preserve the existing CIFAR protocol while fixing K/Q and the attenuation contrast before outcomes.

In parallel, the downstream mediator should be attacked with prospective training-only function-space diagnostics rather than additional representation-rank normalizations. Other priorities are a new tail-safety theory, pinned-hardware numerical studies, and GPU comparisons with genuinely matched costs.

## Citation and license

Citation metadata: [CITATION.cff](CITATION.cff). License: [Apache-2.0](LICENSE).
