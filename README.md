# Seed-Guided Optimization

**Gradient-aware selection of stochastic training environments under a finite update budget.**

Random seeds index stochastic environments, which induce different model-conditioned gradients. Seed-Guided Optimization (SGO) studies whether a fixed subset-update budget can be allocated more effectively by retaining hard environments while reducing redundancy among those gradient directions. Seed integers are not assumed to have intrinsic semantic classes or universal quality.

> **Status — 2026-09-06:** experimental research code. The strongest mechanism evidence is now finite-budget subset allocation. Three preregistered n=30 Q-scaling experiments within the Digits/geometric family—two MLP blocks and one SmallCNN block—show positive low-Q versus high-Q benefit attenuation and exact method identity when Q=K. The downstream function-space mediator remains unidentified. A full-strength geometric benefit replicates in MLP and SmallCNN, but the MLP full-minus-clean interaction does **not** replicate in SmallCNN.

## Latest completed experiments

| Experiment | Result | Interpretation |
|---|---|---|
| Dual evaluation, Issue #59, 30 pairs | shared +0.7817 pp, p=0.0998; specificity +0.0067 pp, p=0.4258 | **NO SHARED REPLICATION**; near-clean matched mixtures |
| MLP fixed-dose, Issue #61, 30 pairs | full +2.842 pp; full-minus-clean +1.621 pp, p=0.04897 | full effect strong; interaction initially borderline |
| MLP reserve-image replication, Issue #64, 30 pairs | full +2.274 pp, p=1.75e-8; full-minus-clean +2.906 pp, p=0.00173 | interaction replicates on disjoint reserve images, but intermediate strengths are non-monotone |
| SmallCNN regime audit, Issue #67, 30 pairs | full +2.417 pp, p=2.34e-5; clean +3.436 pp; full-minus-clean -1.019 pp, p=0.792 | full effect replicates; full-specific interaction does not |
| SmallCNN Q-scaling, Issue #70, 30 pairs | low-Q minus high-Q attenuation **+1.265 pp**, 95% CI [+0.463,+2.067], p=0.00155; Q=16 exact identity | **CNN FINITE-BUDGET COVERAGE REPLICATES** |

The SmallCNN Q-scaling result independently verified 300 checkpoint hashes/state digests and 24,000 environment rows. See [CNN budget result](docs/CNN_BUDGET_SCALING_RESULT.md), [research status](docs/RESEARCH_STATUS.md), and the individual experiment documents.

## Research claim

For parameters `theta`, minibatch/data state `B`, and stochastic environment `e_s` indexed by seed `s`:

```text
s -> e_s -> g(theta, B, e_s) -> optimization trajectory -> learned function -> held-out behavior
```

SGO is **model-conditioned stochastic-environment selection / trajectory shaping**, not seed-number optimization. The practical selector keeps a hard anchor and adds candidates whose head-gradient signatures are less redundant with those already selected.

The current supported mechanism statement is deliberately narrow:

> Under a binding subset-update budget, hard + gradient-nonredundant environment selection can outperform hardness-only allocation in tested structured regimes. Within the Digits/geometric family, this finite-budget dependence survives both MLP and SmallCNN parameterizations and becomes exactly zero when all K candidates contribute. The downstream learned-function mediator remains open.

Do **not** read this as a universal gradient-diversity rule. Pure diversity can be weak, larger accumulated gradient rank is not sufficient, and full-vs-clean dose interaction is architecture-dependent.

## Evidence at a glance

| Evidence | Result | Scope |
|---|---|---|
| Digits MLP geometric shifts | gradient-novel > parameter-novel by +1.45 pp held-out mean | structured small-scale regime |
| Original SmallCNN replication | +2.25 pp mean and +2.57 pp minimum vs loss-hard, both Holm-significant | tested CNN protocol |
| Optimizer replication | gains survive tuned AdamW and SGD+momentum | not explained by AdamW alone in tested MLP |
| CIFAR-10 / ResNet-20 primary, 40 pairs | mean +0.1206 pp, Holm(5) p=0.01336 | mean supported; tails unconfirmed |
| Finite-budget Q-scaling | MLP attenuation +2.034 pp and +2.086 pp; SmallCNN +1.265 pp; Q=K exact in all three | architecture-robust **within Digits/geometric**, not cross-dataset proof |
| Raw representation-rank prospective record | registered condition-average direction matches across multiple datasets/architectures | fixed-parameterization marker, not causal law |
| Function-preserving rank intervention | raw effective rank changes while predictions remain identical | raw rank is not functionally intrinsic |
| Standardized-rank budget test | mediator criterion failed while benefit attenuation replicated | normalization did not rescue the mediator hypothesis |
| Hosted-CPU reproducibility audit | one thread did not remove CIFAR cross-run drift | bitwise cross-hardware reproducibility not established |

## What the Q-scaling result means

For K=16, the latest SmallCNN held-out mean benefits were:

| Q | gradnov - loss-hard |
|---:|---:|
| 2 | +2.034 pp |
| 4 | +2.073 pp |
| 8 | +1.195 pp |
| 12 | +1.578 pp |
| 16 | 0 exactly |

The curve is not strictly monotonic—Q=12 exceeds Q=8. The supported test is the preregistered low-Q versus high-Q contrast, not a universal monotone law.

At Q=16, every candidate contributes. Loss-hard and gradnov then had identical model-state SHA256 digests, bitwise-identical parameter tensors, identical training diagnostics, and identical held-out/clean metrics in all 30 pairs. This is direct evidence that the method advantage requires selector freedom under a binding subset budget.

Recompute the primary SmallCNN Q-scaling statistic without retraining:

```bash
python experiments/check_cnn_budget_paired.py --input-dir results
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

The repeated structured-vs-nuisance matching attempts also did not identify reusable-factor causality. The downstream map from broader finite-budget coverage to clean/shifted performance appears architecture/optimization dependent.

## External validation: CIFAR-10 / ResNet-20

The primary 40-pair gradient-novel minus loss-hard mean difference was +0.1206 pp, raw p=0.002672 and Holm(5) p=0.01336. p10 and minimum were positive but not corrected-significant. See [CIFAR primary](docs/CIFAR_RESNET_PRIMARY.md) and [CPU reproducibility audit](docs/CIFAR_CPU_REPRO_AUDIT.md).

## Reproducibility

Accuracy-like CSV fields use fractions: `0.01` is one percentage point. The public CPU dependency baseline is pinned in `requirements.txt`; individual workflows additionally pin CPU/GPU-specific packages where required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/mlp_geometric.py --start 0 --end 20 --output mlp_geometric.csv
```

Use [experiments/README.md](experiments/README.md) and [docs/README.md](docs/README.md) for the evidence archive and exact workflows.

## Current research priority

The next high-value falsification is **cross-task Q-scaling**, not another Digits replication. The frozen low-vs-high Q contrast should be repeated on a materially different task/architecture with protocol choices fixed before outcomes. FashionMNIST/Transformer or CIFAR/ResNet are candidates.

Other priorities include training-only function-space diagnostics, a new tail-safety theory, pinned-hardware numerical studies, and GPU comparisons with genuinely matched computational budgets.

## Citation and license

Citation metadata: [CITATION.cff](CITATION.cff). License: [Apache-2.0](LICENSE).
