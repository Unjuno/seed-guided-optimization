# Seed-Guided Optimization

**Gradient-aware selection of stochastic training environments under a finite update budget.**

Random seeds index stochastic environments, which induce different model-conditioned gradients. Seed-Guided Optimization (SGO) studies whether a fixed subset-update budget can be allocated more effectively by retaining hard environments while reducing redundancy among those gradient directions. Seed integers are not assumed to have intrinsic semantic classes or universal quality.

> **Status — 2026-09-07:** experimental research code. Five preregistered n=30 Q-scaling blocks support a finite-budget subset-allocation effect across Digits, FashionMNIST and CIFAR-10, with different strengths. A new fixed-Q, shared-reference loss-stratified intervention also passes: MAXNOV schedules beat MINNOV by **+4.669 pp** held-out mean while reference hardness is TOST-equivalent. However MAXNOV simultaneously increases known physical transformation-parameter diversity, so the result is direct support for a nonredundancy-selected schedule intervention, not yet proof that gradient novelty is the unique causal variable. The downstream learned-function mediator remains unidentified.

## Latest completed experiments

| Experiment | Result | Interpretation |
|---|---|---|
| Loss-stratified nonredundancy intervention, Issue #80, 30 pairs | novelty manipulation +0.2316; hardness 90% CI [-0.0133,-0.0049] inside ±0.10 SD; heldout **+4.669 pp**, 95% CI [+3.187,+6.152], p=2.39e-7 | strongest direct intervention evidence so far; physical parameter diversity remains a competing covariate |
| SmallCNN Q-scaling, Issue #70, 30 pairs | attenuation **+1.265 pp**, 95% CI [+0.463,+2.067], p=0.00155; Q=16 exact identity | strong architecture-robust finite-budget evidence within Digits/geometric |
| FashionMNIST Tiny Transformer Q-scaling, Issue #73, 30 pairs | attenuation **+0.314 pp**, 95% CI [-0.0065,+0.6340], preregistered one-sided p=0.0273; Q=8 exact identity | cross-task support, but borderline |
| CIFAR-10 ResNet-20 Q-scaling, Issue #76, 30 pairs | attenuation **+0.1168 pp**, 95% CI [+0.0335,+0.2000], p=0.00380; Q=8 exact identity | strong larger-task cross-task replication |
| CIFAR Q=K endpoint sensitivity, post-hoc | low mean(Q2,Q4)-Q6 **+0.0968 pp**, 95% CI [+0.0032,+0.1904], descriptive p=0.0215 | attenuation remains positive after removing the structurally forced zero endpoint |

See [direct intervention result](docs/LOSS_STRATIFIED_NONREDUNDANCY_RESULT.md), [CIFAR budget result](docs/CIFAR_BUDGET_SCALING_RESULT.md), [Fashion budget result](docs/FASHION_BUDGET_SCALING_RESULT.md), [CNN budget result](docs/CNN_BUDGET_SCALING_RESULT.md), and [research status](docs/RESEARCH_STATUS.md).

## Research claim

For parameters `theta`, minibatch/data state `B`, and stochastic environment `e_s` indexed by seed `s`:

```text
s -> e_s -> g(theta, B, e_s) -> optimization trajectory -> learned function -> held-out behavior
```

SGO is **model-conditioned stochastic-environment selection / trajectory shaping**, not seed-number optimization. The practical selector keeps a hard anchor and adds candidates whose head-gradient signatures are less redundant with those already selected.

The current supported mechanism statement is:

> Under a binding subset-update budget, hard + model-conditioned nonredundant environment selection can outperform hardness-only allocation in several tested structured regimes. Finite-budget dependence is replicated across MLP and SmallCNN within Digits/geometric and has cross-task FashionMNIST and CIFAR evidence. A fixed-Q shared-reference intervention further shows that, within tightly matched loss-rank strata, schedules chosen for high reference gradient novelty strongly outperform schedules chosen for low novelty. Because high-novelty schedules also have greater known transformation-parameter diversity, gradient novelty is not yet isolated as the unique causal property. The downstream learned-function mediator remains open.

Q=K identity is an implementation/control fact, not standalone causal-mediation proof. The CIFAR post-hoc sensitivity is useful because attenuation remains positive even after the exact-zero Q=K endpoint is removed.

Do **not** read this as a universal gradient-diversity law. Pure diversity can be weak, larger accumulated gradient rank is not sufficient, Q-response shape is task dependent, and the direct MAXNOV/MINNOV schedule contrast still moves an environment-space covariate.

## Evidence at a glance

| Evidence | Result | Scope |
|---|---|---|
| Loss-stratified direct intervention | fixed Q4; reference hardness equivalent; MAXNOV-MINNOV heldout +4.669 pp | strongest direct schedule intervention; parameter-diversity confound remains |
| Digits MLP geometric shifts | gradient-novel > parameter-novel by +1.45 pp held-out mean | structured small-scale regime |
| Original SmallCNN replication | +2.25 pp mean and +2.57 pp minimum vs loss-hard, both Holm-significant | tested CNN protocol |
| Optimizer replication | gains survive tuned AdamW and SGD+momentum | not explained by AdamW alone in tested MLP |
| CIFAR-10 / ResNet-20 primary, 40 pairs | mean +0.1206 pp, Holm(5) p=0.01336 | mean supported; tails unconfirmed |
| Digits finite-budget Q-scaling | MLP attenuation +2.034 and +2.086 pp; SmallCNN +1.265 pp; Q=K exact | strong within-family architecture robustness |
| FashionMNIST/Tiny Transformer Q-scaling | +0.314 pp attenuation; one-sided p=0.0273; Q=8 exact | cross-task support but two-sided CI crosses zero |
| CIFAR-10/ResNet-20 Q-scaling | +0.1168 pp attenuation; one-sided p=0.00380; Q=8 exact | strong larger-task cross-task support |
| Raw representation-rank prospective record | registered condition-average direction matches across multiple datasets/architectures | fixed-parameterization marker, not causal law |
| Function-preserving rank intervention | raw effective rank changes while predictions remain identical | raw rank is not functionally intrinsic |
| Standardized-rank budget test | mediator criterion failed while benefit attenuation replicated | normalization did not rescue the mediator hypothesis |
| Hosted-CPU reproducibility audit | one thread did not remove CIFAR cross-run drift | bitwise cross-hardware reproducibility not established |

## Direct intervention evidence

Issue #80 fixes K=16 and Q=4 in both arms. On one common reference trajectory, both arms always retain loss rank1 and select one environment from each adjacent pair `(2,3)`, `(4,5)`, `(6,7)`. MAXNOV and MINNOV are chosen from the same eight feasible subsets before either intervention arm is trained.

| Frozen quantity | Result |
|---|---:|
| reference gradient-novelty contrast | **+0.231634**, 30/30 positive |
| reference standardized hardness contrast | **-0.00910 SD** |
| hardness TOST 90% CI | **[-0.01328,-0.00492] SD**, inside ±0.10 |
| heldout mean MAXNOV-MINNOV | **+4.669 pp** |
| 95% CI | **[+3.187,+6.152] pp** |
| one-sided p | **2.39e-7** |
| positive heldout pairs | **26/30** |

Secondary clean effect is -1.755 pp. p10/minimum improvements are descriptive and not a tail-safety claim.

Post-hoc, the mean pairwise distance between standardized seven-dimensional physical transformation parameters is also higher in MAXNOV by **+0.4927** (30/30 positive). This is now the main competing explanation to attack.

## Q-scaling evidence

Digits/SmallCNN, K=16:

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

CIFAR-10/ResNet-20, K=8:

| Q | gradnov - loss-hard |
|---:|---:|
| 2 | +0.1960 pp |
| 4 | +0.0775 pp |
| 6 | +0.0399 pp |
| 8 | 0 exactly |

The CIFAR curve is monotone in this observed block; the SmallCNN and Fashion curves are not. Universal monotonicity is not claimed. Exact model-state identity was verified at Q=K in every paired run of all five preregistered Q-scaling blocks.

Recompute published paired statistics without retraining:

```bash
python experiments/check_cnn_budget_paired.py --input-dir results
python experiments/check_fashion_budget_paired.py --input-dir results
python experiments/check_cifar_budget_paired.py --input-dir results
python experiments/check_loss_stratified_nonredundancy.py --input-dir results
```

## Mechanism boundaries

The evidence does **not** support:

- universally good seed families;
- a universal selector/controller setting;
- “more gradient diversity is always better”;
- a universal monotone Q curve;
- treating Q=K identity alone as causal proof;
- treating the current MAXNOV/MINNOV intervention as gradient-specific after ignoring its physical-parameter diversity shift;
- a universal `stronger shift -> larger SGO benefit` law;
- raw or standardized representation effective rank as a validated causal mediator;
- calibrated per-run gating;
- confirmed CIFAR p10/worst-case robustness;
- general large-Transformer validity;
- bitwise cross-hardware hosted-CPU reproducibility;
- GPU efficiency claims from CPU experiments.

The downstream map from finite-budget gradient allocation to learned function and clean/shifted performance remains architecture/optimization dependent. The structured-vs-nuisance matching program has not established reusable-factor causality.

## External validation: CIFAR-10 / ResNet-20

The original primary 40-pair Q=4 gradient-novel minus loss-hard mean difference was +0.1206 pp, raw p=0.002672 and Holm(5) p=0.01336. The independent Q-scaling block uses fresh replicate/environment seeds and shows low-vs-high attenuation +0.1168 pp, p=0.003804, with Q=8 exact identity. p10/minimum results remain secondary and do not establish tail safety.

## Reproducibility

Accuracy-like CSV fields use fractions: `0.01` is one percentage point. The public CPU dependency baseline is pinned in `requirements.txt`; individual workflows additionally pin CPU-specific packages where required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use [experiments/README.md](experiments/README.md) and [docs/README.md](docs/README.md) for exact workflows and evidence files.

## Current research priority

The next decisive test is **parameter-diversity-matched gradient-novelty reversal**. Preserve the shared-reference, fixed-Q, loss-stratified design, but choose MAXNOV/MINNOV subsets only among pairs that are also matched in standardized physical transformation-parameter diversity. This directly attacks the main post-hoc alternative explanation uncovered by Issue #80.

In parallel, the downstream mediator should be attacked with prospective training-only function-space diagnostics rather than additional representation-rank normalizations. Other priorities are a new tail-safety theory, pinned-hardware numerical studies, and GPU comparisons with genuinely matched costs.

## Citation and license

Citation metadata: [CITATION.cff](CITATION.cff). License: [Apache-2.0](LICENSE).
