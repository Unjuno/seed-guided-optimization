# Theoretical framework: finite-budget coverage of unresolved gradient directions

## Status

This is a **working theory**, not a theorem.

The evidence now separates SGO into a relatively well-supported upstream component and an unresolved downstream component:

```text
binding subset-update budget
    + hard, model-conditioned non-redundant candidate gradients
    -> different coverage of unresolved learning directions
    -> optimization trajectory / learned-function change
    -> performance change whose expression depends on architecture/task/regime
```

The upstream finite-budget dependence has passed **three preregistered n=30 Q-scaling experiments within the Digits/geometric family**: two MLP blocks and one SmallCNN block. In all three, the frozen low-Q versus high-Q benefit contrast is positive and significant, and the two methods become exactly identical when Q=K.

The downstream mediator remains unidentified. Raw hidden effective rank is coordinate-dependent and non-causal; channel-standardized effective rank failed a fresh quantitative mediator test. A later full-vs-clean dose interaction replicated within MLP but failed in SmallCNN, showing that the way SGO benefit appears in learned function space is not architecture-general.

The shortest current claim is:

> **SGO is a finite-budget stochastic-environment subset-selection method. When only a subset of candidate environments can contribute to an update, retaining hard examples while reducing model-conditioned gradient redundancy can improve performance in tested structured regimes. This budget dependence is robust to the tested MLP/SmallCNN architecture change within Digits/geometric. The functionally meaningful downstream mediator remains open.**

## 1. Seed is an environment index

Let `s` index a stochastic environment `e_s`. The relevant chain is

```text
s -> e_s -> g(theta, B, e_s) -> optimization trajectory -> learned function -> held-out behavior
```

with

```text
g(theta, B, e) = grad_theta L(theta; B, e).
```

SGO does not optimize integer seed values. It allocates finite update budget across model-conditioned stochastic environments.

## 2. Finite-budget selection problem

Suppose each step exposes `K` candidate environments but only `Q < K` can contribute to the backward update. Loss-hard selection spends this budget on the highest-current-loss candidates. Gradient-novel selection keeps a hard anchor while preferring additional hard candidates whose gradient signatures are less redundant with those already selected.

An idealized quantity is residual energy outside the selected gradient span:

```text
novelty(e | S) = ||g_e - P_S g_e||^2 / (||g_e||^2 + eps).
```

The implementation uses cheaper final-layer cosine/signature proxies. Its role is approximately

```text
error relevance × directional non-redundancy.
```

Hardness reduces allocation to arbitrary unusual directions; novelty reduces repeated spending on nearly identical corrections.

## 3. Direct Q-scaling evidence

All three confirmatory tests used `K=16`, Q=`{2,4,8,12,16}`, paired methods and fixed low-vs-high contrast

```text
A_B = mean(B_Q2, B_Q4) - mean(B_Q12, B_Q16),
```

where `B_Q` is gradnov minus loss-hard held-out mean performance.

### MLP audit 1

Per-Q benefit: `+2.151, +2.590, +2.076, +0.673, 0.000` pp.

- attenuation: **+2.034 pp**;
- one-sided p: **1.06e-7**;
- Q=16: exact scientific identity in 30/30 pairs.

### MLP audit 2, fresh block

Per-Q benefit: `+2.613, +2.443, +2.156, +0.885, 0.000` pp.

- attenuation: **+2.086 pp**;
- one-sided p: **5.16e-7**;
- Q=16: exact scientific identity in 30/30 pairs.

### SmallCNN audit, fresh block

Per-Q benefit: `+2.034, +2.073, +1.195, +1.578, 0.000` pp.

- attenuation: **+1.265 pp**;
- paired SE: `0.392 pp`;
- 95% t29 interval: **+0.463 to +2.067 pp**;
- one-sided p: **0.001548**;
- Q=16: zero mismatches in state digests, parameter tensors, training diagnostics and evaluation metrics.

The SmallCNN curve is not strictly monotone because Q=12 exceeds Q=8. Therefore the evidence supports the preregistered low-vs-high contrast and exact Q=K disappearance, **not a universal monotone Q law**.

Selected pairwise-gradient novelty in the SmallCNN audit decreased from `+0.3095` at Q=2 to `+0.02325` at Q=12 and became exactly zero at Q=16.

### Interpretation

The strongest direct mechanism statement is:

> **The selector advantage requires subset-selection freedom. When every K candidate contributes, selector choice has no remaining degree of freedom and the learned model itself becomes identical. Across the tested MLP and SmallCNN parameterizations, relaxing the subset constraint reduces the frozen low-vs-high advantage.**

This is architecture robustness **within one dataset/generator family**, not cross-dataset universality.

## 4. Why immediate descent and mean-gradient estimation are insufficient

Loss-hard can produce a larger immediate one-step loss decrease, and gradient novelty did not outperform random sampling as a mean-gradient estimator in the mechanism audit.

If several hard environments induce nearly the same correction,

```text
g_1 ~= g_2 ~= g_3,
```

then allocating multiple backward contributions to them can be locally reasonable while covering little new unresolved directional structure. SGO can trade some local greediness for broader finite-budget coverage.

## 5. Diversity alone is not the law

The stronger rule

```text
more gradient diversity/rank -> more benefit
```

is false. Breast-high is a direct counterexample, and pure diversity without a hard anchor is weak. Useful novelty must remain task-relevant and must alter the learned function productively.

Thus the theory is not “maximize diversity.” It is **budgeted allocation of hard, non-redundant corrections**.

## 6. Representation rank: marker, not mediator

Raw hidden representation effective rank has a strong prospective fixed-parameterization condition-average directional record. That empirical record remains useful as a marker.

But direct mechanism tests narrow its meaning:

1. raw-rank attenuation failed the first frozen Q-scaling mediator threshold (`p=0.1095`);
2. a positive diagonal function-preserving reparameterization moved raw rank by roughly +2 to -6 while logits, predictions and metrics were unchanged;
3. therefore raw rank is coordinate-dependent and cannot itself be an intrinsic causal state variable;
4. channel-standardized rank removed that trivial scaling freedom but failed the fresh budget-coupling test (`p=0.5063`).

The next mediator search should not be another post-hoc rank normalization.

## 7. Downstream conversion is architecture dependent

A fixed-dose MLP experiment found a strong full-strength geometric benefit. Its initial full-minus-clean interaction was borderline, then replicated on a disjoint 364-image reserve subset:

- MLP reserve full benefit: `+2.274 pp`;
- full-minus-clean interaction: `+2.906 pp`, p=`0.001726`.

However, a preregistered SmallCNN replication found:

- full benefit: `+2.417 pp`, p=`2.34e-5`;
- clean benefit: `+3.436 pp`;
- full-minus-clean interaction: `-1.019 pp`, p=`0.792`.

Therefore the upstream finite-budget effect can survive an architecture change while the **functional expression of that trajectory change does not**.

This rejects a universal rule such as

```text
stronger shift -> larger SGO benefit.
```

It also weakens any simple claim that reusable-factor conversion has already been identified.

## 8. Current causal picture

The evidence currently supports:

```text
binding subset budget
    -> hard/non-redundant gradient allocation
    -> changed coverage of unresolved directions
    -> changed optimization trajectory / learned function
    -> architecture/task-dependent performance expression
```

The open object is the map from trajectory coverage to a **functionally meaningful learned change**.

A viable mediator should ideally be:

- invariant to trivial function-preserving reparameterizations;
- measurable from training-only information before held-out construction;
- coupled to the finite-budget effect;
- prospective rather than fitted to final held-out outcomes;
- stable across materially different tasks/architectures.

## 9. Predicted help/failure regimes

SGO should be most useful when:

- Q is materially smaller than K;
- candidate gradients contain non-redundant task-relevant corrections;
- the model can use those corrections under the available training budget;
- novelty does not displace necessary hard examples.

Weak/null effects are expected when:

- Q approaches K;
- candidate gradients are already redundant;
- novelty is dominated by nuisance/noise;
- selected directions conflict rather than complement one another;
- the target metric depends on a downstream functional regime where the trajectory difference is not beneficial.

At Q=K=16, exact method identity has now been observed in all **three** preregistered Q-scaling blocks.

## 10. Highest-value next falsification

A fourth Digits Q-scaling replication would add little. The next decisive test is **cross-task budget scaling** using the same frozen low-vs-high contrast and Q=K identity logic on a materially different task/architecture.

Priority candidates:

1. FashionMNIST Transformer, where prior SGO/rank experiments already define a stable training protocol;
2. CIFAR-10 / ResNet-20, where a small corrected-significant mean benefit already exists but compute cost is higher.

The protocol must be fixed before outcomes, including K/Q values, optimizer, environment generator, held-out construction and the exact attenuation statistic.

Separately, training-only function-space diagnostics should target the unresolved downstream mediator rather than continuing proxy-rank search.

## Working claim

> **Under a binding finite update budget, selecting stochastic environments that are both currently hard and non-redundant in model-conditioned gradient space can improve performance in tested structured regimes. The finite-budget dependence has now replicated in two MLP blocks and a SmallCNN block within Digits/geometric, with exact identity whenever Q=K. This supports an architecture-robust subset-allocation mechanism within that family. It does not establish cross-dataset universality, a monotone Q law, or the downstream function-space mediator.**
