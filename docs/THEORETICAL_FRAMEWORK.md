# Theoretical framework: finite-budget allocation of unresolved gradient directions

## Status

This is a **working theory**, not a theorem.

The evidence separates SGO into a relatively well-supported upstream finite-budget phenomenon and an unresolved causal/downstream component:

```text
binding subset-update budget
    + hard, model-conditioned non-redundant candidate gradients
    -> different allocation across unresolved learning directions
    -> optimization trajectory / learned-function change
    -> performance change whose expression depends on architecture/task/regime
```

Five preregistered n=30 Q-scaling blocks are complete: two Digits/MLP blocks, one Digits/SmallCNN block, FashionMNIST/Tiny Transformer, and CIFAR-10/ResNet-20. Exact Q=K model-state identity appears in all five. The three Digits tests are strong; Fashion passes the preregistered one-sided rule but its two-sided 95% interval narrowly crosses zero; CIFAR gives a strong cross-task replication whose attenuation remains positive in a post-hoc comparison that excludes the exact-zero Q=K endpoint.

The downstream mediator remains unidentified. Raw hidden effective rank is coordinate-dependent and non-causal; channel-standardized effective rank failed a fresh quantitative mediator test. A full-vs-clean dose interaction replicated within MLP but failed in SmallCNN, showing that the functional expression of trajectory change is architecture dependent.

Current claim:

> **SGO is a finite-budget stochastic-environment subset-selection method. When only a subset of candidate environments can contribute to an update, retaining hard examples while reducing model-conditioned gradient redundancy can improve performance in several tested structured regimes. Finite-budget dependence is strongly architecture-robust within Digits/geometric and now has strong CIFAR-10/ResNet-20 cross-task support, with weaker FashionMNIST/Tiny Transformer support. The causal contribution of non-redundancy itself and the functionally meaningful downstream mediator remain open.**

## 1. Seed is an environment index

Let `s` index a stochastic environment `e_s`:

```text
s -> e_s -> g(theta, B, e_s) -> optimization trajectory -> learned function -> held-out behavior
```

with `g(theta,B,e)=grad_theta L(theta;B,e)`. SGO does not optimize integer seed values; it allocates finite update budget across model-conditioned stochastic environments.

## 2. Finite-budget selection problem

Suppose each step exposes K candidate environments but only Q<K can contribute. Loss-hard spends this budget on highest-current-loss candidates. Gradient-novel keeps a hard anchor while preferring additional hard candidates whose gradient signatures are less redundant with already selected directions.

An idealized quantity is residual energy outside the selected gradient span:

```text
novelty(e | S) = ||g_e - P_S g_e||^2 / (||g_e||^2 + eps).
```

The implementation uses cheaper final-layer cosine/signature proxies. Their intended role is approximately `error relevance × directional non-redundancy`.

This equation motivates the selector but is **not yet an identified causal mediator**. Q-scaling changes both available selector freedom and the feasible subset geometry, so it cannot alone prove that non-redundancy is the causal ingredient.

## 3. Direct Q-scaling evidence

### Digits/geometric, MLP #1

K=16; Q={2,4,8,12,16}. Per-Q held-out mean benefit: `+2.151,+2.590,+2.076,+0.673,0.000` pp.

- frozen low-minus-high attenuation: **+2.034 pp**;
- one-sided p: **1.06e-7**;
- Q=16: exact identity in 30/30 pairs.

### Digits/geometric, MLP #2 fresh

Per-Q benefit: `+2.613,+2.443,+2.156,+0.885,0.000` pp.

- attenuation: **+2.086 pp**;
- one-sided p: **5.16e-7**;
- Q=16: exact identity in 30/30 pairs.

### Digits/geometric, SmallCNN

Per-Q benefit: `+2.034,+2.073,+1.195,+1.578,0.000` pp.

- attenuation: **+1.265 pp**;
- 95% t29 interval: **+0.463 to +2.067 pp**;
- one-sided p: **0.001548**;
- Q=16: exact state-tensor, diagnostic and evaluation identity in 30/30 pairs.

A post-hoc endpoint sensitivity removed Q16 and compared low mean(Q2,Q4) with Q12 alone. The result was +0.476 pp, 95% CI [-0.537,+1.490], descriptive p=0.172. Thus the frozen PASS is valid, but this block alone does not establish attenuation among non-full Q values.

### FashionMNIST, Tiny Transformer

K=8; Q={2,4,6,8}. Per-Q held-out mean benefit: `+0.846,+0.026,+0.245,0.000` pp.

- low-Q mean benefit: **+0.436 pp**;
- high-Q mean benefit: **+0.123 pp**;
- attenuation: **+0.314 pp**;
- paired SE: **0.157 pp**;
- two-sided 95% t29 interval: **-0.0065 to +0.6340 pp**;
- preregistered one-sided p: **0.027264**;
- Q=8: exact identity in 30/30 pairs.

This is direct cross-task Q-scaling support, but weaker than the Digits evidence because the two-sided interval narrowly includes zero and Q6 exceeds Q4.

### CIFAR-10, ResNet-20

K=8; Q={2,4,6,8}. Fresh paired reps50-79 reused the established CIFAR primary architecture/data/optimizer protocol and fresh environment seeds.

Per-Q held-out mean benefit: `+0.1960,+0.0775,+0.0399,0.0000` pp.

Frozen statistic:

```text
A_B = mean(B_Q2,B_Q4) - mean(B_Q6,B_Q8).
```

- low-Q mean benefit: **+0.13675 pp**;
- high-Q mean benefit: **+0.01997 pp**;
- attenuation: **+0.11679 pp**;
- paired SE: **0.04071 pp**;
- two-sided 95% t29 interval: **+0.03353 to +0.20005 pp**;
- one-sided p: **0.003804**;
- Q=8: exact state/diagnostic/evaluation identity in 30/30 pairs.

The CIFAR curve is monotone in this observed block, but monotonicity was not the primary rule and should not be universalized.

Crucially, a post-hoc endpoint sensitivity removes Q8 and compares the same low-Q mean with Q6 alone:

- low mean(Q2,Q4)-Q6: **+0.09682 pp**;
- 95% CI: **+0.00324 to +0.19040 pp**;
- descriptive one-sided p: **0.02152**.

This is not a new preregistered result, but it shows that the CIFAR attenuation is not solely an arithmetic consequence of averaging an exact-zero Q=K endpoint into the high-Q group.

### Common manipulation pattern

Selected pairwise-gradient novelty differences shrink toward Q=K:

- SmallCNN/Digits: +0.3095, +0.1475, +0.0546, +0.02325, 0;
- Fashion/Tiny Transformer: +0.1618, +0.0476, +0.0243, 0;
- CIFAR/ResNet-20: +0.1004, +0.0324, +0.0125, 0.

The performance Q response is not universally monotonic. The robust common structure is:

1. selector freedom exists only while Q<K;
2. the measured non-redundancy manipulation contracts as coverage increases;
3. when Q=K, both methods become the same update rule and the learned state becomes exactly identical;
4. the preregistered low-vs-high benefit contrast is positive in all five completed blocks, with different statistical strength;
5. CIFAR provides the first completed block in which an exploratory non-Q=K attenuation contrast also has a two-sided interval above zero.

## 4. What Q=K identity does and does not prove

At Q=K, the two selectors choose the same full candidate set in the same order. Exact state identity is therefore a strong pipeline/control check and demonstrates that the observed method difference requires selector freedom.

However, this control alone does not identify *why* selector freedom helps. It cannot distinguish gradient non-redundancy from every other property that changes when different subsets are selected. For that reason, direct redundancy reversal at fixed Q and matched hardness is now a higher-value falsification than additional Q-scaling repetition.

## 5. Why immediate descent and mean-gradient estimation are insufficient

Loss-hard can produce larger immediate one-step loss decrease, and gradient novelty did not outperform random sampling as a mean-gradient estimator in the mechanism audit. If several hard environments induce nearly the same correction, spending multiple update slots on them can be locally sensible while covering little new unresolved structure.

Thus SGO can trade local greediness for broader finite-budget allocation. The theory is not “maximize gradient diversity.” Pure diversity without hardness is weak, and larger accumulated gradient effective rank is not sufficient.

## 6. Representation rank: marker, not mediator

Raw hidden representation effective rank has a strong fixed-parameterization condition-average predictive record, but direct interventions narrow its interpretation:

1. raw-rank attenuation failed the first frozen budget mediator threshold;
2. function-preserving positive diagonal reparameterization moved raw rank strongly while logits/predictions/metrics remained identical;
3. raw rank is therefore coordinate-dependent and not an intrinsic causal state variable;
4. channel-standardized rank removed that scaling freedom but failed the fresh budget-coupling test (`p=0.5063`).

The mediator search should not continue by inventing post-hoc rank normalizations.

## 7. Downstream conversion is architecture dependent

MLP reserve-image testing found full geometric benefit +2.274 pp and full-minus-clean interaction +2.906 pp (`p=0.001726`). SmallCNN then found full benefit +2.417 pp but clean benefit +3.436 pp and full-minus-clean interaction -1.019 pp (`p=0.792`).

Therefore finite-budget selection effects can survive architecture/task changes while their **functional expression** changes. A universal `stronger shift -> larger SGO benefit` law is rejected.

## 8. Current causal picture

```text
binding subset budget
    -> selector freedom among hard candidates
    -> model-conditioned gradient allocation differs
    -> optimization trajectory / learned function differs
    -> architecture/task-dependent performance expression
```

The evidence strongly supports the first two structural statements across the tested Q-scaling blocks. The more specific arrow

```text
higher useful gradient non-redundancy -> better learned function
```

is still a hypothesis requiring direct intervention.

A viable downstream mediator should be invariant to trivial function-preserving reparameterizations, preferably training-only and prospective, coupled to the budget effect, and stable across materially different tasks.

## 9. Predicted help/failure regimes

SGO should be most useful when Q is materially smaller than K, candidate gradients contain task-relevant non-redundant corrections, the model can exploit those corrections under the available training budget, and novelty does not displace necessary hard examples.

Weak/null effects are expected as Q approaches K, when gradients are already redundant, when novelty is dominated by nuisance/noise, or when the downstream function-space change is not useful for the target metric.

At Q=K, exact method identity has now been observed in **all five** preregistered Q-scaling blocks.

## 10. Highest-value next falsification

The next decisive mechanism test is **loss-stratified gradient-redundancy reversal** at fixed Q. A clean implementation should:

1. expose the same K candidate environments to both methods;
2. always retain the hardest anchor;
3. force both selectors to choose from identical adjacent loss-rank strata, holding the hardness structure approximately fixed;
4. choose the feasible subset with maximum versus minimum pairwise gradient novelty;
5. preregister a scale-free hardness-equivalence margin before held-out outcomes;
6. require a successful novelty manipulation and hardness balance before interpreting any held-out difference.

A positive held-out effect under this design would be substantially more direct evidence for non-redundancy itself than another Q=K disappearance result. A null result with a valid manipulation would materially narrow the current mechanism.

Separately, the downstream mediator should be attacked with prospective training-only function-space diagnostics rather than additional rank proxies.

## Working claim

> **Under a binding finite update budget, selecting stochastic environments that are both currently hard and non-redundant in model-conditioned gradient space can improve performance in several tested structured regimes. Finite-budget dependence is strongly replicated across MLP and SmallCNN within Digits/geometric and now has strong CIFAR-10/ResNet-20 cross-task support, with weaker FashionMNIST/Tiny Transformer support. Exact Q=K identity across all five blocks establishes that selector freedom is necessary for method differences under the controlled update rule, while CIFAR endpoint sensitivity shows the attenuation need not vanish when the forced zero endpoint is removed. This still does not prove that gradient non-redundancy is itself the causal mediator or identify the downstream learned-function mechanism.**
