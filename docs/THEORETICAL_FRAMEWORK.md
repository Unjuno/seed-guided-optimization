# Theoretical framework: finite-budget coverage of unresolved gradient directions

## Status

This is a **working theory**, not a theorem.

The evidence separates SGO into a relatively well-supported upstream component and an unresolved downstream component:

```text
binding subset-update budget
    + hard, model-conditioned non-redundant candidate gradients
    -> different coverage of unresolved learning directions
    -> optimization trajectory / learned-function change
    -> performance change whose expression depends on architecture/task/regime
```

The upstream finite-budget dependence has now passed four preregistered n=30 Q-scaling tests: two Digits/MLP blocks, one Digits/SmallCNN block, and one FashionMNIST/Tiny Transformer block. Exact Q=K model-state identity appears in all four. The three Digits tests are strong; the Fashion cross-task attenuation passes the preregistered one-sided rule but has a two-sided 95% interval that narrowly crosses zero, so it is **borderline cross-task support**, not universal confirmation.

The downstream mediator remains unidentified. Raw hidden effective rank is coordinate-dependent and non-causal; channel-standardized effective rank failed a fresh quantitative mediator test. A full-vs-clean dose interaction replicated within MLP but failed in SmallCNN, showing that the functional expression of the trajectory change is architecture dependent.

Current claim:

> **SGO is a finite-budget stochastic-environment subset-selection method. When only a subset of candidate environments can contribute to an update, retaining hard examples while reducing model-conditioned gradient redundancy can improve performance in tested structured regimes. The mechanism is strongly architecture-robust within Digits/geometric and receives preregistered but borderline cross-task support on FashionMNIST/Tiny Transformer. The functionally meaningful downstream mediator remains open.**

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

## 3. Direct Q-scaling evidence

### Digits/geometric, MLP #1

K=16; Q={2,4,8,12,16}. Per-Q heldout mean benefit: `+2.151,+2.590,+2.076,+0.673,0.000` pp.

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

This extends the finite-budget effect across an architecture change within Digits/geometric.

### FashionMNIST, Tiny Transformer

K=8; Q={2,4,6,8}. Per-Q heldout mean benefit: `+0.846,+0.026,+0.245,0.000` pp.

Frozen statistic:

```text
A_B = mean(B_Q2,B_Q4) - mean(B_Q6,B_Q8).
```

- low-Q mean benefit: **+0.436 pp**;
- high-Q mean benefit: **+0.123 pp**;
- attenuation: **+0.314 pp**;
- paired SE: **0.157 pp**;
- two-sided 95% t29 interval: **-0.0065 to +0.6340 pp**;
- preregistered one-sided p: **0.027264**;
- Q=8: exact state-tensor, diagnostic and evaluation identity in 30/30 pairs.

This is the first direct Q-scaling support outside Digits. It is weaker than the Digits evidence because the two-sided interval narrowly includes zero and the pattern is irregular: Q6 exceeds Q4.

### Common manipulation pattern

Selected pairwise-gradient novelty differences shrink toward Q=K:

- SmallCNN/Digits: +0.3095, +0.1475, +0.0546, +0.02325, 0;
- Fashion/Tiny Transformer: +0.1618, +0.0476, +0.0243, 0.

The Q response is not universally monotonic in performance. The robust common structure is instead:

1. selector freedom exists only while Q<K;
2. non-redundancy manipulation contracts as coverage increases;
3. when Q=K, both methods become the same update rule and the learned state becomes exactly identical;
4. the preregistered low-vs-high benefit contrast is positive in all four completed blocks, with differing strength.

## 4. Why immediate descent and mean-gradient estimation are insufficient

Loss-hard can produce larger immediate one-step loss decrease, and gradient novelty did not outperform random sampling as a mean-gradient estimator in the mechanism audit. If several hard environments induce nearly the same correction, spending multiple update slots on them can be locally sensible while covering little new unresolved structure.

Thus SGO can trade local greediness for broader finite-budget coverage. The theory is not “maximize gradient diversity.” Pure diversity without hardness is weak, and larger accumulated gradient effective rank is not sufficient.

## 5. Representation rank: marker, not mediator

Raw hidden representation effective rank has a strong fixed-parameterization condition-average predictive record, but direct interventions narrow its interpretation:

1. raw-rank attenuation failed the first frozen budget mediator threshold;
2. function-preserving positive diagonal reparameterization moved raw rank strongly while logits/predictions/metrics remained identical;
3. raw rank is therefore coordinate-dependent and not an intrinsic causal state variable;
4. channel-standardized rank removed that scaling freedom but failed the fresh budget-coupling test (`p=0.5063`).

The mediator search should not continue by inventing post-hoc rank normalizations.

## 6. Downstream conversion is architecture dependent

MLP reserve-image testing found full geometric benefit +2.274 pp and full-minus-clean interaction +2.906 pp (`p=0.001726`). SmallCNN then found full benefit +2.417 pp but clean benefit +3.436 pp and full-minus-clean interaction -1.019 pp (`p=0.792`).

Therefore the upstream finite-budget effect can survive architecture/task changes while the **functional expression** of that trajectory difference changes. A universal `stronger shift -> larger SGO benefit` law is rejected.

## 7. Current causal picture

```text
binding subset budget
    -> hard/non-redundant gradient allocation
    -> changed coverage of unresolved directions
    -> changed optimization trajectory / learned function
    -> architecture/task-dependent performance expression
```

The open object is the map from coverage to a functionally meaningful learned change. A viable mediator should be invariant to trivial function-preserving reparameterizations, preferably training-only and prospective, coupled to the budget effect, and stable across materially different tasks.

## 8. Predicted help/failure regimes

SGO should be most useful when Q is materially smaller than K, candidate gradients contain task-relevant non-redundant corrections, the model can exploit those corrections under the available training budget, and novelty does not displace necessary hard examples.

Weak/null effects are expected as Q approaches K, when gradients are already redundant, when novelty is dominated by nuisance/noise, or when the downstream function-space change is not useful for the target metric.

At Q=K, exact method identity has now been observed in **all four** preregistered Q-scaling blocks, including FashionMNIST/Tiny Transformer.

## 9. Highest-value next falsification

The next decisive budget test is **CIFAR-10 / ResNet-20 Q-scaling** under the existing primary CIFAR protocol. It would test the same finite-budget logic on a larger convolutional task where a small corrected-significant Q=4 mean benefit already exists.

The CIFAR experiment should preregister K/Q values, low-vs-high contrast, optimizer, candidate/heldout generators, exact Q=K identity fields, replicate count and hardware/runtime constraints before outcomes.

Separately, the downstream mediator should be attacked with prospective training-only function-space diagnostics rather than additional rank proxies.

## Working claim

> **Under a binding finite update budget, selecting stochastic environments that are both currently hard and non-redundant in model-conditioned gradient space can improve performance in tested structured regimes. The finite-budget dependence is strongly replicated across MLP and SmallCNN within Digits/geometric and receives preregistered but borderline cross-task support on FashionMNIST/Tiny Transformer. Exact Q=K identity across all four blocks supports selector freedom as a necessary component of the observed advantage. This does not establish universal task validity, a monotone Q law, or the downstream function-space mediator.**
