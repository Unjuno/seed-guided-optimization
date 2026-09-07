# Theoretical framework: finite-budget allocation of unresolved gradient directions

## Status

This is a **working theory**, not a theorem.

The evidence now separates three levels:

```text
(1) binding finite subset budget
    -> selector freedom matters

(2) within fixed Q and tightly matched reference hardness,
    schedules selected for high model-conditioned gradient nonredundancy
    -> substantially different learned function / shifted heldout performance

(3) downstream internal mediator
    -> still unidentified
```

Five preregistered n=30 Q-scaling blocks support level (1): two Digits/MLP, one Digits/SmallCNN, FashionMNIST/Tiny Transformer, and CIFAR-10/ResNet-20. Exact Q=K model-state identity appears in all five; CIFAR additionally retains a positive exploratory attenuation after the exact-zero endpoint is removed.

Preregistered Issue #80 provides a more direct fixed-Q intervention for level (2). On one common SmallCNN reference trajectory, both arms use Q=4, retain identical loss-rank strata, and are defined before either arm trains. MAXNOV schedules create a large reference gradient-novelty manipulation while passing a frozen hardness-equivalence TOST, then outperform MINNOV by +4.669 pp held-out mean. However post-hoc analysis shows MAXNOV also increases diversity in the known physical transformation parameters. Thus the experiment supports a **nonredundancy-selected schedule intervention**, but gradient novelty has not yet been isolated as the unique causal property.

The downstream mediator remains unidentified. Raw hidden effective rank is coordinate-dependent and non-causal; channel-standardized effective rank failed a fresh quantitative mediator test. A full-vs-clean dose interaction replicated within MLP but failed in SmallCNN, showing that the functional expression of trajectory change is architecture dependent.

Current claim:

> **SGO is a finite-budget stochastic-environment subset-selection method. When only a subset of candidate environments can contribute to an update, retaining hard examples while reducing model-conditioned redundancy can improve performance in several tested structured regimes. Q-scaling establishes a robust finite-budget dependence across multiple architectures/tasks. A separate fixed-Q shared-reference intervention shows that schedules maximizing reference gradient novelty within tightly matched loss strata strongly outperform schedules minimizing it. Because that intervention also shifts known physical transformation diversity, gradient novelty is not yet isolated as the unique causal variable, and the functionally meaningful downstream mediator remains open.**

## 1. Seed is an environment index

Let `s` index stochastic environment `e_s`:

```text
s -> e_s -> g(theta, B, e_s) -> optimization trajectory -> learned function -> held-out behavior
```

with `g(theta,B,e)=grad_theta L(theta;B,e)`. SGO does not optimize integer seed values; it allocates finite update budget across model-conditioned stochastic environments.

## 2. Finite-budget selection problem

Suppose each step exposes K candidate environments but only Q<K can contribute. Loss-hard spends this budget on highest-current-loss candidates. Gradient-novel keeps a hard anchor while preferring additional hard candidates whose gradient signatures are less redundant with already selected directions.

An idealized quantity is residual gradient energy outside the selected span:

```text
novelty(e | S) = ||g_e - P_S g_e||^2 / (||g_e||^2 + eps).
```

The implementation uses cheaper final-layer cosine/signature proxies. Their intended role is approximately `error relevance × directional non-redundancy`.

This equation motivates the selector but is not itself an identified mediator.

## 3. Q-scaling evidence: selector freedom is finite-budget dependent

### Digits/geometric, MLP #1

K=16; Q={2,4,8,12,16}. Heldout mean benefits: `+2.151,+2.590,+2.076,+0.673,0.000` pp.

- frozen attenuation: **+2.034 pp**;
- one-sided p: **1.06e-7**;
- Q=16 exact identity 30/30.

### Digits/geometric, MLP #2 fresh

Benefits: `+2.613,+2.443,+2.156,+0.885,0.000` pp.

- attenuation: **+2.086 pp**;
- one-sided p: **5.16e-7**;
- Q=16 exact identity 30/30.

### Digits/geometric, SmallCNN

Benefits: `+2.034,+2.073,+1.195,+1.578,0.000` pp.

- attenuation: **+1.265 pp**;
- 95% CI: **+0.463 to +2.067 pp**;
- p: **0.001548**;
- Q=16 exact identity 30/30.

Post-hoc exclusion of Q16 gives low mean(Q2,Q4)-Q12 +0.476 pp, 95% CI [-0.537,+1.490], p=0.172. Thus this block alone does not establish attenuation among non-full Q values.

### FashionMNIST, Tiny Transformer

K=8; Q={2,4,6,8}. Benefits: `+0.846,+0.026,+0.245,0.000` pp.

- attenuation: **+0.314 pp**;
- SE: 0.157 pp;
- 95% CI: **-0.0065 to +0.6340 pp**;
- preregistered one-sided p: **0.027264**;
- Q=8 exact identity 30/30.

This is cross-task support but weaker than the Digits blocks.

### CIFAR-10, ResNet-20

K=8; Q={2,4,6,8}. Benefits: `+0.1960,+0.0775,+0.0399,0.0000` pp.

- low-Q mean: **+0.13675 pp**;
- high-Q mean: **+0.01997 pp**;
- attenuation: **+0.11679 pp**;
- SE: 0.04071 pp;
- 95% CI: **+0.03353 to +0.20005 pp**;
- p: **0.003804**;
- Q=8 exact identity 30/30.

Post-hoc exclusion of Q8 gives low mean(Q2,Q4)-Q6 **+0.09682 pp**, 95% CI **+0.00324 to +0.19040 pp**, descriptive p=0.02152. This reduces concern that the frozen CIFAR PASS is only an arithmetic consequence of averaging an exact-zero Q=K endpoint.

### Common Q-scaling structure

Selected pairwise-gradient novelty differences contract toward Q=K:

- SmallCNN/Digits: +0.3095,+0.1475,+0.0546,+0.02325,0;
- Fashion/Tiny Transformer: +0.1618,+0.0476,+0.0243,0;
- CIFAR/ResNet-20: +0.1004,+0.0324,+0.0125,0.

The performance curve is not universally monotonic. The robust common facts are:

1. selector freedom exists only while Q<K;
2. the measured nonredundancy manipulation contracts as coverage increases;
3. Q=K makes the two update rules exactly identical;
4. preregistered low-vs-high benefit contrasts are positive in all five blocks, with different strength;
5. CIFAR provides positive exploratory attenuation even without the Q=K point.

Q=K identity is a strong control, not causal proof.

## 4. Fixed-Q shared-reference intervention

Issue #80 attacks nonredundancy more directly without varying Q.

For each replicate and reference step:

1. expose the same K=16 candidates to one common reference model state;
2. rank by current loss;
3. always retain rank1;
4. require exactly one selection from each identical adjacent loss-rank pair `(2,3)`, `(4,5)`, `(6,7)`;
5. enumerate the same eight feasible Q=4 subsets;
6. define MAXNOV as highest mean pairwise gradient novelty and MINNOV as lowest;
7. advance only the common reference with ordinary loss-hard top4;
8. seal complete MAXNOV/MINNOV schedules;
9. reset two identical SmallCNNs and train them on those fixed schedules;
10. seal both arms before constructing fresh heldout environments.

This removes adaptive arm-state selection as a source of schedule divergence.

### Frozen manipulation

`N_r = mean_step(novelty_MAXNOV - novelty_MINNOV)`.

- mean: **+0.231634**;
- SE: 0.002623;
- 95% CI: **[+0.226269,+0.236999]**;
- p: **4.12e-37**;
- 30/30 positive.

### Frozen hardness equivalence

`z_hard=(selected_mean_loss-candidate_mean_loss)/(candidate_loss_sd+1e-8)`.

The frozen equivalence margin is ±0.10 candidate-pool SD units.

- mean MAXNOV-MINNOV: **-0.009100 SD**;
- SE: 0.002458;
- 90% CI: **[-0.013276,-0.004924]**;
- TOST lower/upper p: **2.95e-26 / 1.62e-28**.

Equivalence passes decisively. The raw reference selected-loss difference is only -0.001248 CE.

### Frozen performance

`B_r=heldout_mean_MAXNOV-heldout_mean_MINNOV`.

- mean: **+4.66929 pp**;
- SE: **0.72486 pp**;
- 95% CI: **[+3.18679,+6.15180] pp**;
- one-sided p: **2.389e-7**;
- 26/30 positive.

This is the strongest direct schedule-intervention result in the project to date.

Secondary clean benefit is -1.755 pp. The intervention therefore appears to improve the full geometric-shift regime while not necessarily improving clean performance. p10/minimum effects are descriptive only.

## 5. Why Issue #80 still does not isolate gradient-specific causality

The environment generator has known physical parameters: rotation, x/y translation, blur, contrast, brightness and noise. In a post-hoc audit, these seven dimensions were standardized across the 64 training environments and mean selected-set pairwise Euclidean distance was measured.

- MAXNOV parameter diversity: 3.97085;
- MINNOV: 3.47815;
- difference: **+0.492697**;
- SE: 0.009559;
- 95% CI: **[+0.473147,+0.512246]**;
- 30/30 positive.

Step-level gradient-novelty difference and parameter-diversity difference are positively associated (r≈0.276).

Thus the treatment defined by reference gradient novelty also changes a known environment-space property. A correct causal statement is:

```text
within matched loss-rank strata,
choosing the feasible schedules with maximum reference gradient novelty
causes a large heldout difference relative to minimum-novelty schedules,
but gradient novelty is correlated with physical transformation diversity in this intervention.
```

It is too strong to say “gradient novelty is the unique causal mediator.”

The earlier gradient-novel versus parameter-novel selector experiment shows that model-conditioned gradient information adds value beyond a simple physical-distance selector in that tested protocol. This makes a pure parameter-distance explanation less plausible, but it does not orthogonalize the two quantities in Issue #80.

## 6. Immediate descent and mean-gradient estimation are insufficient

Loss-hard can produce larger one-step loss decrease, and gradient novelty did not outperform random sampling as a mean-gradient estimator in the mechanism audit. If several hard environments induce nearly the same correction, spending multiple update slots on them can be locally sensible while covering little new unresolved structure.

SGO can therefore trade local greediness for broader allocation. The theory is not “maximize diversity” without qualification: pure diversity without hardness is weak, and larger accumulated gradient effective rank is insufficient.

## 7. Representation rank: marker, not mediator

Raw hidden representation effective rank has a fixed-parameterization condition-average predictive record, but direct interventions narrow its interpretation:

1. raw-rank attenuation failed the first budget mediator threshold;
2. function-preserving positive diagonal reparameterization moved raw rank while logits/predictions/metrics remained identical;
3. raw rank is coordinate-dependent and not an intrinsic causal state variable;
4. channel-standardized rank removed that scaling freedom but failed fresh budget coupling (`p=0.5063`).

The mediator search should not continue through post-hoc rank normalizations.

## 8. Downstream conversion is architecture/regime dependent

MLP reserve-image testing found full geometric benefit +2.274 pp and full-minus-clean interaction +2.906 pp (`p=0.001726`). SmallCNN found full benefit +2.417 pp but clean benefit +3.436 pp and full-minus-clean interaction -1.019 pp (`p=0.792`).

Issue #80 adds another architecture-specific expression: MAXNOV strongly improves shifted heldout mean but has a negative secondary clean contrast. Therefore the internal trajectory change and its task-metric expression must be kept separate.

## 9. Current causal picture

```text
binding subset budget
    -> selector freedom among hard candidates
    -> model-conditioned schedule allocation differs
    -> optimization trajectory / learned function differs
    -> architecture/task-dependent performance expression
```

Q-scaling strongly supports the first structural step. Issue #80 strengthens the middle step by intervening at fixed Q and matched reference hardness. The unresolved specificity is now:

```text
gradient nonredundancy itself
    versus correlated environment-space diversity / other schedule properties.
```

The downstream functionally meaningful mediator also remains open.

## 10. Highest-value next falsification

The next decisive experiment is **physical-parameter-diversity-matched gradient-novelty reversal** under the same shared-reference/fixed-Q framework.

A clean design should:

1. expose the same K candidates on one common reference state;
2. keep Q fixed and maintain a frozen hardness equivalence rule;
3. compute both reference gradient novelty and the seven-dimensional standardized physical transformation diversity for every feasible subset;
4. form MAXNOV/MINNOV schedules only from subset pairs whose physical diversity is within a preregistered matching margin;
5. require a successful gradient-novelty manipulation and both hardness and parameter-diversity equivalence before heldout interpretation;
6. seal all schedules before arm training and all states before heldout construction.

A positive result would substantially strengthen the claim that model-conditioned gradient nonredundancy contributes beyond obvious environment-parameter diversity. A valid null would directly narrow the present interpretation.

Separately, the downstream mediator should be attacked with prospective training-only function-space diagnostics rather than rank proxies.

## Working claim

> **Under a binding finite update budget, selecting stochastic environments that are both currently hard and non-redundant in model-conditioned gradient space can improve performance in several tested structured regimes. Finite-budget dependence is replicated across multiple architectures/tasks. A fixed-Q, shared-reference, loss-stratified intervention additionally shows a large benefit for schedules chosen to maximize reference gradient novelty over schedules chosen to minimize it. This is strong direct intervention evidence for the usefulness of nonredundancy-associated schedule allocation, but the intervention also increases known physical transformation diversity, so gradient novelty has not yet been isolated as the unique causal variable. The downstream learned-function mediator remains unidentified.**
