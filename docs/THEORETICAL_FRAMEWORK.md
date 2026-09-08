# Theoretical framework: finite-budget allocation and unresolved mediation

Updated2026-09-08. This is a working causal model, not a theorem that SGO improves generalization.

## 1. Separate three questions

**Algorithmic effect:** does replacing one environment-selection procedure with another
change the learned function and heldout performance under a fixed protocol?

**Mechanism specificity:** is that difference specifically attributable to gradient
nonredundancy, rather than correlated environment properties or other schedule changes?

**Downstream mediation:** through what learned functional change does the schedule
produce a performance difference?

The experiments support selected algorithmic contrasts. Mechanism specificity and
downstream mediation are not fully identified. Seed integers serve as environment
indices; no intrinsic semantic classes or universal seed-quality ordering are assumed.

## 2. What the finite-budget experiments establish

| Symbol | Meaning (Japanese) | SI unit | Definition | Domain/assumptions | Type |
|---|---|---|---|---|---|
| K | 候補環境数 |1| number available at a step |positive integer|scalar integer|
| Q | 採用環境数 |1| number contributing to the update |integer from1 through K|scalar integer|

Five preregistered30-repetition budget blocks have positive frozen low-minus-high
contrasts: Digits/MLP+2.034 and+2.086 pp,Digits/SmallCNN+1.265 pp,
FashionMNIST/Tiny Transformer+0.314 pp,and CIFAR/ResNet-20+0.11679 pp.
The Fashion two-sided95% interval crosses zero; its support is borderline.

At Q=K the selectors have no subset-choice freedom. Under identical starting states,
ordered inputs, preprocessing, optimizer states and deterministic numerical execution,
the two complete-subset updates are identical. Repeating the same update from the same
state then preserves equality at every subsequent step. This argument requires that
selector computations do not introduce unequal model/optimizer/RNG side effects.
It does not guarantee equality across hardware/backend changes.

Exact all-candidate identity is therefore an implementation control, not an independent
proof of why smaller-subset performance improves. Removing the structurally zero
endpoint post-hoc retains positive CIFAR attenuation but not an established SmallCNN
contrast. The budget-response shape is not a universal monotone law.

## 3. Fixed-budget interventions strengthen the procedural evidence

Issue80 fixes the adopted subset size while matching reference-loss strata. High
reference-gradient-novelty schedules beat low-novelty schedules by+4.669 pp, but also
increase scalar physical transformation diversity. This is not unique-gradient evidence.

Issue83 additionally matches standardized reference hardness and scalar total physical
diversity within frozen±0.05 calipers. Its high-minus-low benefit is+3.969922 pp,
95% CI[+3.050617,+4.889226],one-sided p5.10e-10. The reference matching gates pass.
The positive schedule contrast survives those two scalar balance constraints.

But an audit reveals more x/y translation variance and less blur/noise/brightness
variance in the high-gradient-novelty schedules. Matching average pairwise distance
cannot ensure factor-specific composition balance. A complete mathematical counterexample,
with variable definitions and unit checks, is in [the Issue83 result](PARAMETER_MATCHED_RESULT.md).

## 4. A fresh test of the alternative

Issue86 prospectively compares two schedule-pair families over the same feasible
subsets and matching constraints. One family maximizes reference gradient-novelty
separation. The other maximizes standardized x/y translation-variance separation
WITHOUT using gradient novelty as the selection score.

Both positive primary conditions reproduce in30 new repetitions:

| Schedule contrast | Heldout mean difference | 95% CI |
|---|---:|---|
| Gradient high minus low |+3.852029 pp|[+2.927096,+4.776962]|
| Translation-score high minus low |+1.768644 pp|[+0.580026,+2.957263]|

This supports an alternative SCORING PROCEDURE, not a gradient-independent causal
path. Translation-based selection also increases reference gradient novelty by0.105610.
Gradient-based selection increases translation variance by0.616683. These variables
remain jointly manipulated.

The descriptive difference between the two high-low contrasts is+2.083385 pp.
Because they have different low baselines and redistribute other physical factors,
this is not a mediated fraction, a unique-gradient-effect estimate, or a direct
ranking of the two high schedules. In particular, translation-high superiority over
the retained loss-hard reference is not established by its95% interval.

See [the complete spatial result](SPATIAL_ALLOCATION_RESULT.md), including baseline
contrasts, manipulation gates, all secondary endpoints and verification.

## 5. Revised candidate explanation

A useful hypothesis is that the selector allocates a limited number of updates to
particular unresolved and reusable variation factors. Gradient signatures may serve
as a model-conditioned way of discovering such factors without knowing their physical
parameter names. An explicit physical score may sometimes approximate this allocation.

This hypothesis explains why total diversity alone need not suffice and why different
selection procedures can be associated with useful learning. It is still a hypothesis:
the experiments do not identify a unique physical factor, task-relevance quantity,
internal feature change or gradient-geometry mediator.

Three distinct statements must not be conflated:

- not using gradient novelty to SCORE subsets;
- not changing gradient geometry;
- gradient geometry not MEDIATING the performance effect.

Issue86 establishes the first for the translation selector. It does not establish
the second or third. Both procedures still learn using gradients.

## 6. Restrictions inherited from previous falsification

Raw hidden effective rank is changed by function-preserving reparameterizations,
so it is not an intrinsic functional mediator. It can remain a marker under a fixed
parameterization. Standardized-rank budget coupling failed; additional post-hoc rank
normalizations are not a justified causal explanation.

Pure diversity and accumulated gradient rank are insufficient. Gradient novelty did
not dominate random sampling for mean-gradient estimation, and loss-hard can produce
a larger immediate loss decrease. Thus a pure one-step descent or estimator-variance
account does not fully explain the recorded final-performance comparisons.

MLP full-minus-clean specificity did not reproduce in SmallCNN. The structured/nuisance
matching program did not identify reusable-factor causality. Tail safety, per-run
controllers and universal cross-task validity remain open.

## 7. Next discriminating experiment, not yet completed

Independently establish whether feasible subset pairs can match reference hardness,
scalar total diversity AND spatial allocation while preserving a gradient-novelty
manipulation. Only then freeze a fresh confirmatory test. Failure of feasibility
should stop interpretation rather than trigger heldout-driven threshold adjustment.

Even that design would leave other factor distributions and downstream mediation
unresolved. A credible theory must state exactly what is manipulated and balanced,
not relabel any positive schedule contrast as proof of a favorite scalar mediator.

## Evidence boundary / ERROR CHECK

The current conclusions distinguish registered endpoints from post-hoc discovery,
reference-state balance from balance along diverging training trajectories, high-low
schedule contrasts from online gradnov/loss-hard comparisons, and structural identity
from causal identification. Statistical repetition units are independent training
blocks, not the many environment observations within each block. The reported errors
are conditional on fixed images/environment pools; there is no general performance theorem.
