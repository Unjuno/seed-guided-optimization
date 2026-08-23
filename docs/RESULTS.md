# Results summary

## Supported findings

### Seed/environment choice affects optimization trajectory

Even with the same underlying training dataset, stochastic environments produce different gradients and final held-out performance.

### Hardness and gradient novelty capture different information

Loss-hard selection concentrates on environments that are immediately difficult. Gradient-novel selection adds environments that produce non-redundant update directions. On the MLP geometric domain-shift benchmark, a hard + gradient-novel policy substantially outperformed both loss-hard and transformation-parameter novelty.

### CNN replication

A small CNN was tested with the same geometric environment generator. Across 20 paired runs, gradient-novel vs. loss-hard improved held-out mean accuracy by +2.25 percentage points and minimum-environment accuracy by +2.57 points. Both survived five-metric Holm correction (`p=0.0171` and `p=0.00573`, respectively). The p10 difference was positive but did not survive correction.

Parameter novelty also improved CNN mean accuracy relative to loss-hard, so the MLP result that gradient novelty strongly dominates transformation-parameter novelty is not yet architecture-general.

### Optimizer replication after tuning

An initial SGD+momentum comparison was confounded by an under-tuned learning rate. A separate loss-hard learning-rate sweep was therefore performed before the final selector comparison. With the tuned setting, gradient-novel improved held-out mean and p10 under both AdamW and SGD+momentum.

- AdamW: mean +2.69 pp (`Holm p=5.1e-5`), p10 +2.94 pp (`Holm p=0.00142`).
- SGD+momentum: mean +1.88 pp (`Holm p=0.0362`), p10 +3.02 pp (`Holm p=0.00250`).

This is evidence against an AdamW-only explanation, not proof of optimizer universality.

### Model-dependent novelty matters

Environment-parameter distance is not sufficient in the MLP benchmark. Parameter-novel and gradient-novel selectors chose environments with similar transformation-space diversity, but gradient-novel achieved materially better held-out performance.

### RNG prefiltering can reduce expensive candidate evaluation

A cheap prefilter based on RNG coordinates that actually drive the environment generator reduced candidate gradient evaluations without degrading the main metrics at moderate compression. The current conservative point is 16 candidates → 12 prefiltered candidates → 4 backward environments.

### Relevant RNG coordinates can be learned from training-only gradients

A 64-coordinate RNG fingerprint was calibrated using separate training-only environment seeds and a fixed calibration model. A RidgeCV mapping from RNG outputs to a low-dimensional gradient representation recovered most of the oracle-relevant coordinates. Using the learned top coordinates for 16→8 prefiltering matched the oracle-coordinate prefilter on all five corrected metrics and outperformed the raw 64-coordinate distance on held-out mean and p10. This supports automatic fingerprint discovery within the tested generator, not universal cross-generator transfer.

A follow-up generator with eight relevant RNG coordinates scattered across the 64-dimensional RNG window showed an important refinement: coordinate relevance was again recoverable, but Euclidean diversity in the recovered coordinates was not the best prefilter. A learned RNG→gradient embedding and a learned gradient-distance metric preserved gradient-space coverage better in training-only calibration. In 20 paired runs, the predicted-gradient prefilter reduced candidate gradient evaluation from 16 to 8 and improved held-out mean by about +2.30 pp and p10 by +1.73 pp versus full 16-candidate evaluation in that follow-up benchmark. The preferred fingerprint representation is therefore generator-dependent.

### Compression has a real failure boundary

Reducing the candidate set to 4 removed too much gradient coverage: p10 and minimum held-out accuracy dropped sharply. Candidate reduction is therefore an optimization problem, not a monotonic speedup.

## Negative results retained

- Worst-only / very narrow CVaR can over-focus on outlier environments.
- Pure gradient diversity without hardness can hurt tail performance.
- Full-gradient signatures do not automatically improve selection over final-layer signatures.
- A fixed selector learned on one task is not universally transferable.
- Long raw RNG fingerprints containing irrelevant coordinates are worse than short relevant fingerprints.
- Low-heterogeneity tasks may show almost no benefit from active seed selection.
- An under-tuned optimizer can create a false negative; optimizer hyperparameters must be tuned independently of the selector comparison.
- Learning which RNG coordinates matter does not imply that Euclidean diversity in those coordinates is the best prefilter metric for every generator.

## Public claim boundary

The present experiments justify saying that gradient-aware environment-seed selection **can** improve optimization/generalization in tested stochastic-shift settings and that the effect has replicated across one MLP, one CNN, AdamW, and tuned SGD+momentum. They also justify proof-of-concept claims that generator-relevant RNG fingerprint components can be inferred from training-only gradient information and that the RNG representation itself can be learned to preserve gradient-space diversity.

They do not justify saying that a universally optimal seed family exists, that seed values themselves have semantic classes, or that the method is validated on modern large-scale neural networks.
