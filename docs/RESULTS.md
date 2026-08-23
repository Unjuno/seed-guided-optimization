# Results summary

## Supported findings

### Seed/environment choice affects optimization trajectory

Even with the same underlying training dataset, stochastic environments produce different gradients and final held-out performance.

### Hardness and gradient novelty capture different information

Loss-hard selection concentrates on environments that are immediately difficult. Gradient-novel selection adds environments that produce non-redundant update directions. On the geometric domain-shift benchmark, a hard + gradient-novel policy substantially outperformed both loss-hard and transformation-parameter novelty.

### Model-dependent novelty matters

Environment-parameter distance is not sufficient. In the geometric experiment, parameter-novel and gradient-novel selectors chose environments with similar transformation-space diversity, but gradient-novel achieved materially better held-out performance.

### RNG prefiltering can reduce expensive candidate evaluation

A cheap prefilter based on RNG coordinates that actually drive the environment generator reduced candidate gradient evaluations without degrading the main metrics at moderate compression. The current conservative point is 16 candidates → 12 prefiltered candidates → 4 backward environments.

### Compression has a real failure boundary

Reducing the candidate set to 4 removed too much gradient coverage: p10 and minimum held-out accuracy dropped sharply. Candidate reduction is therefore an optimization problem, not a monotonic speedup.

## Negative results retained

- Worst-only / very narrow CVaR can over-focus on outlier environments.
- Pure gradient diversity without hardness can hurt tail performance.
- Full-gradient signatures do not automatically improve selection over final-layer signatures.
- A fixed selector learned on one task is not universally transferable.
- Long raw RNG fingerprints containing irrelevant coordinates are worse than short relevant fingerprints.
- Low-heterogeneity tasks may show almost no benefit from active seed selection.

## Public claim boundary

The present experiments justify saying that gradient-aware environment-seed selection **can** improve optimization/generalization in tested stochastic-shift settings.

They do not justify saying that a universally optimal seed family exists, that seed values themselves have semantic classes, or that the method is already validated on modern large-scale neural networks.
