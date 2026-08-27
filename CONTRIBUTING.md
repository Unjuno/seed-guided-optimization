# Contributing

This repository is experimental research code. Contributions are welcome when they preserve the distinction between exploratory evidence and supported public claims.

## Before adding an experiment

State the question and the main failure condition before looking at final held-out results. For selector comparisons, specify:

- dataset / model / optimizer;
- stochastic environment generator;
- training/selection seed pool;
- final held-out environment seed pool;
- candidate count `K` and backward/update count `Q`;
- primary comparison and metrics;
- replicate count;
- statistical correction family where applicable.

## Pairing and leakage

Whenever possible, compared methods should share within each replicate:

- model initialization;
- data split;
- minibatch order;
- candidate-environment schedule;
- final held-out evaluation environments.

Do not tune selector coefficients, controller thresholds, optimizer hyperparameters, or stopping rules using the final held-out environment pool.

## Negative results

Do not remove a method or experiment because it fails. The project deliberately retains failure modes such as worst-only training, pure diversity, stale RNG fingerprints, over-compression, absolute cross-task cosine targets, and underpowered external-validation results.

## Statistics

If multiple outcome metrics are treated as one hypothesis family, report the stated multiple-testing correction. Do not promote an isolated raw `p<.05` result as confirmatory when the corrected comparison is not significant.

Accuracy-like differences should be reported clearly as proportions or percentage points; avoid mixing the two.

## Reproducibility

Prefer scripts that accept explicit replicate ranges and deterministic seeds. Commit compact paired/statistical summaries to `results/`; large transient artifacts may remain in CI artifacts when regeneration is documented.

Update these files when a result changes the public claim boundary:

- `README.md`
- `docs/RESEARCH_STATUS.md`
- `docs/RESULTS.md`
- `docs/LIMITATIONS.md`
- `experiments/README.md` or `results/README.md` when new scripts/evidence files are added.

## Scope of language

Prefer statements such as "improves in the tested setting" or "supports the hypothesis under these conditions." Avoid universal claims about good seed numbers, universal gradient geometry, optimizer independence, or large-scale efficiency unless directly tested.
