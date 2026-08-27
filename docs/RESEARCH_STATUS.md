# Research status

This file is the shortest map from experiment to claim status. A result is promoted to **supported** only when the stated comparison and statistical rule are satisfied. Null and negative results are retained.

## Supported findings

| Topic | Evidence | Current conclusion |
|---|---|---|
| Structured Digits geometric shifts | MLP paired experiments | hardness + gradient novelty can improve held-out mean/tail performance over loss-hard selection |
| Model-conditioned vs physical diversity | gradient novelty vs transformation-parameter novelty | model-dependent gradient diversity contains useful information beyond physical environment distance in the tested MLP |
| CNN replication | 20 paired runs | mean and minimum-environment gains vs loss-hard survive Holm correction |
| Optimizer replication | tuned AdamW and tuned SGD+momentum | effect is not explained by AdamW alone in the tested MLP |
| RNG candidate compression | prefix/compression sweeps | moderate cheap prefiltering can reduce gradient evaluations; aggressive compression loses tail coverage |
| Learned RNG relevance | original + shifted-coordinate generators | useful RNG relevance can be learned from training-only gradient information; stale fingerprint transfer fails |
| Soft RNG relevance | cross-generator weighted-top12 control | soft relevance weighting is more robust than forcing an exact hard top-k coordinate set in the tested generator |
| Relative redundancy control | Digits + independent Synthetic | absolute cosine targets do not transfer; a within-step normalized operating point transfers better in the tested tasks |
| Gradient mechanism audit | mean/tail gradient + one-step update tests | final gains are not explained by superior mean-gradient estimation or maximal one-step loss decrease alone |

## Suggestive / not confirmatory

| Topic | Evidence | Current conclusion |
|---|---|---|
| CIFAR-10 / ResNet-20 | first 20 paired primary runs | mean +0.138 pp and p10 +0.143 pp, but five-metric Holm correction is not significant; 40-pair extension is in PR #11 |
| Cross-task controller universality | relative-control studies | normalized control is promising, but the correct operating point is not established as universal |

## Important negative or null results

- More candidate seeds are not monotonically better under fixed compute.
- Worst-only training can damage mean and clean performance.
- Pure gradient-diversity selection without hardness is weak.
- Full-network gradient signatures did not outperform the cheaper final-layer proxy in the tested MLP.
- A selector learned on one task does not transfer universally.
- Long raw RNG fingerprints with irrelevant coordinates can be worse than compact relevant fingerprints.
- An old learned RNG fingerprint does not transfer automatically when the generator moves relevant coordinates.
- Gradient novelty is not a superior estimator of the mean expected gradient relative to random sampling in the mechanism audit.
- Loss-hard selection produces larger immediate one-step loss decrease in the tested audit, so the long-run benefit is not greedy loss reduction.
- An absolute selected-gradient cosine target can be infeasible on another task and force the controller to saturate.
- Initial optimizer under-tuning can create a false selector failure; optimizer tuning must be separated from selector evaluation.

## In progress

- **Issue #1 / PR #11:** extend CIFAR-10 / ResNet-20 from 20 to 40 paired replicates without changing the protocol.
- **Issue #2:** GPU-vectorized wall-clock benchmark.
- **PR #10:** relative redundancy control on Breast Cancer Wisconsin low/high corruption regimes.
- **Issue #12:** identify trajectory-level diagnostics that predict when novelty helps without using held-out test environments.

## Completed roadmap items

- **Issue #3:** training-only discovery of compact RNG relevance without being told the relevant generator coordinates — closed as completed.

## Public claim boundary

Safe public wording:

> Gradient-aware selection of stochastic training environments can improve held-out optimization/generalization under some structured stochastic shifts in the tested settings, and candidate-selection overhead can be reduced using learned or known stochastic fingerprints.

Do not claim a universally good seed family, universal selector, universal redundancy target, large-scale validation, or GPU efficiency advantage.
