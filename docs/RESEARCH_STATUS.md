# Research status

A finding is labeled **supported** only within its stated comparison, task, and statistical rule. Negative/null results are retained.

## Supported findings

| Topic | Evidence | Current conclusion |
|---|---|---|
| Structured Digits geometric shifts | MLP paired experiments | hardness + gradient novelty can improve held-out mean/tail performance over loss-hard selection |
| Model-conditioned vs physical diversity | gradient novelty vs transformation-parameter novelty | model-dependent gradient diversity contains useful information beyond physical environment distance in the tested MLP |
| CNN replication | 20 paired runs | mean and minimum-environment gains vs loss-hard survive Holm correction |
| Optimizer replication | tuned AdamW and tuned SGD+momentum | effect is not explained by AdamW alone in the tested MLP |
| RNG candidate compression | prefix/compression sweeps | moderate cheap prefiltering can reduce gradient evaluations; aggressive compression loses tail coverage |
| Learned RNG relevance | original + shifted-coordinate generators | useful RNG relevance can be learned from training-only gradient information; stale fingerprint transfer fails |
| Soft RNG relevance | cross-generator weighted-top12 control | soft relevance weighting is more robust than forcing an exact hard top-k set in the tested generator |
| Relative redundancy control | Digits + independent Synthetic | absolute cosine targets do not transfer; within-step normalization transfers better in the tested tasks |
| CIFAR-10 / ResNet-20 mean | 40 paired primary runs | mean +0.1206 pp; Holm(5) p=0.01336 under the unchanged primary protocol |
| Gradient mechanism audit | mean/tail gradient + one-step tests | final gains are not explained by superior mean-gradient estimation or maximal one-step loss decrease alone |

## Prospectively supported but not universal-proof

| Topic | Evidence | Current conclusion |
|---|---|---|
| Representation effective-rank sign | seven registered tests across five datasets | sign(delta representation effective rank) matched sign(mean held-out benefit) in all seven tests; the new FashionMNIST/Tiny Transformer test changed dataset, generator, and architecture and passed with delta rank +0.0542 and mean benefit +0.736 pp |
| CIFAR p10/minimum direction | 40 paired primary runs | positive direction, but Holm p=0.05294 / 0.08815; tail robustness remains unconfirmed |

## Important negative or null results

- More candidate seeds are not monotonically better under fixed compute.
- Worst-only training can damage mean and clean performance.
- Pure gradient-diversity selection without hardness is weak.
- Full-network gradient signatures did not outperform the cheaper final-layer proxy in the tested MLP.
- A selector learned on one task does not transfer universally.
- Long raw RNG fingerprints with irrelevant coordinates can be worse than compact relevant fingerprints.
- An old learned RNG fingerprint does not transfer automatically when the generator moves relevant coordinates.
- Gradient novelty is not a superior estimator of the mean expected gradient relative to random sampling in the mechanism audit.
- Loss-hard selection produces larger immediate one-step loss decrease in the tested audit.
- Larger accumulated gradient effective rank does not imply benefit; Breast-high is a direct counterexample.
- An absolute selected-gradient cosine target can be infeasible on another task and force controller saturation.
- Relative redundancy control is not a task-level safety guarantee; Breast-high remains a counterexample.
- The attempted per-environment representation-rank-SD tail rule failed on Wine and is retired.
- Optimizer under-tuning can create a false selector failure.

## Current open work

- Complete the separately preregistered CIFAR-10 / ResNet-20 representation-rank trajectory audit.
- Prospectively falsify the representation-rank sign rule on a larger or otherwise materially different architecture after the Tiny Transformer PASS.
- Develop a new training-only tail-safety diagnostic without retuning the failed rule.
- Run GPU-vectorized wall-clock benchmarks at matched update/candidate budgets.
- Test naturally stochastic simulators or non-handcrafted environment processes.

## Public claim boundary

Safe public wording:

> Gradient-aware selection of stochastic training environments can improve held-out optimization/generalization under some structured stochastic shifts in the tested settings. A fixed-protocol CIFAR-10 / ResNet-20 study supports a small mean improvement, and training-only representation effective-rank change is a promising prospective predictor of the sign of mean benefit. The directional rule has now passed a small FashionMNIST Transformer architecture-shift test as well as the earlier MLP-based tests.

Do not claim a universally good seed family, universal selector/controller, universal representation-rank gate, general Transformer validation, confirmed CIFAR tail robustness, or GPU efficiency advantage.
