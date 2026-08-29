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
| CIFAR-10 / ResNet-20 primary mean | 40 paired runs | mean +0.1206 pp; Holm(5) p=0.01336 under the unchanged primary protocol |
| Gradient mechanism audit | mean/tail gradient + one-step tests | final gains are not explained by superior mean-gradient estimation or maximal one-step loss decrease alone |
| FashionMNIST/Tiny Transformer extension | 20 independent new pairs | frozen condition-average rank direction replicated: rank +0.07853 and held-out mean +0.4377 pp |

## Prospectively supported but not universal proof

| Topic | Evidence | Current conclusion |
|---|---|---|
| Representation effective-rank sign | **8 registered condition-level tests across 6 datasets** | sign(delta representation effective rank) matched sign(mean held-out benefit) in all eight registered conditions; this is condition-average evidence, not a per-run gate |
| FashionMNIST/Tiny Transformer | initial n=10 + independent extension n=20 | initial PASS was followed by an exact-protocol independent directional replication; combined n=30 is precision-only |
| CIFAR-10 / ResNet-20 representation rank | independent reps 40-49, n=10 | mean delta rank +0.03205 (>0.01 tolerance) predicted positive; held-out mean +0.1703 pp; frozen decision PASS |
| CIFAR p10/minimum direction | 40 paired primary runs | positive direction, but Holm p=0.05294 / 0.08815; tail robustness remains unconfirmed |

The eight registered conditions span Digits (three generators), Wine, Iris, Diabetes regression, FashionMNIST, and CIFAR-10. Three conditions share Digits, so the record must not be treated as eight independent datasets or eight independent Bernoulli trials.

## Current working mechanism

The best-fitting theory is no longer "more gradient diversity is better." The repository now uses the narrower working model:

```text
hard + non-redundant environment-induced gradients
    -> better finite-budget coverage of unresolved task-relevant directions
    -> trajectory change
    -> richer reusable representation when conversion succeeds
    -> held-out mean benefit.
```

Raw accumulated gradient effective rank is not sufficient. Breast-high is the central counterexample: gradient rank expands strongly while representation rank falls and held-out mean does not improve.

Representation effective rank is therefore treated as a **training-only directional proxy for representation conversion**, not as a proven causal variable. See `THEORETICAL_FRAMEWORK.md`.

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
- The representation-rank rule is not reliable as a per-replicate gate; both Fashion and CIFAR contain individual sign mismatches.
- Optimizer under-tuning can create a false selector failure.
- Single-thread deterministic hosted-CPU execution did **not** eliminate CIFAR cross-run numerical drift; the preregistered execution audit decision is **DRIFT PERSISTS**.

## Execution reproducibility finding

A preregistered hosted-CPU audit reran CIFAR reps 45 and 46 twice with OMP/MKL/OpenBLAS/PyTorch thread counts forced to one.

- all scientific fields bitwise equal: false;
- max rank drift: 0.427255;
- max accuracy drift: 0.014667;
- aggregate rank direction: positive in A and B;
- aggregate held-out mean direction: positive in A and B;
- decision: **DRIFT PERSISTS**.

Rep45 was bitwise-identical between AMD EPYC 7763 and AMD EPYC 9V74. Rep46 drifted between AMD EPYC 7763 and Intel Xeon 6973P-C under the same pinned software/thread controls. Hardware-dependent numerical paths are therefore a strong candidate, but the sole cause has not been isolated.

## Current open work

1. Test the proposed mechanism causally with mediation, representation intervention, and matched structured-vs-unstructured novelty experiments.
2. Prospectively falsify the frozen condition-average rank rule on a materially larger/different architecture or modality.
3. Derive an actionable early-trajectory diagnostic; final representation-rank comparison currently requires training both methods.
4. Develop a new training-only tail-safety theory without retuning the failed rank-SD rule.
5. Run exact reproducibility experiments on pinned hardware and GPU-vectorized wall-clock benchmarks at matched update/candidate budgets.
6. Test naturally stochastic simulators or non-handcrafted environment processes.

## Public claim boundary

Safe public wording:

> Gradient-aware selection of stochastic training environments can improve held-out optimization/generalization under some structured stochastic shifts in the tested settings. A fixed-protocol CIFAR-10 / ResNet-20 study supports a small mean improvement. Across eight registered prospective conditions on six datasets, training-only representation effective-rank direction has matched the sign of condition-average mean benefit, with an independent FashionMNIST/Tiny Transformer replication and a CIFAR/ResNet PASS. The current mechanism hypothesis is finite-budget coverage of unresolved task-relevant gradient directions followed by conversion into reusable representation structure.

Do not claim a universally good seed family, universal selector/controller, causal representation-rank law, calibrated per-run gate, confirmed CIFAR tail robustness, general large-Transformer validity, bitwise cross-hardware hosted-CPU reproducibility, or GPU efficiency advantage.
