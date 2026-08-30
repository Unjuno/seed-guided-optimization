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
| Finite-budget coverage mechanism | preregistered Q-scaling, 30 pairs | low-vs-high Q benefit attenuation +2.034 pp, one-sided p=1.06e-7; at Q=K both methods are exactly identical |

## Prospectively supported but not universal proof

| Topic | Evidence | Current conclusion |
|---|---|---|
| Representation effective-rank sign | **9 registered condition-level tests across 6 datasets** | sign(delta representation effective rank) matched sign(mean held-out benefit) in all nine registered conditions; this is condition-average evidence, not a per-run gate or causal law |
| FashionMNIST/Tiny Transformer | initial n=10 + independent extension n=20 | initial PASS was followed by an exact-protocol independent directional replication; combined n=30 is precision-only |
| FashionMNIST/Medium Transformer | n=10 architecture-capacity falsification | rank +0.10293 predicted positive and held-out mean was +0.9579 pp; frozen decision PASS |
| CIFAR-10 / ResNet-20 representation rank | independent reps 40-49, n=10 | mean delta rank +0.03205 (>0.01 tolerance) predicted positive; held-out mean +0.1703 pp; frozen decision PASS |
| CIFAR p10/minimum direction | 40 paired primary runs | positive direction, but Holm p=0.05294 / 0.08815; tail robustness remains unconfirmed |

The nine registered conditions span Digits (three generators), Wine, Iris, Diabetes regression, FashionMNIST (Tiny and Medium Transformer conditions), and CIFAR-10. They are not nine independent datasets or independent Bernoulli trials.

## Current working mechanism

The best-supported theory is now narrower and better separated into a supported component and an open mediator:

```text
binding subset-update budget
    + hard, non-redundant environment-induced gradients
    -> better coverage of unresolved task-relevant directions
    -> trajectory/internal-learning change
    -> held-out mean benefit in structured regimes.
```

The **finite-budget coverage component now has direct preregistered support**. In the Q-scaling audit with K=16, held-out mean benefit was +2.151 pp at Q=2, +2.590 pp at Q=4, +2.076 pp at Q=8, +0.673 pp at Q=12, and exactly 0 at Q=16. The frozen low-Q minus high-Q attenuation was +2.034 pp (`p=1.06e-7`, one-sided). At Q=16 every non-timing scientific output was exactly identical across methods in all 30 pairs.

Raw accumulated gradient effective rank is not sufficient. Breast-high remains the central counterexample.

Representation effective rank remains a useful **condition-average directional marker**, but the new budget audit did not support it as the direct quantitative mediator: low-vs-high rank attenuation was +0.0735 with one-sided `p=0.1095`. The frozen mechanism decision was therefore **PARTIAL PASS**, not STRONG THEORY PASS.

See `THEORETICAL_FRAMEWORK.md` and `BUDGET_COVERAGE_MECHANISM.md`.

## Important negative or null results

- More candidate seeds are not monotonically better under fixed compute.
- Worst-only training can damage mean and clean performance.
- Pure gradient-diversity selection without hardness is weak.
- Full-network gradient signatures did not outperform the cheaper final-layer proxy in the tested MLP.
- A selector learned on one task does not transfer universally.
- Long raw RNG fingerprints with irrelevant coordinates can be worse than compact relevant fingerprints.
- An old learned RNG fingerprint does not transfer automatically when the generator moves relevant coordinates.
- Gradient novelty is not a superior estimator of the mean expected gradient relative to random sampling in the mechanism audit.
- Loss-hard selection can produce larger immediate one-step loss decrease.
- Larger accumulated gradient effective rank does not imply benefit; Breast-high is a direct counterexample.
- An absolute selected-gradient cosine target can be infeasible on another task and force controller saturation.
- Relative redundancy control is not a task-level safety guarantee.
- The attempted per-environment representation-rank-SD tail rule failed on Wine and is retired.
- The representation-rank rule is not reliable as a per-replicate gate; Fashion and CIFAR contain individual sign mismatches.
- **Representation-rank attenuation did not pass the preregistered Q-scaling mechanism test** (`p=0.1095`), so effective rank should not be claimed as the sole/direct mediator of budget dependence.
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

Rep45 was bitwise-identical between AMD EPYC 7763 and AMD EPYC 9V74. Rep46 drifted between AMD EPYC 7763 and Intel Xeon 6973P-C under the same pinned software/thread controls. Hardware-dependent numerical paths are a strong candidate, but the sole cause has not been isolated.

## Current open work

1. **Identify the mediator after coverage:** run representation interventions and matched-coverage/different-conversion experiments to distinguish causal representation structure from effective-rank marking.
2. Run a structured-vs-unstructured matched-novelty experiment with comparable gradient-novelty magnitude but different latent-factor reuse.
3. Derive an actionable early-trajectory mediator/predictor without calibrating on final held-out results.
4. Prospectively falsify the frozen condition-average rank marker on a materially different modality or larger architecture.
5. Develop a genuinely new training-only tail-safety theory without retuning the failed rank-SD rule.
6. Repeat the frozen Q-scaling contrast on a different task only after the mediator hypothesis is fixed.
7. Use pinned hardware for bitwise studies and run GPU-vectorized wall-clock benchmarks at matched update/candidate budgets.
8. Test naturally stochastic simulators or non-handcrafted environment processes.

## Public claim boundary

Safe public wording:

> Gradient-aware selection of stochastic training environments can improve held-out performance under some structured stochastic shifts. A preregistered Q-scaling audit directly supports a finite-budget subset-coverage explanation: the gradient-novelty advantage is large when only a small subset of candidates can contribute to each update, attenuates when most candidates are used, and becomes exactly zero when all K candidates are used. Across nine registered prospective conditions on six datasets, training-only representation effective-rank direction has matched the sign of condition-average mean benefit, but the Q-scaling audit did not support effective rank as the direct quantitative mediator of the budget effect.

Do not claim a universally good seed family, universal selector/controller, causal representation-rank law, calibrated per-run gate, confirmed CIFAR tail robustness, general large-Transformer validity, bitwise cross-hardware hosted-CPU reproducibility, or GPU efficiency advantage.
