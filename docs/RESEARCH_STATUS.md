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
| FashionMNIST/Tiny Transformer extension | 20 independent new pairs | frozen condition-average raw-rank direction replicated: rank +0.07853 and held-out mean +0.4377 pp |
| Finite-budget coverage mechanism | two preregistered n=30 Q-scaling runs on fresh blocks | low-vs-high benefit attenuation +2.034 pp (`p=1.06e-7`) then +2.086 pp (`p=5.16e-7`); Q=K gives exact method identity in both runs |
| Raw-rank coordinate dependence | preregistered function-preserving intervention, 20 reps × 2 methods | raw hidden effective rank can move by roughly +2 to -6 while logits/predictions/metrics remain identical; raw rank is not a functionally intrinsic causal quantity |

## Prospectively supported but not universal proof

| Topic | Evidence | Current conclusion |
|---|---|---|
| Raw representation effective-rank sign | **9 registered condition-level tests across 6 datasets** | sign(delta raw representation effective rank) matched sign(mean held-out benefit) in all nine registered conditions; fixed-parameterization condition-average marker only |
| FashionMNIST/Tiny Transformer | initial n=10 + independent extension n=20 | initial PASS followed by exact-protocol independent directional replication; combined n=30 is precision-only |
| FashionMNIST/Medium Transformer | n=10 architecture-capacity test | raw rank +0.10293 predicted positive and held-out mean was +0.9579 pp; frozen decision PASS |
| CIFAR-10 / ResNet-20 representation rank | independent reps 40-49, n=10 | raw rank +0.03205 predicted positive; held-out mean +0.1703 pp; frozen decision PASS |
| CIFAR p10/minimum direction | 40 paired primary runs | positive direction, but Holm p=0.05294 / 0.08815; tail robustness remains unconfirmed |

The nine registered raw-rank conditions span Digits (three generators), Wine, Iris, Diabetes regression, FashionMNIST (Tiny and Medium Transformer conditions), and CIFAR-10. They are not nine independent datasets or independent Bernoulli trials.

## Current working mechanism

The best-supported theory is now:

```text
binding subset-update budget
    + hard, non-redundant environment-induced gradients
    -> broader coverage of unresolved task-relevant directions
    -> trajectory / learned-function change
    -> held-out mean benefit in tested structured regimes.
```

The first Q-scaling audit with K=16 gave mean benefits +2.151, +2.590, +2.076, +0.673, and exactly 0 pp for Q=2,4,8,12,16. Frozen low-minus-high attenuation was +2.034 pp (`p=1.06e-7`).

A fully fresh preregistered replication using reps 300-329 and held-out seeds 17000-17079 gave +2.613, +2.443, +2.156, +0.885, and exactly 0 pp. Frozen attenuation was +2.086 pp (`p=5.16e-7`, approximate 95% interval +1.393 to +2.778 pp). Q=16 again had zero scientific mismatches.

Thus **finite-budget subset coverage is independently replicated within the Digits/geometric family**.

The causal internal mediator remains unidentified.

## Representation-rank mechanism status

Three results must be kept separate.

1. **Raw rank predicts condition-average direction under fixed parameterizations.** The prospective record remains 9/9 registered conditions.
2. **Raw rank is not a causal/functionally intrinsic quantity.** Issue #35 changed raw rank strongly under an exactly function-preserving positive diagonal reparameterization while every prediction and held-out/clean metric remained unchanged.
3. **Channel-standardized rank does not rescue the mediator theory.** In fresh Issue #38, benefit attenuation replicated strongly, but standardized-rank attenuation was `-0.00271` with one-sided `p=0.5063`. Frozen decision: **COVERAGE REPLICATION ONLY**.

Raw rank was secondary in Issue #38 and again showed positive attenuation (`+0.1780`, one-sided `p=0.0180`). This reinforces its interpretation as a repeatable **trajectory marker** rather than a causal state variable.

See `THEORETICAL_FRAMEWORK.md`, `FUNCTION_PRESERVING_RANK_INTERVENTION.md`, and `STANDARDIZED_RANK_BUDGET_RESULT.md`.

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
- The raw representation-rank rule is not reliable as a per-replicate gate.
- Raw representation-rank attenuation failed the first frozen Q-scaling mediator threshold (`p=0.1095`).
- Raw representation rank is coordinate-dependent under function-preserving reparameterization.
- Channel-standardized effective rank is scale-invariant but **failed** the fresh preregistered budget-coupling test (`p=0.5063`).
- Optimizer under-tuning can create a false selector failure.
- Single-thread deterministic hosted-CPU execution did **not** eliminate CIFAR cross-run numerical drift; frozen decision **DRIFT PERSISTS**.

## Execution reproducibility finding

A preregistered hosted-CPU audit reran CIFAR reps 45 and 46 twice with OMP/MKL/OpenBLAS/PyTorch thread counts forced to one.

- all scientific fields bitwise equal: false;
- max rank drift: 0.427255;
- max accuracy drift: 0.014667;
- aggregate rank direction: positive in A and B;
- aggregate held-out mean direction: positive in A and B;
- decision: **DRIFT PERSISTS**.

Rep45 was bitwise-identical between AMD EPYC 7763 and AMD EPYC 9V74. Rep46 drifted between AMD EPYC 7763 and Intel Xeon 6973P-C under matching software/thread controls. Hardware-dependent numerical paths remain a strong candidate, not an isolated sole cause.

## Current open work

1. **Move beyond rank-based mediators.** Test training-only function-space diagnostics such as cross-environment probe transfer, class-conditional margin structure, or predictive disagreement.
2. Run a matched-coverage / different-reusability experiment to isolate what happens after gradient-subspace coverage.
3. Run structured-vs-unstructured matched-novelty tests with comparable selected-gradient non-redundancy.
4. Repeat the frozen Q-scaling contrast on a materially different task/architecture; a third same-task replication is lower value.
5. Derive an actionable early-trajectory function-space predictor without calibrating on final held-out results.
6. Prospectively challenge the existing raw-rank marker on a materially different modality while keeping its fixed rule unchanged.
7. Develop a genuinely new training-only tail-safety theory without recycling the failed rank-SD rule.
8. Use pinned hardware for bitwise studies and run GPU-vectorized wall-clock comparisons at matched budgets.

## Public claim boundary

Safe public wording:

> Gradient-aware selection of stochastic training environments can improve held-out performance under some structured stochastic shifts. Two preregistered n=30 Q-scaling experiments on fresh blocks support a finite-budget subset-coverage explanation: the gradient-novelty advantage is large when only a small subset of candidates can contribute to each update, attenuates as Q approaches K, and becomes exactly zero when all K candidates are used. Raw hidden representation effective rank has a strong fixed-parameterization condition-average predictive record, but function-preserving intervention shows it is coordinate-dependent and not itself causal. Channel-standardized effective rank is scale-invariant but failed as the quantitative mediator in a fresh preregistered test.

Do not claim a universally good seed family, universal selector/controller, causal representation-rank law, calibrated per-run gate, confirmed CIFAR tail robustness, general large-Transformer validity, bitwise cross-hardware hosted-CPU reproducibility, or GPU efficiency advantage.
