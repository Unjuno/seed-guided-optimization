# Limitations and open questions

## Current limitations

1. **Scale and effect size.** CIFAR-10 / ResNet-20 shows a corrected-significant mean effect at 40 pairs, but the effect is small (+0.1206 pp) and tail metrics do not survive the same Holm family.
2. **Architecture range.** MLP, small CNN, ResNet-20, and one small 2-layer patch Transformer have been tested. FashionMNIST/Tiny Transformer replicated in an independent 20-pair extension, and the CIFAR/ResNet prospective rank audit passed, but this is not evidence for large vision Transformers, language Transformers, RL policies, or substantially deeper architectures.
3. **Optimizer range.** The main MLP effect survived tuned AdamW and tuned SGD+momentum. Broader optimizer/schedule/large-batch regimes remain untested.
4. **Environment construction.** Several strong benchmarks use synthetic stochastic transformations. Naturally occurring domain shifts and real stochastic simulators are largely untested.
5. **Gradient proxy.** Final-layer gradient signatures are empirically useful and cheaper than full gradients in the tested MLP, but there is no theorem that they preserve relevant geometry in larger models.
6. **Mechanism is not causal yet.** Raw gradient-rank expansion fails as an explanatory criterion. Representation effective-rank direction is a stronger prospective predictor, but effective rank may be a marker rather than the causal mediator.
7. **Prospective dependence and scale.** Eight registered condition-level representation-rank direction tests have matched across six datasets, but three share Digits. The evidence supports a condition-average relation, not a universal calibrated gate.
8. **Per-run gating is unsupported.** Fashion and CIFAR both contain individual replicate-level sign mismatches despite aggregate direction matches.
9. **Tail gating is unsolved.** The attempted per-environment representation-rank-SD tail rule failed on Wine and is retired. There is no reliable training-only p10/worst predictor.
10. **Relative control is not task safety.** Cross-task normalization helps control redundancy strength but does not guarantee that novelty should be used; Breast-high remains a counterexample.
11. **RNG relevance scope.** Learned RNG relevance has been demonstrated on handcrafted generators, not arbitrary augmentation stacks, simulators, PRNG layouts, or model families.
12. **Compression ratio.** A useful RNG fingerprint does not make aggressive prefiltering free; fingerprint quality and candidate compression are separate optimization problems.
13. **Execution reproducibility.** A preregistered single-thread hosted-CPU audit still produced large cross-run numerical drift. Rep45 was bitwise identical across two AMD EPYC models, while rep46 drifted between AMD EPYC and Intel Xeon under the same pinned software/thread settings. Hardware-dependent numerical paths are plausible but not isolated as the sole cause.
14. **Hardware performance.** CPU wall-clock results do not determine GPU-optimal candidate counts or production efficiency.
15. **Meta-policy transfer.** Selector coefficients learned on one/few tasks did not transfer universally.
16. **Hyperparameter fairness.** Optimizer/controller/candidate budgets must be tuned independently of final held-out selector evaluation.
17. **Exploratory volume.** Public claims must be anchored to pre-specified paired comparisons and correction families rather than selected raw p-values.

## Claims to avoid

Do not claim that:

- seed integers form semantic clusters;
- a universally good seed family exists independently of model/data/generator/state;
- more seeds always improve learning;
- gradient diversity alone is universally optimal;
- a fixed relative-redundancy parameter is universal;
- learned RNG relevance is generator-independent;
- representation effective rank is proven causal;
- the representation-rank sign rule is a universal/calibrated or per-run gate;
- the retired tail rule works;
- CIFAR p10/worst robustness is confirmed;
- the method prevents ordinary dataset overfitting;
- hosted-CPU execution is bitwise reproducible across heterogeneous hardware;
- current CPU results imply a GPU efficiency advantage;
- the Tiny Transformer results establish general Transformer or large-model validity.

## Current theoretical uncertainty

The best working explanation is finite-budget coverage of unresolved task-relevant gradient directions followed by conversion into reusable representation structure. This explains the strongest positive and negative observations, but several alternatives remain possible:

- effective rank may proxy another representation property such as factor separation, margin geometry, or feature reuse;
- gradient-novel selection may alter implicit regularization through a mechanism not captured by rank;
- the benefit may depend on interactions among architecture, optimizer, and environment generator rather than a single universal mediator;
- condition-average directional success may not extrapolate to much larger models or naturally stochastic processes.

See [`THEORETICAL_FRAMEWORK.md`](THEORETICAL_FRAMEWORK.md).

## Highest-value next experiments

1. **Mechanism mediation:** measure gradient coverage, representation geometry, and held-out benefit under interventions that can break the proposed causal chain.
2. **Representation intervention:** alter rank/geometry without SGO to test whether the diagnostic is causal or merely predictive.
3. **Matched structured-vs-unstructured novelty:** equalize gradient novelty while changing latent-factor reuse; the theory predicts benefit mainly for reusable structured novelty.
4. **Budget scaling:** test whether SGO advantage contracts when ordinary sampling receives enough compute to cover the relevant gradient subspace.
5. **Larger/different architecture prospective test:** keep the condition-average direction rule frozen before held-out evaluation.
6. **Early-trajectory diagnostic:** derive a practical signal available before fully training both methods; freeze it before new-task validation.
7. **New tail-safety theory:** derive a genuinely different hypothesis rather than retuning the failed rank-SD rule.
8. **Pinned-hardware reproducibility:** separate same-hardware determinism from cross-CPU numerical path differences.
9. **GPU-vectorized wall-clock benchmark:** report update budget, candidate-forward/backward budget, and actual wall-clock separately.
10. **Naturally stochastic benchmark:** simulator or process where seeds control non-handcrafted stochastic dynamics.
