# Limitations and open questions

## Current limitations

1. **Scale and effect size.** CIFAR-10 / ResNet-20 shows a corrected-significant mean effect at 40 pairs, but the effect is small (+0.1206 pp) and tail metrics do not survive the same Holm family.
2. **Architecture range.** MLP, small CNN, ResNet-20, and one small 2-layer patch Transformer have now been tested. The FashionMNIST/Tiny Transformer prospective rank test passed, but this is not evidence for larger vision Transformers, language Transformers, RL policies, or substantially deeper architectures.
3. **Optimizer range.** The main MLP effect survived tuned AdamW and tuned SGD+momentum. Broader optimizer/schedule/large-batch regimes remain untested.
4. **Environment construction.** Several strong benchmarks use synthetic stochastic transformations. Naturally occurring domain shifts and real stochastic simulators are largely untested.
5. **Gradient proxy.** Final-layer gradient signatures are empirically useful and cheaper than full gradients in the tested MLP, but there is no theorem that they preserve relevant geometry in larger models.
6. **Mechanism is not closed.** Raw gradient-rank expansion fails as an explanatory criterion. Representation effective-rank change is a stronger candidate, but it remains a predictor hypothesis rather than a causal proof.
7. **Prospective dependence and scale.** Seven registered representation-rank direction tests have matched across five datasets, but three share Digits and the new Transformer is deliberately tiny. The record is stronger than the original six-test set but still far from a calibrated universal gate.
8. **Tail gating is unsolved.** The attempted per-environment representation-rank-SD tail rule failed on Wine and is retired. There is no reliable training-only p10/worst predictor yet. FashionMNIST tail metrics are descriptive only and do not revive that rule.
9. **Relative control is not task safety.** Cross-task normalization helps control redundancy strength but does not guarantee that novelty should be used; Breast-high is a counterexample.
10. **RNG relevance scope.** Learned RNG relevance has been demonstrated on two handcrafted generators, not arbitrary augmentation stacks, simulators, PRNG layouts, or model families.
11. **Compression ratio.** A useful RNG fingerprint does not make aggressive prefiltering free; fingerprint quality and candidate compression are separate optimization problems.
12. **Hardware.** CPU wall-clock results do not determine GPU-optimal candidate counts or production efficiency.
13. **Meta-policy transfer.** Selector coefficients learned on one/few tasks did not transfer universally.
14. **Hyperparameter fairness.** Optimizer/controller/candidate budgets must be tuned independently of final held-out selector evaluation.
15. **Exploratory volume.** Public claims must be anchored to pre-specified paired comparisons and correction families rather than selected raw p-values.

## Claims to avoid

Do not claim that:

- seed integers form semantic clusters;
- a universally good seed family exists independently of model/data/generator/state;
- more seeds always improve learning;
- gradient diversity alone is universally optimal;
- a fixed relative-redundancy parameter is universal;
- learned RNG relevance is generator-independent;
- the representation-rank sign rule is already a universal/calibrated gate;
- the retired tail rule works;
- CIFAR p10/worst robustness is confirmed;
- the method prevents ordinary dataset overfitting;
- current CPU results imply a GPU efficiency advantage;
- one tiny FashionMNIST Transformer establishes general Transformer or large-model validity.

## Highest-value next experiments

1. **Complete CIFAR/ResNet trajectory falsification:** apply the already frozen training-only rank rule under the independent ResNet protocol.
2. **Larger/different architecture prospective test:** move beyond the Tiny Transformer while keeping the rule frozen before held-out evaluation.
3. **New tail-safety diagnostic:** derive a different hypothesis before looking at held-out tails.
4. **GPU-vectorized wall-clock benchmark:** report update budget, candidate-forward/backward budget, and actual wall-clock separately.
5. **Naturally stochastic benchmark:** simulator or process where seeds control non-handcrafted stochastic dynamics.
