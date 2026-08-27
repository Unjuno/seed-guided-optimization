# Limitations and open questions

## Current limitations

1. **Scale and effect size.** The strongest corrected effects still come from structured small-scale benchmarks. CIFAR-10 / ResNet-20 produced only a small positive signal in the first 20 paired primary runs, and that signal did not survive five-metric Holm correction. The 40-pair extension is still in progress.
2. **Architecture range.** The effect replicated from an MLP to a small CNN and has been tested on ResNet-20, but Transformers, larger vision models, RL policies, and substantially deeper architectures remain insufficiently tested.
3. **Optimizer range.** The effect survived tuned AdamW and tuned SGD+momentum in the main MLP benchmark. Broader optimizer families, schedules, and large-batch regimes are untested.
4. **Environment construction.** Several strong benchmarks use synthetic stochastic transformations. Naturally occurring domain shifts and real stochastic simulators remain largely untested.
5. **Gradient proxy.** Final-layer gradient signatures are empirically useful and cheaper than full gradients in the tested MLP, but there is no theorem guaranteeing that they preserve the relevant geometry in larger models.
6. **Mechanism.** Direct audits show that the final gains are not explained simply by better mean-gradient estimation or larger one-step loss reduction. The trajectory-level mechanism—representation geometry, margins, accumulated gradient covariance, basin selection, or related effects—remains unresolved.
7. **When to use novelty.** Relative redundancy control can normalize novelty strength across tasks, but it does not yet solve the higher-level gating problem of deciding when novelty should be used at all. This is tracked in Issue #12.
8. **RNG relevance scope.** Learned RNG relevance has been demonstrated on two handcrafted generators, including a changed coordinate layout. This is evidence against requiring explicit coordinate labels, but not evidence that the same relevance-learning method transfers to arbitrary augmentation stacks, simulators, PRNG layouts, or model families.
9. **Compression ratio.** Learning a useful RNG fingerprint does not make aggressive prefiltering free. Fingerprint quality and candidate compression must be treated as separate optimization problems.
10. **Hardware.** Wall-clock optima depend on vectorization and hardware utilization. Current CPU results do not determine GPU-optimal candidate counts or whether selector overhead is worthwhile in production.
11. **Meta-policy transfer.** Selector coefficients learned on one or two tasks did not transfer universally. Cross-task robustness is therefore a separate problem from within-task selection quality.
12. **Hyperparameter fairness.** The initial SGD false negative showed that optimizer tuning must be separated from selector comparison. The same rule applies to controller thresholds, candidate budgets, and environment-generation parameters.
13. **Multiple outcomes and exploratory volume.** The project contains many exploratory experiments. Public claims should be anchored to explicitly paired comparisons and their reported correction family rather than selected raw p-values.

## Claims to avoid

Do not claim that:

- seed integers themselves form meaningful semantic clusters;
- a universally good seed family exists independently of model/data/generator/state;
- more seeds always improve learning;
- gradient novelty alone is universally optimal;
- the current relative redundancy setting is a universal constant;
- learned RNG relevance is generator-independent;
- the method prevents ordinary dataset overfitting;
- a positive raw p-value is confirmatory when the pre-specified corrected family is not significant;
- current CPU results imply a GPU efficiency advantage;
- the method is already validated for Transformers, RL, large modern vision models, or production-scale training.

## Most important remaining experiments

1. **Finish CIFAR-10 / ResNet-20 at 40 paired replicates** under the unchanged primary protocol (Issue #1 / PR #11).
2. **GPU-vectorized wall-clock benchmark** at matched wall-clock and candidate-evaluation budgets (Issue #2).
3. **Trajectory-level mechanism/gating study** across tasks where novelty helps and where it is neutral/harmful (Issue #12).
4. **Naturally stochastic or simulator-based benchmark** where environments are not hand-built corruption transforms.
5. **Architecture/optimizer scaling** after the mechanism and wall-clock questions are better constrained.

## Resolved roadmap item

The original requirement to learn a compact RNG fingerprint without being told the relevant generator coordinates has been completed as a proof-of-concept (closed Issue #3). The remaining question is **generality**, not whether such training-only relevance learning is possible at all.
