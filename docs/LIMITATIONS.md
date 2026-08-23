# Limitations and open questions

## Current limitations

1. **Scale.** The strongest replicated results still use scikit-learn Digits and small networks. One small CNN replication is not evidence for modern large-scale architectures.
2. **Architecture range.** The effect replicated from an MLP to a small CNN, but Transformers, residual networks, RL policies, and larger vision models are untested.
3. **Optimizer range.** The effect survived tuned AdamW and tuned SGD+momentum in the MLP benchmark, but broader optimizer families and schedules are untested.
4. **Environment construction.** Several benchmarks use synthetic stochastic transformations; naturally occurring real-world domain shifts remain untested.
5. **Gradient proxy.** Final-layer gradient signatures are empirically useful here, but their adequacy is not theoretically guaranteed for deeper/larger models.
6. **RNG fingerprint transfer.** A learned fingerprint can recover generator-relevant RNG coordinates in the current proof of concept, but the best representation is generator-dependent. Cross-generator transfer and automatic metric selection remain open.
7. **Hardware.** Wall-clock optima depend on vectorization and hardware utilization; CPU results do not determine GPU-optimal seed counts.
8. **Meta-policy transfer.** Selector weights learned on one or two tasks did not transfer universally.
9. **Effect size / overhead.** Some statistically reproducible effects are modest and may not justify candidate forward/gradient overhead on every workload.
10. **Hyperparameter fairness.** The initial SGD false negative demonstrated that optimizer tuning must be separated from selector evaluation; selector comparisons can otherwise inherit optimizer mis-tuning.

## Claims to avoid

Do not claim that seed integers themselves form meaningful semantic clusters; that more seeds always improve training; that gradient novelty alone is universally optimal; that the current selector is globally optimal; that the method prevents ordinary dataset overfitting; or that results already generalize to Transformers, RL, large real-image benchmarks, or large-scale GPU training.

## Most important remaining experiments

1. Larger real-image dataset replication with a CNN/ResNet-family model.
2. GPU wall-clock study with vectorized environment batches.
3. Cross-generator validation of learned RNG-to-gradient representations and automatic representation/metric choice.
4. More optimizer/schedule families and stronger architecture scaling.
5. A benchmark where stochasticity comes from a real simulator or naturally varying domain rather than a hand-built corruption generator.
