# Limitations and open questions

## Current limitations

1. **Scale.** The strongest replicated results still use scikit-learn Digits and small networks. One small CNN replication is not evidence for modern large-scale architectures.
2. **Architecture range.** The effect replicated from an MLP to a small CNN, but Transformers, residual networks beyond the current pilot work, RL policies, and larger vision models remain insufficiently tested.
3. **Optimizer range.** The effect survived tuned AdamW and tuned SGD+momentum in the MLP benchmark, but broader optimizer families and schedules are untested.
4. **Environment construction.** Several benchmarks use synthetic stochastic transformations; naturally occurring real-world domain shifts remain untested.
5. **Gradient proxy.** Final-layer gradient signatures are empirically useful here, but their adequacy is not theoretically guaranteed for deeper/larger models.
6. **Automatic RNG fingerprint discovery.** A first training-only ridge relevance experiment recovered an oracle-like prefilter on the existing geometric generator without being told the seven relevant coordinates. This is still same-generator evidence: transfer to different augmentation stacks, simulators, PRNG layouts, and larger models is untested.
7. **Compression ratio.** Learning a useful RNG fingerprint does not make aggressive prefiltering free. At 16 → 8, the learned fingerprint matched the oracle representation but both lost lower-tail performance relative to evaluating all 16 gradient candidates. Fingerprint quality and compression level must be tuned separately.
8. **Hardware.** Wall-clock optima depend on vectorization and hardware utilization; CPU results do not determine GPU-optimal seed counts.
9. **Meta-policy transfer.** Selector weights learned on one or two tasks did not transfer universally.
10. **Effect size / overhead.** Some statistically reproducible effects are modest and may not justify candidate forward/gradient overhead on every workload.
11. **Hyperparameter fairness.** The initial SGD false negative demonstrated that optimizer tuning must be separated from selector evaluation; selector comparisons can otherwise inherit optimizer mis-tuning.

## Claims to avoid

Do not claim that seed integers themselves form meaningful semantic clusters; that more seeds always improve training; that gradient novelty alone is universally optimal; that the current selector is globally optimal; that the method prevents ordinary dataset overfitting; that the learned RNG relevance rule is generator-independent; or that results already generalize to Transformers, RL, large real-image benchmarks, or large-scale GPU training.

## Most important remaining experiments

1. Larger real-image dataset replication with a ResNet-family model.
2. GPU wall-clock study with vectorized environment batches.
3. Cross-generator validation of learned RNG relevance on an augmentation stack whose relevant random coordinates differ from the current geometric generator.
4. More optimizer/schedule families and stronger architecture scaling.
5. A benchmark where stochasticity comes from a real simulator or naturally varying domain rather than a hand-built corruption generator.
