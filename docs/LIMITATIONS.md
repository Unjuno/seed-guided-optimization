# Limitations and open questions

## Current limitations

1. **Scale.** Most experiments use small MLPs and scikit-learn datasets.
2. **Architecture.** CNN replication is required before claiming architecture robustness.
3. **Optimizer.** Most reported experiments use AdamW; optimizer dependence requires controlled ablation.
4. **Environment construction.** Several benchmarks use synthetic stochastic transformations; real-world domain shifts remain untested.
5. **Gradient proxy.** Final-layer gradient signatures are empirically useful here, but their adequacy is not theoretically guaranteed.
6. **RNG fingerprint knowledge.** The strongest RNG prefilter assumes knowledge of which RNG coordinates affect the environment generator. Automatic discovery is open.
7. **Hardware.** Wall-clock optima depend on vectorization and hardware utilization; CPU results do not determine GPU-optimal seed counts.
8. **Meta-policy transfer.** Selector weights learned on one or two tasks did not transfer universally.
9. **Effect size.** Some statistically reproducible effects are small in absolute accuracy and may not justify selector overhead on every workload.

## Claims to avoid

Do not claim that seed integers themselves form meaningful semantic clusters; that more seeds always improve training; that gradient novelty alone is universally optimal; that the current selector is globally optimal; that the method prevents ordinary dataset overfitting; or that results already generalize to CNNs, Transformers, RL, or large-scale GPU training without further tests.

## Most important remaining experiments

1. CNN replication under the same geometric environment shift.
2. AdamW vs. SGD-family optimizer interaction.
3. Larger real-image dataset replication.
4. GPU wall-clock study with vectorized environment batches.
5. Learned RNG fingerprint/projection when relevant RNG coordinates are unknown.
