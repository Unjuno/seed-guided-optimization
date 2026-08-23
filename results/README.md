# Results

Committed CSVs are snapshots of the experiments used in the public README.

## Conventions

Accuracy-like metrics are stored on the `[0, 1]` scale. A difference of `0.01` equals **1 percentage point**.

`rep` identifies a paired replicate. Paired comparisons use the same replicate number across methods.

Core metrics:

- `mean_test`: mean accuracy across held-out environments;
- `sd_test`: standard deviation across held-out environments;
- `p10_test`: 10th percentile held-out accuracy;
- `min_test`: minimum held-out environment accuracy;
- `clean_test`: clean-data accuracy.

Paired summary files contain paired differences, unadjusted p-values, and Holm-adjusted p-values when multiple metrics are treated as one family.

## Key files

- `public_key_findings.csv` — compact headline table.
- `cnn_replication_all20.csv` — full CNN replicate-level results.
- `cnn_replication_paired20.csv` — CNN paired tests.
- `optimizer_ablation_all20.csv` — full AdamW / SGD+momentum selector results.
- `optimizer_ablation_paired20.csv` — optimizer paired tests.
- `sgd_lr_sweep_low.csv`, `sgd_lr_sweep_high.csv` — independent loss-hard SGD tuning.
- `rng_compression_summary40.csv` — RNG prefilter compression sweep.
- `geometric_paired.csv` — main geometric selector comparisons.
- `gradient_vs_parameter_paired.csv` — gradient novelty vs transformation-parameter novelty.
- `wallclock_summary.csv` — vectorized seed-count wall-clock summary.

Do not infer universality from a single CSV. See `docs/LIMITATIONS.md` and `docs/RESULTS.md`.
