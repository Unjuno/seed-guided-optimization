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

## Key committed evidence

- `public_key_findings.csv` — compact headline table.
- `cnn_replication_paired20.csv` — 20-pair CNN replication tests.
- `optimizer_ablation_paired20.csv` — tuned AdamW / SGD+momentum selector tests.
- `sgd_lr_sweep_summary.csv` — independent loss-hard-only SGD learning-rate tuning summary.
- `rng_compression_summary40.csv` — 40-pair RNG prefilter compression sweep.
- `geometric_paired.csv` — main geometric active-selector paired tests.
- `gradient_vs_parameter_paired.csv` — gradient novelty vs transformation-parameter novelty controls.
- `wallclock_summary.csv` — vectorized seed-count wall-clock summary.

Full replicate-level outputs can be regenerated with the scripts in `experiments/`. The committed paired/statistical summaries are the evidence snapshots used for public claims.

Do not infer universality from a single CSV. See `docs/LIMITATIONS.md` and `docs/RESULTS.md`.
