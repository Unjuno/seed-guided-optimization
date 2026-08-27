# Results

This directory contains committed CSV snapshots used to support the public research narrative. It is an evidence archive, not a claim that every exploratory run is confirmatory.

## Conventions

Accuracy-like metrics are stored on the `[0, 1]` scale. A difference of `0.01` equals **1 percentage point**.

`rep` identifies a paired replicate. Within paired comparisons, methods with the same replicate ID share the intended nuisance randomness and held-out environment pool.

Common metrics:

- `mean_test` — mean accuracy across held-out environments;
- `sd_test` — standard deviation across held-out environments;
- `p10_test` — 10th-percentile held-out accuracy;
- `min_test` — minimum held-out environment accuracy;
- `clean_test` — clean-data accuracy.

Paired summary files may contain raw p-values and Holm-adjusted p-values. A raw p-value below `.05` is not treated as confirmatory when the stated five-metric correction family is not significant.

## 1. Core selector evidence

- `public_key_findings.csv` — compact historical headline table.
- `geometric_paired.csv` — main structured geometric selector comparisons.
- `gradient_vs_parameter_paired.csv` — gradient novelty vs physical transformation-parameter novelty.
- `cnn_replication_paired20.csv` — 20-pair small-CNN replication.
- `optimizer_ablation_paired20.csv` — tuned AdamW / SGD+momentum selector comparisons.
- `sgd_lr_sweep_summary.csv` — independent loss-hard-only SGD tuning evidence.

## 2. Mechanism audits

- `gradient_direction_audit_paired.csv` — paired mean/tail gradient-alignment comparisons.
- `gradient_direction_audit_stage_summary.csv` — stage-wise gradient geometry summary.
- `gradient_one_step_loss_paired.csv` — paired realized one-step loss-reduction comparisons.
- `gradient_one_step_loss_summary.csv` — method means for the one-step audit.

These files are especially important because they contain negative controls: gradient novelty is not simply the best mean-gradient estimator or the selector with the largest immediate loss decrease.

## 3. RNG compression and learned relevance

- `rng_compression_summary40.csv` — candidate-compression trade-off.
- `learned_rng_calibration_size_sweep.csv` — calibration-size sensitivity.
- `learned_rng_calibration_stability_summary.csv` — relevance-ranking stability.
- `learned_rng_cross_generator_calibration.csv` — shifted-coordinate generator calibration.
- `learned_rng_cross_generator_summary20.csv` — cross-generator method summary.
- `learned_rng_cross_generator_weighting_paired20.csv` — soft relevance-weighting paired tests.

The main interpretation is not that seed integers have semantic classes. The evidence supports **generator/model-conditioned relevance** over finite RNG fingerprints.

## 4. Adaptive / relative novelty control

- `adaptive_beta_summary20.csv` — absolute feedback-controller method summary.
- `adaptive_beta_key_paired20.csv` — key paired controller comparisons.
- `relative_redundancy_digits_summary20.csv` — relative-controller Digits summary.
- `relative_redundancy_digits_paired20.csv` — relative-controller Digits paired tests.
- `relative_redundancy_synthetic_summary20.csv` — independent Synthetic summary.
- `relative_redundancy_synthetic_paired20.csv` — independent Synthetic paired tests.

The absolute controller is retained because its Synthetic saturation is a useful failure case. The relative controller normalizes the operating point within the feasible per-step gradient-redundancy range.

## 5. External validation

- `cifar_resnet_primary20_method_means.csv` — first 20 CIFAR-10 / ResNet-20 primary method means.
- `cifar_resnet_primary20_paired.csv` — first 20 paired primary tests.

These CIFAR files are **suggestive, not confirmatory**: mean and p10 are positive with raw `p<.05`, but neither survives the stated five-metric Holm correction. Independent reps 20-39 are being run in PR #11.

## 6. Efficiency

- `wallclock_summary.csv` — vectorized CPU seed-count / wall-clock summary.

Wall-clock results are hardware-specific. Do not infer GPU performance from this CSV.

## How to use this directory

For a result quoted publicly:

1. identify the reproduction script in [`../experiments/README.md`](../experiments/README.md);
2. read the method/metric definition in [`../docs/METHODS.md`](../docs/METHODS.md);
3. inspect the relevant paired CSV here;
4. check [`../docs/RESEARCH_STATUS.md`](../docs/RESEARCH_STATUS.md) to see whether the result is classified as supported, suggestive, negative/null, or in progress;
5. read [`../docs/LIMITATIONS.md`](../docs/LIMITATIONS.md) before generalizing beyond the tested regime.

Full replicate-level outputs are not committed for every exploratory experiment. Committed paired/statistical summaries are the stable public evidence snapshots; reproduction scripts regenerate the underlying runs where provided.
