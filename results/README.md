# Results

Committed CSV snapshots supporting the public research narrative. This is an evidence archive, not a claim that every exploratory run is confirmatory.

## Conventions

Accuracy-like metrics are on `[0,1]`; `0.01` equals one percentage point. `rep` identifies a paired replicate. Raw p-values are not promoted when the stated corrected family is not significant.

## 1. Core selector evidence

- `public_key_findings.csv`
- `geometric_paired.csv`
- `gradient_vs_parameter_paired.csv`
- `cnn_replication_paired20.csv`
- `optimizer_ablation_paired20.csv`
- `sgd_lr_sweep_summary.csv`

## 2. Mechanism audits

Existing one-step/direction controls:
- `gradient_direction_audit_paired.csv`
- `gradient_direction_audit_stage_summary.csv`
- `gradient_one_step_loss_paired.csv`
- `gradient_one_step_loss_summary.csv`

Trajectory-level addition:
- `trajectory_mechanism_cross_task_summary20.csv` — four-task aggregate showing that accumulated gradient effective rank is not itself a success criterion and motivating representation effective-rank change as the next candidate diagnostic.

## 3. RNG compression / learned relevance

- `rng_compression_summary40.csv`
- `learned_rng_calibration_size_sweep.csv`
- `learned_rng_calibration_stability_summary.csv`
- `learned_rng_cross_generator_calibration.csv`
- `learned_rng_cross_generator_summary20.csv`
- `learned_rng_cross_generator_weighting_paired20.csv`

## 4. Adaptive / relative control

- `adaptive_beta_summary20.csv`
- `adaptive_beta_key_paired20.csv`
- `relative_redundancy_digits_summary20.csv`
- `relative_redundancy_digits_paired20.csv`
- `relative_redundancy_synthetic_summary20.csv`
- `relative_redundancy_synthetic_paired20.csv`

## 5. CIFAR-10 / ResNet-20

- `cifar_resnet_primary_all40.csv` — replicate-level combined 40-pair output.
- `cifar_resnet_primary_paired40.csv` — paired tests with the five-metric Holm family.
- `cifar_resnet_primary_summary40.csv` — method-level means.

The 40-pair held-out mean gain is +0.1206 pp with Holm(5) p=0.013361. p10/minimum are positive but not corrected-significant.

## 6. Prospective representation-rank validation

- `prospective_rep_rank_predictions6.csv` — frozen predictions and observed directions for the six registered tests.
- `prospective_rep_rank_summary6.csv` — compact diagnostic/outcome aggregate.

The frozen sign rule matched all six registered directions. Three tests share Digits; do not treat 6/6 as six independent datasets. The attempted tail rule is retired.

## 7. Efficiency

- `wallclock_summary.csv`

Do not infer GPU performance from CPU evidence.

## Reading order

For any public claim: locate the script in `../experiments/`, read the metric/protocol in `../docs/`, inspect the paired/summary CSV here, then check `../docs/RESEARCH_STATUS.md` and `../docs/LIMITATIONS.md`.
