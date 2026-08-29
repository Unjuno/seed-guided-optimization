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

- `trajectory_mechanism_cross_task_summary20.csv` — four-task aggregate showing that accumulated gradient effective rank is not itself a success criterion and motivating representation conversion as the next mechanism candidate.

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

## 5. CIFAR-10 / ResNet-20 primary benefit

- `cifar_resnet_primary_all40.csv` — replicate-level combined 40-pair output.
- `cifar_resnet_primary_paired40.csv` — paired tests with the five-metric Holm family.
- `cifar_resnet_primary_summary40.csv` — method-level means.

The 40-pair held-out mean gain is +0.1206 pp with Holm(5) p=0.013361. p10/minimum are positive but not corrected-significant.

## 6. Prospective representation-rank validation

Original six-test sequence:

- `prospective_rep_rank_predictions6.csv` — frozen predictions and observed directions.
- `prospective_rep_rank_summary6.csv` — compact diagnostic/outcome aggregate.

FashionMNIST / Tiny Transformer:

- `fashion_transformer_rep_rank_all10.csv`
- `fashion_transformer_rep_rank_deltas10.csv`
- `fashion_transformer_rep_rank_decision10.csv`
- `fashion_transformer_rep_rank_extension_all20.csv`
- `fashion_transformer_rep_rank_extension_deltas20.csv`
- `fashion_transformer_rep_rank_extension_decision20.csv`
- `fashion_transformer_rep_rank_combined_decision30.csv`

The initial Fashion test was PASS. The independent 20-pair extension again produced positive condition-average rank change (+0.07853) and positive held-out mean benefit (+0.4377 pp), so the frozen relation **REPLICATES**. The 30-pair combined estimate is precision-only.

CIFAR/ResNet prospective test:

- `cifar_resnet_rep_rank_all10.csv`
- `cifar_resnet_rep_rank_deltas10.csv`
- `cifar_resnet_rep_rank_decision10.csv`

The independent 10-pair CIFAR audit produced mean delta representation rank +0.03205 and held-out mean benefit +0.1703 pp; the preregistered directional decision was **PASS**.

Across the registered condition-level tests, the frozen sign rule is now 8/8 across six datasets. Three tests share Digits; this is a condition-average predictor record, not a universal or per-run gate.

## 7. Hosted-CPU reproducibility audit

- `cifar_cpu_repro_decision.csv` — frozen preregistered decision.
- `cifar_cpu_repro_directions.csv` — secondary aggregate direction stability.
- `cifar_cpu_repro_comparison.csv` — field-level A/B equality and drift.
- `cifar_cpu_repro_runtime.csv` — CPU model and pinned software/thread provenance.

Decision: **DRIFT PERSISTS** under single-thread hosted CPU. Max representation-rank drift is 0.427255 and max accuracy-metric drift is 0.014667, while aggregate rank and mean directions remain positive in both repeats.

Do not combine scientific rows from separate hosted-runner executions.

## 8. Efficiency

- `wallclock_summary.csv`

Do not infer GPU performance from CPU evidence.

## Reading order

For any public claim: locate the script in `../experiments/`, read the metric/protocol in `../docs/`, inspect the paired/summary CSV here, then check `../docs/RESEARCH_STATUS.md` and `../docs/LIMITATIONS.md`.
