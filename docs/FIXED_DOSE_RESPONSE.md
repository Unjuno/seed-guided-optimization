# Fixed-dose functional-response audit — Issue #61

## Frozen result and evidence strength

**DOSE-DEPENDENT BENEFIT PASS**, under the preregistered conjunction of two positive one-sided tests. **The interaction is borderline, not strong mechanism evidence.**

The first complete valid Actions run was [33990885955](https://github.com/Unjuno/seed-guided-optimization/actions/runs/33990885955), scientific head `26b73e3ee35489824b82e745dcd924674789c84f`, implementation PR #62. All 30 paired replicates 1200–1229 completed without an outcome-dependent extension or stopping rule.

| Estimand | Mean benefit (pp) | Paired SE (pp) | Two-sided t29 95% CI (pp) | One-sided p | Positive pairs |
|---|---:|---:|---:|---:|---:|
| Full-strength geometric benefit | 2.84223 | 0.29673 | [2.23536, 3.44910] | 8.68266e-11 | 28/30 |
| Clean benefit, secondary | 1.22097 | 0.94936 | [-0.72068, 3.16263] | 0.104290 | 23/30 |
| Full-strength minus clean benefit | 1.62125 | 0.94811 | [-0.31784, 3.56035] | 0.0489740 | 19/30 |

The full-strength effect is supported in this block. The full-minus-clean interaction narrowly passes the fixed one-sided 0.05 rule; its two-sided p is 0.0979481 and its two-sided 95% interval includes zero. These statements are compatible: the one-sided test and two-sided interval use different tail allocations. Do not silently replace the registered rule, and do not describe the interaction as conclusive.

## What was fixed before execution

Preregistration: [Issue #61](https://github.com/Unjuno/seed-guided-optimization/issues/61). No evaluation-strength calibration or selection was performed.

Training used the existing `transfer_specificity.train_one` and `schedule` functions: Digits MLP 64–128–ReLU–10; AdamW learning rate 0.01, weight decay 0.001; 10 epochs; batch size 128; 16 candidate environments and 4 selected per update; gradient-novelty weight 0.6. Training environment seeds were 41000–41063. Paired methods shared initialization and candidate/minibatch schedules, not the resulting parameter trajectory.

Each shard trained all 10 states before constructing evaluation environments. Identical frozen model states were evaluated across all conditions. Geometric-image interpolation weights were fixed at 0, 0.01, 0.10, 0.25, 0.50 and 1.0; geometric seeds were 42000–42079. Zero is clean, evaluated once rather than counted as 80 independent repetitions; one is the original full geometric transformation. Intermediate weights mix the clean and transformed images; they do not scale the generator's rotation angle or every latent factor by that weight.

Secondary permutation-mixture controls used weights 0.005, 0.05 and 0.10 and seeds 43000–43079. They were NOT difficulty matched and cannot establish transfer specificity.

### Variables, units and decision definitions

| Symbol | Meaning | SI unit | Definition | Domain / assumptions | Type |
|---|---|---|---|---|---|
| r | Paired training replicate | 1 | Index of initialization/schedule pair | Integers 1200–1229 | Scalar index |
| m | Selection method | Not applicable | loss_hard or gradnov | Exactly two registered methods | Categorical index |
| lambda | Geometric-image mixture weight | 1 | Weight of transformed image in the convex mixture | Frozen six-point grid above | Real scalar |
| A(r,m,lambda) | Mean evaluation accuracy | 1 | Correct-prediction fraction averaged over the fixed environment sample | [0,1]; fixed 445 test images | Real scalar |
| B(r,lambda) | Selection benefit | 1 | A(r,gradnov,lambda) minus A(r,loss_hard,lambda) | [-1,1] | Real scalar |
| I(r) | Full-minus-clean interaction | 1 | B(r,1) minus B(r,0) | [-2,2] | Real scalar |
| p_full, p_interaction | One-sided test results | 1 | One-sample t-test against zero for paired B(r,1) and I(r) | [0,1]; 30 training pairs, not environments as replicates | Probability-valued scalars |
| k | Coverage multiplier | 1 | 97.5th percentile of t with 29 degrees of freedom | 2.0452296421 | Positive real scalar |

FULL passes when mean B(r,1) is positive and p_full is below 0.05. INTERACTION passes when mean I(r) is positive and p_interaction is below 0.05. Both are required for the registered PASS. Intermediate doses, nuisance outcomes, loss or disagreement cannot upgrade a failure.

Unit check: accuracies, their paired differences, SEs and interval endpoints are all dimensionless. CSVs store accuracy fractions; multiplying a difference by 100 converts it to percentage points, not a relative percentage improvement. For example, 0.0284222849 becomes 2.84222849 pp.

## Fixed dose profile and function-space diagnostics

| Geometric image weight | Mean gradnov minus loss-hard (pp) | Mean prediction disagreement from each model's clean prediction (%) |
|---:|---:|---:|
| 0 | 1.2210 | 0.0000 |
| 0.01 | 1.2374 | 0.3520 |
| 0.10 | 1.1255 | 3.8136 |
| 0.25 | 1.1054 | 10.6996 |
| 0.50 | 1.4479 | 25.7552 |
| 1.00 | 2.8422 | 60.1808 |

These intermediate-dose means and disagreement diagnostics are descriptive. The benefit curve is not monotonically increasing. Disagreement is averaged over both methods and all 30 pairs; it is disagreement with the same model's clean prediction, not disagreement between the two methods.

The weak mixture changes predictions very little on this new block, whereas the full transformation changes them substantially. This supports an interpretation that the previous near-clean comparison was a weak manipulation. It does not retrospectively turn the previous negative result into a PASS or identify an internal mediator.

## What this does not establish

The preceding dual-evaluation experiment #59 remains **NO SHARED REPLICATION**: shared benefit +0.78165 pp, one-sided p0.0997763; shared-minus-nuisance contrast +0.00674 pp, p0.425828. Its condition was a 1% geometric-image mixture versus a 0.5% permutation mixture. See [DUAL_TRANSFER_RESULT.md](DUAL_TRANSFER_RESULT.md).

This audit tests functional response at fixed strengths, not difficulty-matched factor reusability. Baseline difficulty changes with strength. Margin redistribution, nuisance suppression, general clean improvement and other changes remain competing explanations. Checkpoint identity controls the trained model across evaluation conditions, not these causal alternatives. Raw or standardized representation rank has not been reinstated as a causal mediator.

Fresh replicates and environment seeds are not fresh images. The fixed Digits split contains 988 training and 445 test examples previously used in the project. Earlier loss-hard-only evaluation calibration used those test-image labels; it was not training-only calibration. All uncertainty intervals here are conditional on the fixed image split and environment bank. Paired SE is not a combined uncertainty estimate over datasets, environment populations and hardware; no such combined estimate is available.

## Runtime, reproducibility and verification

Actions used Python 3.12.14, PyTorch 2.10.0+cpu, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 and scikit-learn 1.8.0. PyTorch intra-op, OMP, MKL and OpenBLAS were set to one thread; deterministic algorithms and the mkldnn backend were enabled. CPU families were AMD EPYC 9V74, AMD EPYC 7763 and Intel Xeon Platinum 8573C. See [runtime rows](../results/fixed_dose_runtime.csv). MHz entries are instantaneous metadata, not controlled clocks. No throughput or GPU-efficiency claim is made, and cross-hardware bitwise reproducibility is not established.

The independent archive audit verified seven ZIP hashes, all 60 checkpoint hashes, archived source hashes, the complete 540-row condition grid, and 38,460 environment-level rows. Reconstructing means/NLL/disagreement from the environment rows differed from the aggregate by at most 6.661338147750939e-16. Means, paired SEs, t29 intervals, one- and two-sided p-values were separately recalculated from the 30 paired rows. Re-running the frozen summarizer also reproduced the decision.

Local verification used Python 3.13.5 with the same scientific package versions and did not retrain any model or select an alternative run. The released evidence bundle contains the original aggregate and six shard ZIPs, all 60 states, the archived training sources, independent verification scripts and full metrics. Repository CSVs include the paired primary endpoints, complete dose summary, runtime metadata and independent statistical check.

## Artifact manifest

All IDs refer to run 33990885955. Original artifacts have a 90-day retention period; archive them for long-term preservation.

| Artifact | ID | SHA256 |
|---|---:|---|
| fixed-dose-aggregate30 | 9976615757 | 40cbc59d29c9d92fd23505edb1bf09d17354c91013a4b70f8cdebfa7f8b97b09 |
| fixed-dose-shard-1200 | 9976600280 | 3c98d1259731f29c511bad82cc09c9b4ceb5e1e1f5dffb7fbf6ac14b0d433737 |
| fixed-dose-shard-1205 | 9976601646 | e6fb8df6e7561e8da80204f5ee3cd797d67e92a4807236dcd7543f9e866ad9f8 |
| fixed-dose-shard-1210 | 9976601101 | 7d9f422849955fac6ce6508bd516f8efc369fa3c77cd2d0fa5c3251dbec7638d |
| fixed-dose-shard-1215 | 9976600633 | 5266dc696e2973ac22b7aa953a26d0649d081f99c82223af013ba9fe9866ecb4 |
| fixed-dose-shard-1220 | 9976603875 | 5bc71c255b174f82567c55281e0f5d787512afc3945f31acee6e0a478be45dc9 |
| fixed-dose-shard-1225 | 9976597473 | a0f518b2cef6e3ea31b7aa4f2d89ce1195c0711e7718ccb4ca0cbd9275ed456e |

## Next falsification target

A new independent replication should challenge the borderline full-minus-clean interaction without selecting strengths or extending this sample. The stronger mechanistic goal still requires an intervention that separates reusable task structure from difficulty and other functional changes. This audit alone does not provide that intervention.
