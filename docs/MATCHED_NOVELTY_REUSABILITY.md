# Matched-novelty structured-vs-unstructured conversion test

## Status

Preregistered in Issue #41 before calibration and confirmatory execution.

Frozen decision: **MATCH FAILURE / INCONCLUSIVE**.

The experiment therefore does **not** establish the planned reusable-structure mechanism claim, even though the descriptive held-out contrast was large.

## Design

The test compared two Digits/MLP environment families under the same optimizer and selector protocol (`K=16`, `Q=4`):

- **structured:** the existing coherent geometric environment family;
- **unstructured:** independent per-example pixel noise/dropout/impulse nuisance.

A training-only calibration block (reps 450-459) selected nuisance severity from the preregistered grid. Confirmatory reps 500-529 and held-out seeds 19000-19079 were fresh.

Calibration and confirmatory training diagnostics were sealed before confirmatory held-out environments were constructed.

## Training-only calibration

The frozen calibration score selected severity **1.10**. However even the best grid point remained far from the structured target:

- calibration structured novelty gain: `0.12108`;
- calibration unstructured novelty gain: `0.01529`;
- novelty mismatch: `0.10579`;
- structured candidate loss: `1.74301`;
- unstructured candidate loss: `1.46159`;
- candidate-loss mismatch: `0.28141`.

This already indicated that the independent per-example nuisance family was not capable of reproducing the structured environment's selector-level gradient non-redundancy under the frozen grid.

## Confirmatory match check

The confirmatory training-only block reproduced the mismatch:

- structured novelty gain: `0.11642`;
- unstructured novelty gain: `0.01634`;
- novelty mismatch: **0.10009** vs frozen tolerance `0.03`;
- structured candidate loss: `1.74449`;
- unstructured candidate loss: `1.46252`;
- candidate-loss mismatch: **0.28197** vs frozen tolerance `0.10`.

Therefore **MATCH PASS = false**.

## Held-out result, descriptive only

The held-out contrast cannot upgrade the frozen decision because matching failed.

Across 30 fresh paired replicates:

- structured gradnov-minus-loss-hard mean benefit: **+3.034 pp**;
- unstructured benefit: **+0.015 pp**;
- structured-minus-unstructured conversion contrast: **+3.018 pp**;
- conversion SE: **0.310 pp**;
- approximate 95% CI: **+2.384 to +3.653 pp**;
- one-sided t-test: `t=9.725`, `p=6.19e-11`;
- structured benefit was positive in **30/30** replicates.

This is a strong descriptive separation, but it is confounded by the failed training-only match and is therefore not confirmatory evidence for the reusability-conversion hypothesis.

## What was learned

The failure is mechanistically informative. Independent per-example pixel nuisance is averaged heavily within a minibatch and produces much smaller environment-level gradient-direction separation than coherent geometric transformations. Increasing nuisance severity raises loss but does not produce comparable gradnov-vs-loss-hard selected-gradient novelty.

Therefore the next matched-novelty design should not merely increase IID noise severity. It should use a **high-dimensional environment-specific nuisance pattern shared within an environment but unrelated across environment seeds**. Such a construction can create large environment-conditioned gradient differences without introducing a reusable low-dimensional geometric factor.

The next experiment must preregister its new nuisance family and calibration rule before any fresh held-out outcomes are observed.

## Evidence

- Issue #41: preregistration and frozen decision rules.
- PR #42: implementation and authoritative Actions execution.
- Actions run `33491983745`.
- `results/matched_novelty_calibration_summary.csv`.
- `results/matched_novelty_confirm_training30.csv`.
- `results/matched_novelty_conversion30.csv`.
- `results/matched_novelty_decision30.csv`.
