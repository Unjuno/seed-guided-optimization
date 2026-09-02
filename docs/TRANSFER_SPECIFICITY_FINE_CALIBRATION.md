# Fine-calibrated identical-training transfer-specificity test

## Status

Frozen result: **CALIBRATION DIFFICULTY MATCH FAILURE**.

Issue #56 moved the loss-hard-only nuisance calibration grid below alpha=0.10 while preserving the identical-training confirmatory design and all statistical thresholds.

## Result

Fresh calibration reps 950-959 produced:

- shared-factor geometric loss-hard mean accuracy: `0.5147640463`;
- best nuisance point: `alpha=0.00`;
- nuisance mean accuracy at alpha=0.00: `0.4539325774`;
- absolute mismatch: `0.0608314689` = **6.083 pp**;
- preregistered tolerance: **1.0 pp**;
- gradnov used for calibration: **no**.

The nuisance curve worsened monotonically from alpha=0.00 through alpha=0.10. At alpha=0 the nuisance construction is the unperturbed clean evaluation, so there is no weaker alpha within this family that can close the baseline-difficulty gap.

The workflow therefore skipped all fresh confirmatory loss-hard/gradnov training and held-out evaluation.

## Design consequence

Difficulty matching cannot be achieved by tuning the nuisance permutation strength alone. The next evaluation-only calibration should vary both held-out families while keeping training fixed: interpolate the shared geometric transformation toward clean with a nonzero shared-factor strength lambda, and independently use a small nonzero nuisance permutation alpha. Select a difficulty-matched pair using loss-hard-only calibration, then freeze both values before any confirmatory gradnov evaluation.
