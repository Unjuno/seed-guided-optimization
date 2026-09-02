# Identical-training transfer-specificity calibration

## Status

Frozen result: **CALIBRATION DIFFICULTY MATCH FAILURE**.

This experiment was preregistered in Issue #53 and implemented in PR #54. The authoritative Actions run used loss-hard-only calibration models trained on geometric environments. Gradnov was not used for calibration.

## Frozen calibration result

The shared-factor geometric held-out mean loss-hard accuracy was `0.5376067424`.

The nuisance alpha grid was `[0.10, 0.15, ..., 0.90]`. The best point was the left boundary, `alpha=0.10`, with nuisance mean loss-hard accuracy `0.4193370785` and absolute mismatch `0.1182696639` (11.827 pp), exceeding the preregistered 1.0 pp tolerance.

The nuisance mean accuracy decreased monotonically as alpha increased, reaching `0.1248426962` at `alpha=0.90`. Therefore the failed gate is not a grid-resolution issue inside the tested interval: the matching region, if it exists for this nuisance construction, lies below `alpha=0.10`.

## Consequence

The workflow correctly skipped fresh confirmatory loss-hard/gradnov training and all confirmatory held-out evaluation. No gradnov confirmatory effect was observed or used to alter this calibration conclusion.

## Next test

Preserve the identical-training transfer-specificity design and the original statistical rules. Run a new loss-hard-only evaluation calibration on fresh calibration reps/seeds with a preregistered fine grid below `alpha=0.10`. Proceed to fresh confirmatory paired training only if the 1.0 pp calibration gate passes.
