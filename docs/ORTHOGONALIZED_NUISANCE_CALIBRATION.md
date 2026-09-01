# Orthogonalized nuisance calibration result

## Status

Preregistered in Issue #50 and executed in GitHub Actions run `33496202097`.

Frozen result: **CALIBRATION MATCH FAILURE**.

No confirmatory training or held-out environments were constructed.

## Design

The nuisance generator separated two intended roles:

- `alpha`: environment-specific random pixel-permutation mixture;
- `gamma`: one contrast attenuation shared by every nuisance environment.

The purpose was to let `alpha` control between-environment high-dimensional structure while `gamma` primarily tuned common difficulty.

Calibration used fresh training-only reps `750-759`, with

- `alpha in {0.45, 0.50, 0.55, 0.60}`;
- `gamma in {0.00, 0.10, 0.20, 0.30, 0.40, 0.50}`.

The original match tolerances remained unchanged:

- novelty mismatch <= `0.03`;
- candidate-loss mismatch <= `0.10`.

## Result

The frozen score selected `alpha=0.60`, `gamma=0.30`.

- structured novelty gain: `0.1489169`;
- nuisance novelty gain: `0.1024473`;
- novelty mismatch: **0.0464696 > 0.03**;
- structured candidate loss: `1.6824316`;
- nuisance candidate loss: `1.5996847`;
- candidate-loss mismatch: **0.0827469 <= 0.10**.

Therefore the calibration gate failed.

The workflow skipped confirmatory training and aggregate evaluation, so the fresh confirmatory held-out block was never constructed.

## What the surface shows

The attempted common difficulty axis was not orthogonal enough in practice. Increasing `gamma` raised candidate loss, but it also reduced selector-level novelty gain. For example at `alpha=0.60`, novelty mismatch was close to tolerance at low gamma (`0.03226` at gamma `0.10`) while candidate loss was much too low; raising gamma toward the loss-matched region pushed novelty farther away.

This is the third training-only matching attempt to show that novelty and difficulty are strongly coupled across substantially different environment generators. Continuing to tune nuisance generators until both numbers match would risk turning the mechanism test into calibration overfitting.

## Consequence for the next experiment

The next falsification should avoid cross-generator training matching entirely. A cleaner design is to train each loss-hard/gradnov model **once on the same geometric environments**, then evaluate the exact same trained models on two fresh held-out families:

1. a shared-factor geometric family drawn from the training generator;
2. an unrelated nuisance family whose baseline difficulty is calibrated on a separate evaluation-only block.

Because training is identical, training trajectory, selected-gradient novelty, candidate loss, optimizer state, and update budget are controlled by construction. The primary question then becomes whether the gradnov advantage transfers preferentially to held-out environments that reuse the training latent structure.

## Claim boundary

This run provides no held-out evidence for or against the reusable-factor hypothesis. It establishes only that the attempted cross-generator training match failed and motivates a same-training transfer-specificity design.
