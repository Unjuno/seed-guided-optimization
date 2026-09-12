# Strong shared-nuisance confirmatory match barrier — result

Issue #116 Stage B was allowed only after the fresh 10-replicate calibration selected shared-nuisance severity `a=3.0`. The confirmatory protocol then required all 120 training states and diagnostics to be sealed before checking whether the structured and shared-nuisance families still matched on training-only selector novelty gain and candidate-loss level across 30 new training blocks.

## Frozen decision

**STRONG SHARED-NUISANCE CONFIRMATORY MATCH FAILURE**

Authoritative GitHub Actions run: `34697836532`.

All 120 states were trained and sealed, but the fresh training-only match barrier failed. Therefore the evaluation and aggregate jobs were skipped and **no heldout environment was constructed**.

Fresh 30-block differences, shared nuisance minus structured:

- selector novelty-gain difference: `dN = -0.0450228621`, outside the preregistered `+/-0.03` tolerance;
- candidate-loss difference: `dC = +0.0160998719`, inside the preregistered `+/-0.10` tolerance.

Thus the stronger nuisance successfully reproduced optimization difficulty but did not reproduce the calibrated novelty-gain match tightly enough on the independent confirmation blocks.

## Interpretation

This is a **confirmatory matching/identification failure**, not a performance null. The reusable-structure conversion hypothesis was never exposed to heldout outcomes in this Stage B, so no structured-versus-nuisance generalization comparison is licensed.

The result also shows that matching the two environment families on selector novelty gain is less stable across fresh training blocks than the 10-replicate calibration suggested. Per the preregistered stopping logic, the present additive shared-pattern route is ended rather than repeatedly retuning severity to the confirmation sample.

A future reusable-structure test would need a design that controls the relevant gradient opportunity by construction rather than by a scalar post-hoc severity calibration.

Training-match seal artifact: `strong-reusability-training-global-seal`, artifact ID `10299028929`, SHA256 `93fc886a21694f25ba4df448760aa92c08ec82c150f110c3511edef2c7c433d8`.
