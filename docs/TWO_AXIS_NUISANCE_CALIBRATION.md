# Two-axis nuisance calibration result

## Status

Preregistered in Issue #47 and executed in GitHub Actions run `33495422438`.

Frozen result: **CALIBRATION MATCH FAILURE**.

No confirmatory training or held-out environments were constructed.

## Frozen design

The nuisance family used two environment-level controls:

- `alpha`: random pixel-permutation mixture strength;
- `beta`: amplitude of an environment-specific fixed 64-D random field.

The calibration target was the structured geometric condition. Reps `650-659` were training-only. The frozen grid was

- `alpha in {0.60, 0.65, 0.70, 0.75}`;
- `beta in {0.03, 0.06, 0.09, 0.12, 0.15}`.

The frozen score was

```text
abs(novelty_gain_nuisance - novelty_gain_structured) / 0.03
+ abs(candidate_loss_nuisance - candidate_loss_structured) / 0.10
```

and the gate required both mismatches to satisfy their original tolerances.

## Result

The selected point was `alpha=0.65`, `beta=0.15`.

- structured novelty gain: `0.1517915`;
- nuisance novelty gain: `0.1077022`;
- novelty mismatch: **0.0440893 > 0.03**;
- structured candidate loss: `1.7303353`;
- nuisance candidate loss: `1.7289712`;
- candidate-loss mismatch: **0.0013641 <= 0.10**.

Therefore the frozen gate failed.

The workflow then skipped both the confirmatory and aggregate jobs, so the fresh confirmatory held-out block was never constructed or inspected.

## Mechanistic lesson

The two-axis design solved the loss-matching problem almost exactly but still under-produced selector-level gradient novelty. The calibration surface is informative:

- decreasing `alpha` moves nuisance novelty toward the structured target;
- increasing `beta` restores candidate loss after reducing `alpha`;
- at `alpha=0.60`, several points approach the novelty tolerance but remain too easy;
- at `alpha=0.65, beta=0.15`, difficulty is essentially exact but novelty remains short.

This suggests the next training-only calibration should explore **lower alpha and higher beta**, rather than relaxing the preregistered novelty tolerance. Because no held-out data were generated in this run, that refinement can be based entirely on training-only calibration evidence.

## Claim boundary

This run provides no evidence for or against the reusability-conversion held-out hypothesis. It is a calibration failure only.
