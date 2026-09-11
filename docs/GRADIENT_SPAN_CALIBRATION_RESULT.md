# Gradient-span discrimination calibration: Stage A FAIL

Issue #96 preregistered a training-reference-only feasibility test asking whether spectral effective rank of selected normalized head-gradient signatures could be changed while approximately matching reference hardness, total physical diversity, translation allocation and mean pairwise gradient novelty.

## Frozen outcome

Authoritative GitHub Actions run: **34581895307**. All five fresh pilot jobs and the aggregate completed successfully. The archived evidence artifact is `gradient-span-calibration-evidence`, artifact ID `10192016112`, SHA256 `00e9533cbb88e44881fb3fb0e0a5c569d2d3f0a796bc33949a3c281b17b0411f`.

**Decision: `SPAN MATCH CALIBRATION FAIL`.**

No intervention arm was trained and no heldout environment was constructed. The conditional Stage B in Issue #96 therefore was not run.

| G caliper | supported steps / 800 | pooled support | minimum per-rep support | mean supported S gap | minimum per-rep mean S gap |
|---:|---:|---:|---:|---:|---:|
| .005 | 209 | .26125 | .2125 | .058442 | .027324 |
| .010 | 327 | .40875 | .3500 | .076093 | .032547 |
| .020 | 444 | .55500 | .5000 | .101540 | .070698 |
| .050 | 613 | .76625 | .7000 | .150933 | .120924 |

The registered PASS gates were pooled support >= .90, every-replicate support >= .80, mean supported S gap >= .20 and every-replicate mean supported S gap >= .10. The largest frozen G caliper still missed three of the four gates.

Committed values: [summary CSV](../results/gradient_span_calibration_summary.csv) and [decision JSON](../results/gradient_span_calibration_decision.json).

## Interpretation

This is an **identification/design failure**, not a negative performance result. Within the frozen 84-subset family, the uncentered spectral effective-rank score could not be separated broadly and strongly enough from mean pairwise novelty while also satisfying the H/P/T constraints.

The experiment therefore supplies no evidence that gradient span has zero effect and no evidence for or against a heldout performance contrast. Running the preregistered Stage B after this failure would violate the registered barrier.

## Post-hoc diagnostic that motivates the next experiment

This section is exploratory and does not alter the frozen decision.

Across all 67,200 archived subset rows, the Pearson correlation between mean pairwise novelty G and the uncentered spectral effective rank S was **0.9584**; Spearman correlation was **0.9600**. Computing the correlation separately within each of the 800 reference steps gave a median Pearson correlation of **0.9358** (5th–95th percentile approximately **0.6756–0.9956**).

Thus the chosen uncentered S measure strongly co-moves with G in the realized candidate geometry. This is consistent with the fact that S includes energy in the mean gradient direction as well as residual directions.

A cleaner next decomposition is to center the selected normalized signatures before taking the spectral effective rank. If every selected signature has unit norm and `C_i = v_i - v_bar`, then

```math
\sum_i \|C_i\|^2
= q(1-\|\bar v\|^2)
= (q-1)G.
```

Therefore, at fixed q and mean pairwise novelty G, the **total centered gradient energy is fixed**, while the normalized spectrum of the centered Gram/covariance can still describe how that fixed residual energy is distributed across independent directions. This gives a more direct separation between directional cancellation/total scatter and residual span shape.

## Variable table for the centered decomposition

| Symbol | Meaning | SI unit | Definition | Domain / assumption | Type |
|---|---|---:|---|---|---|
| q | selected signature count | 1 | here 4 | integer >=2 | integer scalar |
| v_i | normalized head-gradient signature | 1 | existing implementation | finite; unit norm when above floor | real vector |
| v_bar | mean signature | 1 | `(1/q) sum_i v_i` | same coordinates | real vector |
| C_i | centered signature | 1 | `v_i-v_bar` | row sum zero | real vector |
| G | mean pairwise novelty | 1 | average `1-v_i dot v_j` | fixed q; unit-norm identity above | real scalar |
| E_C | total centered energy | 1 | `sum_i ||C_i||^2` | nonnegative | real scalar |

### Dimensional check

All signatures are normalized model-gradient coordinates and dimensionless in this measurement convention. Squared norms, G and centered energy are therefore dimensionless, and `E_C=(q-1)G` is dimensionally consistent.

## ERROR CHECK

- The authoritative frozen result comes from the first complete valid post-registration Actions run.
- Stage A used fresh repetitions 2300–2309 and training environments 65000–65063.
- The workflow recorded zero intervention arms and no heldout construction.
- The post-hoc correlations and centered decomposition do not change the preregistered Stage-A decision.
