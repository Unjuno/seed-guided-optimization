# Corrected centered residual-span calibration: PASS

Issue #100 repeated the centered residual-span support calibration on fresh repetitions after Issue #98 exposed an invalid exact-unit numerical audit. The scientific matching, support and span thresholds were unchanged; only the algebraic identity audit was corrected for the approximately unit-norm float32 signatures returned by the implementation.

## Frozen outcome

Authoritative GitHub Actions run: **34583466925**. Evidence artifact: `corrected-centered-span-calibration-evidence`, artifact ID **10192667413**, SHA256 `4f38ba998f26c356454b44c46a90a99cb45faf3e80fa332be87b365c36c85fe0`.

**Decision: `CORRECTED CENTERED SPAN CALIBRATION PASS`.** The selected smallest passing mean-novelty caliper is **c_G = 0.05**.

| G caliper | support | min rep support | mean centered-span gap | min rep mean gap | corrected identity | decision |
|---:|---:|---:|---:|---:|---:|---|
| .005 | .53500 | .4125 | .304440 | .226627 | pass | fail support |
| .010 | .71750 | .5375 | .329305 | .305546 | pass | fail support |
| .020 | .86250 | .7625 | .369600 | .334624 | pass | fail support |
| **.050** | **.96125** | **.9250** | **.441581** | **.399689** | **pass** | **PASS** |

At the selected caliper, 769 of 800 reference steps have an eligible pair. Mean absolute matched differences over supported steps were G .021263, H .021760, selected-loss variance z-score .021432, physical-diversity z-score .024115 and translation score .100686.

The maximum residual of the corrected general identity was **4.440892098500626e-15**, below the frozen `1e-10` gate. The maximum observed deviation of mean squared returned-row norm from one was `1.9582034149756566e-07`; this was descriptive and not an exclusion gate.

No intervention arm was trained and no heldout environment was constructed in this calibration. The conditional fresh Stage B registered in Issue #100 is now authorized with frozen `c_G=.05`.

Committed values: [summary CSV](../results/corrected_centered_span_calibration_summary.csv) and [decision JSON](../results/corrected_centered_span_calibration_decision.json).

## Exact identity used

For returned signature rows `v_i`, define the row mean `v_bar`, mean squared row norm `m_2`, mean pairwise novelty `G` and centered energy `E_c` by

```math
m_2=\frac{1}{q}\sum_i\|v_i\|^2,\qquad
E_c=\sum_i\|v_i-\bar v\|^2.
```

Then

```math
E_c=(q-1)(G+m_2-1).
```

This does not assume exact unit row norms. It therefore matches the actual float32-normalized implementation while retaining the conceptual decomposition: when `G` and the tiny row-norm correction are matched, centered total scatter is controlled while the normalized centered spectrum can vary.

## Variable table

| Symbol | Meaning | SI unit | Definition | Domain / assumption | Type |
|---|---|---:|---|---|---|
| q | selected signature count | 1 | fixed 4 | integer >=2 | integer scalar |
| v_i | returned head-gradient signature | 1 | float32-normalized implementation row, evaluated in float64 | finite | real vector |
| v_bar | mean returned signature | 1 | `q^-1 sum_i v_i` | finite | real vector |
| m_2 | mean squared row norm | 1 | `q^-1 sum_i ||v_i||^2` | nonnegative | real scalar |
| G | mean pairwise novelty | 1 | mean `1-v_i dot v_j` | finite | real scalar |
| E_c | total centered energy | 1 | `sum_i ||v_i-v_bar||^2` | nonnegative | real scalar |
| S_c | centered spectral effective rank | 1 | entropy effective rank of `C C^T` | nominal [1,3] for nondegenerate q=4 | real scalar |

Dimensional check: all quantities are dimensionless, and every term in the corrected identity has compatible units.

## Interpretation boundary

This PASS establishes feasibility of the intended manipulation, not a performance effect. It shows that in fresh reference trajectories, a large centered-spectrum-shape difference can be created on most steps while keeping the registered H/L/P/T/G summaries within their fixed calipers. Whether that difference changes heldout accuracy remains the question for Stage B.

## ERROR CHECK

- First complete valid post-registration run used.
- Fresh reps 2700–2709 and train environments 73000–73063.
- Scientific thresholds unchanged from Issue #98.
- General identity used consistently in float64.
- Zero intervention arms and no heldout construction before the PASS decision.
