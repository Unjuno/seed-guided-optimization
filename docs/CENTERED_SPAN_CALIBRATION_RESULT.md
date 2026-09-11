# Centered residual-span calibration: numerical-gate FAIL

Issue #98 prospectively replaced Issue96's uncentered spectral score with a centered residual spectral effective rank and expanded the fixed subset family from 84 to 455 subsets per reference step. It also matched selected-loss variance in addition to hardness mean, physical diversity, translation allocation and mean pairwise novelty G.

## Frozen result

Authoritative GitHub Actions run: **34582755422**. Evidence artifact: `centered-span-calibration-evidence`, ID **10192374381**, SHA256 `ccf757eaff339de6cb433bc9cc162d615f44cdc771d7b0814401469a8c9b4435`.

**Decision: `CENTERED SPAN CALIBRATION FAIL`.** No intervention arm was trained and no heldout environment was constructed. Stage B was not run.

| G caliper | support | min rep support | mean centered-span gap | min rep mean gap | identity pass |
|---:|---:|---:|---:|---:|---:|
| .005 | .58750 | .5000 | .308791 | .260218 | no |
| .010 | .74750 | .6750 | .344726 | .288074 | no |
| .020 | .87750 | .8500 | .394021 | .358452 | no |
| .050 | **.96625** | **.9375** | **.468571** | **.430082** | **no** |

At c_G=.05 all registered scientific support/span thresholds passed. The failure came from the separate numerical audit: the preregistration required `max abs(E_c-(Q-1)G) <= 1e-8`, while the observed maximum was **6.201591e-7**.

Committed values: [summary CSV](../results/centered_span_calibration_summary.csv) and [decision JSON](../results/centered_span_calibration_decision.json).

## Why the numerical identity failed

The simplified identity assumes exact unit row norms. `head_gradient_directions` normalizes in float32, so returned row norms are numerically close to, but not exactly, one. For arbitrary returned rows `v_i`, let

```math
m_2 = \frac{1}{q}\sum_i \|v_i\|^2.
```

Then the exact algebraic relation is

```math
E_c = \sum_i\|v_i-\bar v\|^2
    = (q-1)(G + m_2 - 1).
```

The Issue98 gate omitted the `m_2-1` correction. This is a specification error in the numerical audit, not evidence against centered residual span. The frozen Issue98 decision nevertheless remains FAIL.

### Variable table

| Symbol | Meaning | SI unit | Definition | Domain / assumption | Type |
|---|---|---:|---|---|---|
| q | selected signature count | 1 | fixed 4 | integer >=2 | integer scalar |
| v_i | returned normalized head-gradient signature | 1 | existing float32 implementation | finite, approximately unit norm | real vector |
| v_bar | mean returned signature | 1 | q^-1 sum v_i | fixed head coordinates | real vector |
| m_2 | mean squared row norm | 1 | q^-1 sum ||v_i||^2 | nonnegative | real scalar |
| E_c | total centered energy | 1 | sum ||v_i-v_bar||^2 | nonnegative | real scalar |
| G | mean pairwise novelty | 1 | mean 1-v_i dot v_j | finite | real scalar |

Dimensional check: all terms are dimensionless, so the corrected equality is dimensionally consistent.

## Post-hoc measurement diagnostic

This does not alter the frozen decision. Across the 364,000 archived subset rows, centered spectral effective rank and G had pooled Pearson correlation **0.1792** and Spearman correlation **0.1896**. The median within-reference-step Pearson correlation was **0.06835**; the 5th–95th percentile range was approximately **-0.4232 to +0.4650**.

This is a large reduction from Issue96's uncentered score (pooled Pearson .9584). The centered score therefore behaves much more like the intended residual-shape quantity rather than a second measurement of total directional cancellation.

## Next test

Do not relax Issue98 retroactively. A new fresh calibration should keep the scientific H/L/P/T/G matching and centered-span thresholds unchanged, but replace the invalid exact-unit audit with the general identity above, computed consistently in float64 from the returned signatures.

## ERROR CHECK

- First complete post-registration run used.
- Fresh reps 2500–2509 and training environments 69000–69063.
- Zero intervention arms and no heldout construction.
- Scientific support/span thresholds and numerical identity gate are reported separately.
- Post-hoc correlations do not upgrade the frozen FAIL.
