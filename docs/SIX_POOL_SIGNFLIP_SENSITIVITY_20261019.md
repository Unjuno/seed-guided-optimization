# Post-hoc six-pool sign-flip sensitivity (NOT a replacement decision)

**Date: 2026-10-19.** This is a sensitivity analysis on the already-frozen [PR #126 six-pool results](ROADMAP_VALIDATION_20260926_RESULT.md) from run 36202615155. It uses only the twelve registered primary contrasts and their 72 archived pool means. It does not create new observations, change earlier test definitions or retroactively downgrade/upgrade the preregistered t5/Holm decisions.

## Method and important assumption

For each of the six fixed pool-level paired contrasts, subtract its originally registered null value (0 or 0.005 accuracy fraction). Enumerate all 64 independent sign reversals of the six centered contrasts; calculate the one-sided or two-sided fraction at least as extreme as the observed group mean. This is an **exact enumeration conditional on sign-exchangeability/symmetry under the null**, **not** an assumption-free permutation test. The policies were not randomly reallocated between outcomes. For the Fashion three-comparison and optimizer seven-comparison families, reapply the same Holm family sizes. The Digits two-component tradeoff is conjunctive; no Holm correction was preregistered for those two components.

## Results

| Study / registered contrast | Preregistered t5 outcome | Post-hoc sign-flip p | Post-hoc Holm-adjusted p |
|---|---|---:|---:|
| Digits: gradnov−anchor minimum | PASS | 0.015625 | not required (conjunctive) |
| Digits: gradnov−anchor clean loss | PASS | 0.015625 | not required (conjunctive) |
| Fashion: anchor−loss-hard mean above +0.5pp | PASS | 0.031250 | **0.093750** |
| Fashion: gradnov−anchor mean above +0.5pp | NOT SUPPORTED | 0.984375 | 0.984375 |
| Fashion: gradnov−anchor minimum above +0.5pp | PASS | 0.031250 | **0.093750** |
| AdamW objective main | NOT SUPPORTED | 0.125000 | 0.375000 |
| AdamW estimator main | PASS | 0.031250 | **0.218750** |
| AdamW objective×estimator | PASS | 0.031250 | **0.218750** |
| SGD objective main | NOT SUPPORTED | 0.125000 | 0.375000 |
| SGD estimator main | PASS | 0.031250 | **0.218750** |
| SGD objective×estimator | NOT SUPPORTED | 0.031250 | 0.218750 |
| Three-way configuration contrast | NOT SUPPORTED | 0.906250 | 0.906250 |

**Interpretation:** The Digits finite-minimum/clean directional pattern has the same six-pool signs and both single-component sign-flip p-values are 0.015625. The Fashion and optimizer t5 significances **are not confirmed by this alternative six-pool sign-flip familywise analysis**. This exposes sensitivity to the t-model and limited n. It does **not** falsify the preregistered t5 results or prove null effects.

With only six pools, any two-sided sign-flip p-value is at least 2/64=0.03125 for nondegenerate paired data. Therefore a Holm7 family **cannot possibly** reach the 0.05 threshold by this procedure: its smallest Holm7 adjusted p is at least 7×2/64=0.21875. This is a power/discreteness limitation, not evidence of no optimizer mechanism. Fashion's two positive primary contrasts each include one of six pool contrasts below the practical +0.5pp null; their one-sided sign-flip p is 2/64, which adjusts to 0.09375 under Holm3.

## Reproduce without training

```bash
python scripts/check_roadmap_validation_results.py
python scripts/check_six_pool_signflip_sensitivity.py
```

Uses [12 frozen primary records](../results/roadmap_validation_20260926_primary.csv) and [72 pool contrasts](../results/roadmap_validation_20260926_pool_contrasts.csv). Data are accuracy fractions, not percentages. Complete training/evaluation sources and hashes remain in the [registered result](ROADMAP_VALIDATION_20260926_RESULT.md).

## H / T / D / C / U

- **H:** Post-hoc robustness of the reported group-level signs/statistics to sign-exchangeable null assumptions.
- **T:** Six originally registered pool means per contrast, exhaustive 64 sign patterns; no added study blocks.
- **D:** Sensitivity conclusion only; never replace Issue124/125 scientific labels or preregistered p-values.
- **C:** Large effects may coexist with model misspecification; a non-PASS may instead be caused by discreteness/low power.
- **U:** Sign-flip requires symmetry/exchangeability. Fixed image set, chosen generator family and CPU backend remain additional unmeasured uncertainty sources. Combined uncertainty across unseen images and hardware is not identified.

## Variable table and dimensional check

| Symbol | Meaning (日本語) | SI unit | Definition | Domain/assumption | Type |
|---|---|---|---|---|---|
| `n` | 環境群数 | 1 | fixed 6 | positive integer | integer scalar |
| `b_j` | 環境群ごとの精度差 | 1 | paired mean difference within pool j | [−1,1] | real scalar |
| `c` | 登録済み帰無仮説閾値 | 1 | 0 or 0.005 | frozen, not fitted | real scalar |
| `s_j` | 符号反転 | 1 | +1 or −1 for each pool | symmetry assumption | integer scalar |
| `p_{sf}` | 符号反転p値 | 1 | extremal fraction among 64 assignments | multiples of 1/64 | probability scalar |
| `p_{Holm}` | Holm補正後p値 | 1 | family size 3 or 7 | [0,1] | probability scalar |

Every accuracy difference and null threshold is dimensionless. 0.005 is 0.5 percentage point; 1pp=0.01 accuracy fraction. No timing/energy/power benchmark is involved.
