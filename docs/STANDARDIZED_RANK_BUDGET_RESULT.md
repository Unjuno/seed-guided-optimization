# Scale-invariant mediator Q-scaling replication

## Status

**Frozen decision: `COVERAGE REPLICATION ONLY`.**

Issue #38 preregistered a fresh n=30 Q-scaling replication after two prior findings:

1. the original budget audit strongly supported attenuation of the gradnov held-out benefit as Q approached K, but raw representation-rank attenuation failed its frozen threshold;
2. the function-preserving intervention in Issue #35 showed that raw hidden effective rank is coordinate-dependent, while channel-standardized effective rank was exactly invariant to positive channel rescaling.

This experiment therefore tested the standardized statistic as a new scale-invariant mediator candidate without recalibrating it on previous held-out results.

Authoritative execution: GitHub Actions run `33392125145`.

## Frozen protocol

- Digits geometric MLP, `64 -> 128 ReLU -> 10`;
- train environment seeds `13000-13063`;
- fresh held-out seeds `17000-17079`;
- fresh paired replicates `300-329`, n=30;
- AdamW lr `1e-2`, weight decay `1e-3`, 10 epochs, batch 128;
- candidate count `K=16`, Q=`[2,4,8,12,16]`;
- loss-hard vs gradnov, novelty weight 0.6;
- fixed training-only 192-example probe over train-environment indices `[1,9,17,25,33,41,49,57]`;
- all training-only ranks/states sealed before fresh held-out construction.

## Q=16 identity

At Q=K both selectors use the complete candidate set in the same sorted order.

Across all 30 replicate pairs:

- exact scientific mismatches: **0**;
- **IDENTITY PASS**.

## Held-out benefit replication

Mean gradnov-minus-loss-hard held-out accuracy by Q:

| Q | Q/K | mean benefit |
|---:|---:|---:|
| 2 | 0.125 | +2.613 pp |
| 4 | 0.250 | +2.443 pp |
| 8 | 0.500 | +2.156 pp |
| 12 | 0.750 | +0.885 pp |
| 16 | 1.000 | 0.000 pp exactly |

Frozen contrast:

- mean low-Q benefit, Q={2,4}: **+2.528 pp**;
- mean high-Q benefit, Q={12,16}: **+0.442 pp**;
- attenuation `A_B`: **+2.086 pp**;
- SE: **0.339 pp**;
- approximate two-sided 95% t interval: **+1.393 to +2.778 pp**;
- t=6.159;
- one-sided p=`5.16e-7`;
- **COVERAGE REPLICATION PASS**.

The selected-gradient novelty manipulation again decayed strongly with Q: mean gradnov-minus-loss-hard novelty was +0.2804 at Q=2, +0.1377 at Q=4, +0.0615 at Q=8, +0.0238 at Q=12, and exactly zero at Q=16.

This independently reproduces the core finite-budget subset-coverage result on fresh replicate and held-out seed blocks.

## Standardized-rank mediator test

Channel-standardized representation effective-rank differences by Q:

| Q | mean standardized-rank delta |
|---:|---:|
| 2 | -0.0006 |
| 4 | +0.0815 |
| 8 | +0.0850 |
| 12 | +0.0863 |
| 16 | 0.0000 |

Frozen low-vs-high contrast:

- mean standardized-rank low: +0.0405;
- mean standardized-rank high: +0.0432;
- attenuation `A_Z`: **-0.00271**;
- SE: 0.1695;
- approximate two-sided 95% t interval: **-0.3493 to +0.3439**;
- t=-0.0160;
- one-sided p=0.5063;
- **STANDARDIZED-MEDIATOR FAIL**.

The secondary replicate-level correlation between benefit attenuation and standardized-rank attenuation was r=-0.064, p=0.736. This is descriptive only.

Thus removing trivial per-channel scale dependence does not rescue effective rank as the quantitative mediator of the Q-scaling effect.

## Raw-rank secondary comparator

Raw effective rank was preregistered as secondary only because Issue #35 already demonstrated that it is coordinate-dependent.

Nevertheless, in this fresh replication its low-vs-high attenuation was:

- mean `A_raw`: **+0.1780**;
- SE: 0.0809;
- approximate 95% interval: **+0.0125 to +0.3436**;
- one-sided p=0.0180.

This cannot upgrade the frozen decision. It does, however, sharpen the interpretation: raw rank repeatedly responds to the training trajectory under the repository's fixed parameterization even though it is not itself a functionally intrinsic causal quantity.

## Interpretation

The combined evidence now separates three claims:

1. **Finite-budget gradient-subset coverage:** strongly supported and independently replicated. Gradnov benefit is large when only a small subset can enter each update, attenuates as Q approaches K, and vanishes exactly when Q=K.
2. **Raw representation effective rank as causal mediator:** rejected. It can be changed strongly without changing the function.
3. **Channel-standardized effective rank as quantitative mediator:** rejected under this frozen Q-scaling test. It is scale-invariant but does not track benefit attenuation.

Therefore the next mechanism search should not simply invent another rank normalization. The missing internal quantity should be tied more directly to **which reusable task functions are preserved or acquired**, and should ideally be invariant to function-preserving reparameterizations.

Promising next classes are function-space or predictive diagnostics: cross-environment probe transfer, class-conditional margin geometry, representation subspace alignment defined after whitening, or linear-probe performance on held-out combinations constructed strictly after the diagnostic definition is frozen.

## Evidence

- preregistration: Issue #38;
- implementation: PR #39;
- Actions run `33392125145`;
- `results/standardized_rank_budget_decision30.csv`;
- `results/standardized_rank_budget_q_summary30.csv`;
- `results/standardized_rank_budget_attenuation30.csv`.
