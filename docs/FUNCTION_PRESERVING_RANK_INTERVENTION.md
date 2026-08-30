# Function-preserving representation-rank intervention

## Status

Preregistered mechanism falsification from Issue #35.

**Frozen decision: COORDINATE-DEPENDENCE PASS.**

The experiment tests whether the repository's raw hidden-representation effective rank can itself be a causal/functionally intrinsic quantity. It uses a positive diagonal reparameterization of the one-hidden-layer MLP:

```text
h = ReLU(Ax + b)
z = W h + c

A' = D A
b' = D b
W' = W D^-1
```

For positive diagonal `D`, the represented function is unchanged mathematically while the coordinates and scale geometry of `h` change.

## Preregistered protocol

- Digits geometric benchmark;
- MLP `64 -> 128 ReLU -> 10`;
- train seeds `13000-13063`;
- fresh held-out seeds `16000-16079`;
- AdamW lr `1e-2`, weight decay `1e-3`;
- 10 epochs, batch 128;
- `K=16`, `Q=4`;
- loss-hard and gradient-novelty (`weight=0.6`);
- paired replicate IDs `200-219`, n=20 per method;
- fixed 192-example training-only probe and train audit environments `[1,9,17,25,33,41,49,57]`;
- interventions fixed from training-only hidden activations before held-out construction.

## Frozen interventions

### Spread

Channel scales were chosen as clipped powers of two to approximately equalize native training-probe channel standard deviations.

### Concentrate

The 16 channels with largest native training-probe standard deviation were scaled by `8`; all remaining channels were scaled by `1/8`.

For both interventions, the first-layer rows/biases were multiplied by the positive scale and final-head columns divided by the same scale. No further training occurred.

## Frozen success rule

`MANIPULATION PASS` required, separately for loss-hard and gradnov:

- mean spread rank change >= `+0.5`;
- mean concentrate rank change <= `-0.5`;
- at least 18/20 replicates with the intended sign for each intervention.

`FUNCTION IDENTITY PASS` required every non-native row to satisfy:

- exactly identical predicted classes on every fresh held-out and clean example;
- exactly identical held-out mean / SD / p10 / minimum / clean accuracy;
- maximum absolute logit difference <= `1e-5`.

Overall `COORDINATE-DEPENDENCE PASS = MANIPULATION PASS + FUNCTION IDENTITY PASS`.

## Authoritative execution

GitHub Actions run `33313908374` completed all four 5-replicate shards and the frozen aggregate successfully.

Independent recomputation from the aggregate artifact reproduced the frozen decision.

## Raw effective-rank manipulation

| Training method | mean spread delta | positive count | mean concentrate delta | negative count | decision |
|---|---:|---:|---:|---:|---|
| loss-hard | **+2.3308** | 20/20 | **-5.8330** | 20/20 | PASS |
| gradnov | **+2.0808** | 20/20 | **-6.1288** | 20/20 | PASS |

Thus the same trained function can be assigned substantially larger or substantially smaller raw hidden effective rank by changing only the hidden coordinate scaling and compensating the final head.

## Function identity

Across `20 reps x 2 methods x 2 non-native interventions = 80` intervention rows:

- prediction mismatch rows: **0**;
- metric mismatch rows: **0**;
- maximum absolute logit difference: **0.0**;
- `FUNCTION IDENTITY PASS = true`.

The held-out and clean behavior was therefore not merely statistically similar: it was identical under the evaluated floating-point execution.

## Scale-standardized rank diagnostic

The preregistered secondary diagnostic z-scales every nonconstant hidden channel before SVD.

Across all 40 trained models:

- spread minus native standardized-rank delta: **0.0 exactly**;
- concentrate minus native standardized-rank delta: **0.0 exactly**.

This confirms that the raw-rank intervention acts entirely through positive channel scaling, and that simple per-channel standardization removes this particular coordinate freedom.

This does **not** prospectively validate standardized rank as a predictor or causal mediator. It is only a candidate invariant diagnostic for future preregistered tests.

## Secondary selector observation

On the fresh held-out seed block, the native gradnov-minus-loss-hard held-out mean benefit was approximately **+2.541 pp** across the 20 paired replicates, positive in 19/20 pairs. This is descriptive only and is not part of the intervention decision.

## Interpretation

This experiment rejects the stronger interpretation:

> increasing raw hidden representation effective rank itself causes improved held-out behavior.

Raw effective rank is not invariant to a function-preserving reparameterization. Therefore it cannot by itself be a functionally intrinsic causal quantity.

The previous prospective rank record remains meaningful in a narrower sense: under a fixed architecture/training parameterization, raw rank has repeatedly acted as a useful **condition-average directional marker**. The present result shows that this empirical marker must not be promoted into a coordinate-free causal law.

The mechanism theory should now separate two claims:

1. **finite-budget gradient-subset coverage** — directly supported by the preregistered Q-scaling audit;
2. **internal representation mediator** — still unidentified. Raw effective rank is an incomplete, coordinate-dependent marker rather than the mediator itself.

A next-generation representation diagnostic should be invariant to trivial positive channel rescalings and should be prospectively validated before being used as a mechanistic gate.

## Evidence

- Issue #35 — preregistration and frozen decision rule;
- PR #36 — implementation and workflow;
- Actions run `33313908374`;
- `results/function_preserving_rank_decision20.csv`;
- `results/function_preserving_rank_manipulation_summary20.csv`;
- `results/function_preserving_rank_deltas20.csv`.
