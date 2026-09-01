# High-dimensional nuisance calibration gate

## Status

Preregistered in Issue #44 as a gated follow-up to the failed IID nuisance match in Issue #41.

Frozen result: **CALIBRATION MATCH FAILURE**.

No confirmatory training ran and **no confirmatory held-out environments were constructed**. The workflow gate therefore preserved the preregistered separation between training-only nuisance calibration and the held-out mechanism test.

## Nuisance construction

Each nuisance environment seed defines a random permutation `P_e` of the 64 input pixels, shared across all examples in that environment. For severity `lambda`,

```text
x' = (1 - lambda) x + lambda x[P_e].
```

Different environment seeds use unrelated permutations. The construction was intended to create coherent environment-conditioned gradient directions without reusing the low-dimensional geometric factors of the structured condition.

## Frozen calibration

Calibration reps: `550-559`.

Severity grid:

```text
0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90
```

The same score and tolerances as the preregistration were used:

- novelty mismatch tolerance: `0.03`;
- candidate-loss mismatch tolerance: `0.10`.

The best score was at **lambda=0.70**:

- structured novelty gain: `0.157319`;
- high-dimensional nuisance novelty gain: `0.099349`;
- novelty mismatch: **0.057970** > `0.03`;
- structured candidate loss: `1.737117`;
- nuisance candidate loss: `1.674508`;
- candidate-loss mismatch: **0.062609** <= `0.10`.

Thus candidate-loss matching succeeded at the best grid point, but selected-gradient novelty matching did not.

## Interpretation

This is a more informative failure than the preceding IID-noise attempt. Environment-shared high-dimensional structure moved the nuisance novelty gain much closer to the structured target while simultaneously allowing candidate-loss matching. However one scalar interpolation parameter could not satisfy both frozen constraints.

The next calibration design should therefore separate the two control axes rather than relax the preregistered tolerances:

1. a permutation-mixing parameter controlling task difficulty / candidate loss;
2. a second environment-specific feature-field parameter controlling environment-conditioned gradient separation.

Only training-only calibration should be used to select those parameters. Confirmatory held-out evaluation should remain gated until both original match tolerances are satisfied.

## Evidence

- Issue #44: preregistration.
- PR #45: implementation and gated workflow.
- Actions run `33493096449`.
- `results/highdim_calibration_decision.csv`.
- `results/highdim_calibration_summary.csv`.
