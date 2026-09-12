# Strong shared-nuisance calibration — result

Issue #116 repeated the coherent environment-shared high-dimensional nuisance calibration on a prospectively fixed stronger severity range after Issue #110 showed novelty matching but insufficient difficulty.

## Frozen Stage-A result

**PASS; selected severity `a=3.0`.**

Authoritative GitHub Actions run: `34697487866`.

No heldout environment was constructed and no confirmatory state was saved in Stage A.

At severity 3.0:
- structured selector novelty gain: `0.122605`;
- shared-nuisance selector novelty gain: `0.110025`;
- `dN = -0.012579`, inside the frozen +/- .03 tolerance;
- structured candidate-loss level: `1.730328`;
- shared-nuisance candidate-loss level: `1.683264`;
- `dC = -0.047065`, inside the frozen +/- .10 tolerance.

It was the only eligible point in the frozen grid. At severity 2.5 difficulty remained too low (`dC=-0.248462`), while at 4.0 it had become too high (`dC=+0.221733`).

## Interpretation

The environment-shared high-dimensional nuisance construction can be calibrated to approximately match coherent geometric environments on both the gradient-novelty opportunity available to the selector and the training candidate-loss level. This resolves the calibration obstacle that made the earlier structured-vs-unstructured comparison inconclusive.

The separately preregistered Issue #116 Stage B is therefore authorized at frozen severity 3.0 on fresh repetitions/environment seeds. A positive structured-minus-nuisance conversion contrast would support the hypothesis that novelty becomes useful when it corresponds to reusable shared structure rather than merely environment-specific high-dimensional disagreement.

Evidence artifact: `strong-shared-nuisance-calibration-evidence`, artifact ID `10299008259`, SHA256 `e14cb1d03b2387c84ff61c7fb367e4afc50c69c4cb6d5906e0847a1a732bd4f9`.
