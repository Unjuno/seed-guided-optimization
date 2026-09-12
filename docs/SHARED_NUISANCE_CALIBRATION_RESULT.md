# Coherent high-dimensional shared-nuisance calibration — result

Issue #110 Stage A replaced the earlier IID-per-example noise family with an environment-shared deterministic 8x8 random pixel pattern. The goal was to match the structured geometric family on selector novelty gain and training candidate-loss level before any heldout evaluation.

## Frozen decision

**SHARED-NUISANCE MATCH CALIBRATION FAIL**

Authoritative GitHub Actions run: `34696852735`.

No heldout environment was constructed and no model state was saved for a confirmatory comparison.

## Main result

The structured family had mean selector novelty gain `0.143981` and mean candidate loss `1.644476`.

As nuisance severity increased from `.05` to `.70`, the shared-nuisance novelty gain increased monotonically from `0.003744` to `0.116837`. At severity `.70`, the novelty-gain difference was only `-0.027145`, which satisfies the frozen +/- .03 novelty matching tolerance. However, candidate loss remained `0.562018`, giving `dC=-1.082458`, far outside the frozen +/- .10 loss tolerance.

Thus the new environment-shared pattern **did solve the old minibatch-averaging problem for gradient novelty**, but the registered amplitude range remained much easier than the coherent geometric environment family.

## Interpretation

This is a calibration failure, not a performance result. The useful discovery is that coherent environment-specific high-dimensional nuisance can generate nearly the same selector novelty gain as structured geometry, but matching optimization difficulty requires a substantially stronger or separately difficulty-controlled nuisance construction.

The frozen grid is not extended in Issue #110. Any stronger severity or two-parameter difficulty calibration must be a separately preregistered experiment with fresh repetitions.

Evidence artifact: `shared-nuisance-calibration-evidence`, artifact ID `10299136963`, SHA256 `297bf85418b6ca1de031ba76f0f72bcdafff42bc72adf2024ac5a99015e56a22`.
