# Pairwise-opposition matched-geometry calibration — result

Issue #103 Stage A tested whether maximum pairwise directional opposition `O=max(1-v_i dot v_j)` can be manipulated while matching selected-loss mean/shape, scalar physical diversity, translation allocation, mean gradient novelty G, and centered residual effective rank S_c.

## Frozen decision

**OPPOSITION CALIBRATION FAIL**

Authoritative GitHub Actions run: `34696012612`.

No intervention arm was trained and no heldout environment was constructed.

## Registered grid

| S_c caliper | support | minimum replicate support | mean supported O gap | minimum replicate mean O gap | pass |
|---:|---:|---:|---:|---:|---|
| .02 | .37875 | .3125 | .036549 | .019148 | no |
| .05 | .60125 | .5000 | .046114 | .034532 | no |
| .10 | .75125 | .6500 | .056692 | .043445 | no |
| .20 | .87250 | .8375 | .073560 | .065115 | no |

Frozen gates were pooled support >=.90, every replicate support >=.80, mean O gap >=.15, every-replicate mean O gap >=.08, plus the numerical identity audit. The largest span tolerance still failed pooled support and both O-gap requirements. The corrected centered-energy identity audit passed, with maximum residual `3.55e-15`.

## Interpretation

This is an **identification/overlap failure**, not a heldout null result. Within the natural 455-subset candidate family, once mean novelty and centered span are simultaneously controlled together with the registered loss/physical summaries, the maximum-pairwise opposition score cannot be separated at the preregistered magnitude and support.

The result therefore blocks a clean confirmatory performance test of this specific opposition-tail mediator using the current candidate geometry. It also suggests that mean novelty, centered spectral shape, and extreme pairwise separation are more geometrically constrained together than the synthetic equal-G counterexamples alone imply.

Evidence artifact: `opposition-calibration-evidence`, artifact ID `10298592213`, SHA256 `67efbc65a0400b28c3b679351e69224d33e9c243c8d4cc6ec64014241e66e905`.
