# Loss-rank allocation matched-geometry calibration — result

Issue #104 Stage A tested whether mean selected candidate loss rank R can be manipulated while matching loss-shape, scalar physical diversity, translation allocation, mean gradient novelty, centered residual span, and maximum pairwise opposition.

## Frozen decision

**RANK-ALLOCATION CALIBRATION PASS**

Authoritative GitHub Actions run: `34696199565`.

No intervention arm was trained and no heldout environment was constructed in Stage A.

## Registered gates

Observed across 800 fresh reference steps:

- supported steps: `765/800 = 0.95625`;
- minimum replicate support: `0.925`;
- mean supported high-R minus low-R gap: `1.589869` candidate-rank units;
- minimum replicate mean rank gap: `1.477273`;
- max corrected general-identity residual: `3.997e-15`.

Frozen gates were pooled support >=.90, every-replicate support >=.80, mean R gap >=1.50, every-replicate mean gap >=.75, and identity residual <=1e-10. All passed.

Matched summaries stayed within their exact per-step calipers on supported comparisons. Mean absolute differences were:

- selected-loss variance z: `0.024524` <= .05;
- physical diversity z: `0.024570` <= .05;
- translation score: `0.098254` <= .20;
- mean gradient novelty: `0.019522` <= .05;
- centered residual effective rank: `0.047437` <= .10;
- maximum pairwise opposition: `0.029118` <= .15.

The intended correlate, selected mean hardness H, was not matched and moved by `-0.293598` z units for high-R minus low-R: the high-R arm uses systematically less-hard candidate sets as intended.

## Interpretation

Unlike centered span and pairwise opposition, loss-rank allocation is independently manipulable with high support under the current 455-subset natural candidate geometry while the registered gradient-geometry and physical summaries remain close. This authorizes the separately preregistered fresh Stage B performance intervention in Issue #104.

Evidence artifact: `rank-allocation-calibration-evidence`, artifact ID `10297439895`, SHA256 `99cf884c9172f83eabdd8d710052360b0f122148ca61b61720cc73ce84216880`.
