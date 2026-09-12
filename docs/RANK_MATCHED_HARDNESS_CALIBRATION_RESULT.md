# Rank-matched selected-hardness calibration — result

Issue #113 Stage A asked whether selected standardized mean CE hardness H can be manipulated while mean candidate loss rank R is held approximately fixed, together with L/P/T/G/centered-span/opposition.

## Frozen decision

**RANK-MATCHED HARDNESS CALIBRATION FAIL**

Authoritative GitHub Actions run: `34697368629`.

No intervention arm was trained and no heldout environment was constructed.

## Registered grid

| R caliper | support | minimum replicate support | mean supported H gap | minimum replicate mean H gap | pass |
|---:|---:|---:|---:|---:|---|
| 0.00 | .19625 | .1375 | .041217 | .021649 | no |
| .25 | .77875 | .7250 | .049743 | .037807 | no |
| .50 | .87875 | .8375 | .079377 | .071258 | no |
| .75 | .92625 | .8875 | .108913 | .094511 | no |

Frozen gates required pooled support >=.90, every-replicate support >=.80, mean high-H minus low-H gap >=.20 z, every-replicate mean gap >=.10 z, plus identity residual <=1e-10. At the widest registered rank caliper (.75), overlap passed but the hardness manipulation remained only about half the registered magnitude.

The other registered matching summaries stayed within their exact per-step calipers. Maximum corrected centered-energy identity residual was `3.997e-15`.

## Interpretation

This is an identification/overlap limitation, not a heldout null result. Together with Issue #112, both directions of the attempted decomposition fail: large rank shifts disappear when H is tightly matched, and large H shifts disappear when R is tightly matched. Under the present candidate generator and subset family, candidate loss rank and selected CE hardness behave as a strongly coupled intervention axis.

Accordingly, Issue #104's confirmed +0.9066 pp mean heldout benefit should be interpreted as support for **relaxing concentration on the hardest candidates / lowering selected hardness jointly**, not as a clean causal effect of rank or CE hardness alone.

Evidence artifact: `rank-matched-hardness-calibration-evidence`, artifact ID `10299292416`, SHA256 `0bcbbbb2f3a1fb23b47b9e2ccd5f13dcc393863afa3423444bf325e44cc37ed2`.
