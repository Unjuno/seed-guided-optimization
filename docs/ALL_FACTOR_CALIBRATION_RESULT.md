# All-coordinate physical-moment matched novelty calibration — result

Issue #105 Stage A tested whether gradient novelty can be separated while matching selected-loss mean/shape, scalar total physical diversity, and the means and population variances of all seven standardized environment coordinates.

## Frozen decision

**ALL-FACTOR MATCH CALIBRATION FAIL**

Authoritative GitHub Actions run: `34696143848`.

No intervention arm was trained and no heldout environment was constructed.

## Registered grid

| physical moment caliper | support | minimum replicate support | mean supported G gap | minimum replicate mean G gap | pass |
|---:|---:|---:|---:|---:|---|
| .05 | 0 / 800 = 0 | 0 | 0 | 0 | no |
| .10 | 0 / 800 = 0 | 0 | 0 | 0 | no |
| .20 | 10 / 800 = .0125 | 0 | .047329 | 0 | no |
| .30 | 64 / 800 = .0800 | .025 | .040843 | .001352 | no |

Frozen gates required pooled support >=.90, every-replicate support >=.80, mean supported novelty gap >=.08, and every-replicate mean gap >=.04.

## Interpretation

This is a severe **overlap/identification failure**, not evidence that novelty has zero performance effect. Matching all fourteen first/second physical-coordinate moments is so restrictive in the fixed 455-subset natural candidate family that essentially no usable high/low novelty comparisons remain. Even the widest preregistered physical-moment caliper (.30 standardized units) produces only 8% pooled support and a mean novelty gap around .041.

This resolves the current all-factor matching route negatively: a clean natural-experiment separation of novelty from all registered physical first/second moments cannot be achieved with this candidate generator/subset family. A different experimental intervention or a factorial/synthetic generator would be required to answer that stronger causal question.

Evidence artifact: `all-factor-calibration-evidence`, artifact ID `10299285571`, SHA256 `d5a61ad74680dc9518d270d8cb6b8894c592ba47b453b452c8c846c8610a2b9d`.
