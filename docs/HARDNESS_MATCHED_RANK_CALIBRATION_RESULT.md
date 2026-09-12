# Hardness-matched loss-rank calibration — result

Issue #112 Stage A asked whether mean candidate loss rank R can be manipulated while selected standardized mean CE hardness H is explicitly matched, together with L/P/T/G/centered-span/opposition.

## Frozen decision

**HARDNESS-MATCHED RANK CALIBRATION FAIL**

Authoritative GitHub Actions run: `34697366689`.

No intervention arm was trained and no heldout environment was constructed.

## Registered grid

| H caliper | support | minimum replicate support | mean supported R gap | minimum replicate mean R gap | pass |
|---:|---:|---:|---:|---:|---|
| .02 | .54500 | .45 | .303326 | .266667 | no |
| .05 | .75250 | .70 | .388704 | .352679 | no |
| .10 | .84375 | .75 | .521852 | .483333 | no |
| .20 | .89375 | .80 | .776573 | .693662 | no |

Frozen gates required pooled support >=.90, every-replicate support >=.80, mean rank gap >=1.00, every-replicate mean rank gap >=.50, plus identity residual <=1e-10. At the widest registered H caliper (.20), minimum replicate support reached the gate but pooled support remained below .90 and the mean rank gap remained below 1.00.

The other registered matching quantities stayed within their per-step calipers among supported pairs. Maximum corrected centered-energy identity residual was `3.997e-15`.

## Interpretation

This is an identification/overlap failure, not a heldout null result. Under the current natural 455-subset family, a large Issue104-scale shift in candidate rank cannot be separated from selected mean CE hardness while also controlling the registered physical and gradient-geometry summaries.

Together with Issue104's confirmed performance benefit and its `H=-0.310 z` co-movement, this strengthens the conclusion that candidate rank and actual selected hardness form a tightly coupled intervention axis in this design. The complementary Issue #113 tests whether hardness can be separated while rank is held approximately fixed.

Evidence artifact: `hardness-matched-rank-calibration-evidence`, artifact ID `10299167753`, SHA256 `d2894e270abca12be2fab3950227f40d9407ba1ed16290877a69847d2f26a7bc`.
