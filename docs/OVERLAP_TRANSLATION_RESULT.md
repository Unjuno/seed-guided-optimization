# Translation-balanced novelty on matched support: Issue91

Frozen decision: **OVERLAP-RESTRICTED TRANSLATION-BALANCED NOVELTY SUPPORT**.
Authoritative run34251284833, sourcebc20ec3516657ec919076f3e2096404262ba89bf, PR92.
Protocol SHA256:e0dfe80f3bae4db248939fd212bad5ada5b96699b379710bc5453c5d7f820688.

## Question and separation from the failed strict design

[Issue89](TRANSLATION_MATCH_CALIBRATION_RESULT.md) failed its all-step calibration.
Its planned strict confirmation was not run. Issue91 is a separately registered
policy: enforce the same matching requirements where an eligible pair exists;
otherwise give both matched arms the IDENTICAL reference loss-hard top4 subset.
No optimizer update is omitted and no caliper is enlarged. The estimand is the
high-minus-low effect of this overlap-restricted schedule policy, not an effect
in every possible reference state or a rescue of Issue89.

## Frozen design and measurement conditions

Fresh reps2200-2229,30 blocks; train seeds63000-63063; heldout64000-64079.
Digits canonical988train/809nontraining images (reused identities), SmallCNN,
AdamW lr.005,weight decay.001,10epochs,batch128,K16/Q4,deterministic1CPU thread.
One common reference trajectory generates84 subsets per step: hardest anchor plus
three of loss ranks2-10. Four intervention schedules are sealed before training.

| Symbol | Meaning (Japanese) | SI unit | Definition | Range/assumption | Type |
|---|---|---|---|---|---|
| U | 平行移動を追加調整しない対照群 |1| high/low G pair matched in H/P only |fixed schedule family|categorical label|
| M | 平行移動も調整する群 |1| high/low G pair matched in H/P/T, common fallback otherwise |fixed schedule family|categorical label|
| G | 勾配方向の平均pairwise novelty |1| average1-cosine of head signatures |finite, nominal0..2|real scalar|
| H | 参照モデル上の平均損失標準化値 |1| selected CE centered/scaled by candidate mean/SD+1e-8 |finite|real scalar|
| P | 総パラメータ距離の標準化値 |1| seven-coordinate mean pair distance, standardized over84 subsets |finite|real scalar|
| T | 平行移動の総分散 |1| population variance of standardized dx plus variance of dy |nonnegative|real scalar|
| B | 反復ごとの平均正答率差 |1| high-arm minus low-arm80-environment mean accuracy |[-1,1],30paired blocks|real scalar|
| c_T | 平行移動の許容差 |1| fixed.20 raw standardized-coordinate variance units |positive|real scalar|

H/P calipers are.05; M also uses T caliper.20, all with1e-12 numerical slack.
U is not unmatched in hardness or total diversity: only translation is unconstrained.
Global reference-support gates precede all120 intervention models. All150 states,
including30reference models, seal before heldout construction.

Matched support was2289/2400steps,95.375%, with a minimum73/80=91.25% within any
replicate. Both exceed the preregistered90%pooled and80%per-replicate gates.
The remaining111steps use common fallback. They remain in the analysis and contribute
zero reference G/H/P/T contrasts. Subsequent arm gradients can differ after divergence.

## Primary results

All intervals below are two-sided95%t29 intervals; p-values are the preregistered
one-sided paired tests. Both effects and all balance/manipulation gates were required.

| Family high-minus-low | Mean benefit | Paired SE |95% CI|one-sided p|positive blocks|
|---|---:|---:|---|---:|---:|
| U: H/P matching only |+3.070818 pp|.503817 pp|[+2.040397,+4.101239] pp|6.13693e-7|26/30|
| M: H/P/T matching and fallback |+2.040585 pp|.576354 pp|[+.861808,+3.219362] pp|.000685123|22/30|

All H/P equivalence gates pass within their frozen margins. M's translation
contrast is+.021821,90% CI[+.017593,+.026049], inside the registered±.20 margin.
This is approximate balance, NOT equality or absence of a detectable residual.
M's hardness difference is-.001799,90% CI[-.003022,-.000577]; scalar parameter
z difference+.000101,90% CI[-.000711,+.000913]. All pass their±.05 TOST gates.

Mean reference G separation is.206506 for U and.100945 for M. Both exceed.05 and
pass the positive-manipulation test. Mean T separation changes from.760933(U) to
.021821(M). This supports a positive schedule contrast under an added translation
balance constraint, but does not eliminate all explanations involving small residual
translation differences, factor composition or the change in attainable G separation.

## Prespecified secondary decomposition

| Comparison | Mean |95% descriptive CI|
|---|---:|---|
| U_high versus reference |+2.523434 pp|[+1.327251,+3.719617]|
| U_low versus reference |-.547384 pp|[-1.838785,+.744018]|
| M_high versus reference |+1.927637 pp|[+.536827,+3.318447]|
| M_low versus reference |-.112948 pp|[-1.443973,+1.218077]|

These are secondary, non-multiplicity-adjusted comparisons; reference and arm
training can span CPUs. The primary+2.040585 pp is NOT an online-gradnov/loss-hard
improvement. Neither low arm is established to differ from its reference here.

The difference of U and M high-low effects is+1.030233 pp,95% CI[-.457078,+2.517543],
descriptive one-sided p.083613. The reduction itself is not established at that
threshold. It must not be turned into a percentage of performance mediated by
translation: support and G manipulation magnitude also changed.

## Deepening the result: residual factor allocation

Prespecified secondary coordinate summaries show M retains a different allocation:

| Coordinate | High-minus-low standardized within-subset variance |95% descriptive CI|
|---|---:|---|
| rotation |+.202555|[+.179467,+.225644]|
| horizontal translation |+.063873|[+.042409,+.085337]|
| vertical translation |-.042052|[-.064063,-.020041]|
| blur |-.077067|[-.094103,-.060032]|
| noise amplitude |-.091188|[-.109816,-.072559]|

All7coordinate means and variances for both families are published in
[translation_coordinate_secondary30.csv](../results/translation_coordinate_secondary30.csv).
These are descriptive, not14or28new confirmatory discoveries. Matching a sum of
translation variances does not fix anisotropy, means, or allocation to rotation.
A specific factor or gradient scalar is still not uniquely identified as the cause.

There is also a measurement distinction: [mean pairwise novelty is a coherence
statistic, not span dimension](NOVELTY_COHERENCE_IDENTITY.md). The algebraic proof
and synthetic counterexample are not a new empirical replication or a generalization
theorem. The averaged intervention score differs from the original greedy online rule.

## Audit / ERROR CHECK / U

Consolidated artifact SHA256:
58a3677ca3e41ac6b0f41d45ff5ad95121def8c541cd02bcc6893b6baecd3f45.
All6reference/arm manifests,source hashes,150checkpoint-file/tensor digests,
30schedules,201600subset rows,4800chosen-pair records and12000environment rows
were independently checked. Re-enumeration reproduced each chosen pair and fallback;
reference diagnostic, coordinate reconstruction and environment reaggregation maximum
errors were0.0. Separate SciPy calculations reproduced the paired t/TOST decisions.
Scientific source bytes match the synthetic-tested local version. No raw-image
inference or training was rerun in this independent audit.

Actions runtime: Python3.12.14,torch2.10.0+cpu,numpy2.3.5,pandas2.2.3,scipy1.17.0,
sklearn1.8.0, MKL2024.2/MKL-DNN3.7.1,one thread,deterministic algorithms.
Intervention runners used AMD EPYC7763 and9V74; reported CPU capability paths include
AVX2/AVX512. All four arms of a replicate share a runner. Clock snapshots were not
fixed-frequency benchmarks. No speedup, GPU or cross-hardware bitwise claim.
Paired SE is conditional on these fixed images/environment samples; a total combined
uncertainty including datasets, sampling of image identities and backends is not
estimated. The95%t29coverage factor is2.04523.

Unit check: CSV accuracy fields are fractions; multiplying by100 converts them to
percentage points. Scores and standardized-coordinate variances are dimensionless,
not physical distances or accuracy percentages.

Reproduce the independent audit from the extracted consolidated artifact:

```bash
python experiments/check_overlap_translation_evidence.py --evidence-dir EXTRACTED --output-dir independent_audit
python experiments/summarize_translation_coordinates.py --input EXTRACTED/aggregate/translation_matched_reference_summary30.csv --output coordinates.csv
python experiments/check_novelty_coherence_identity.py
```

The next distinct question is whether positive effects require independent gradient
span, directional opposition, or other task-relevant schedule properties. That is
not settled by this matched-score experiment and should not be replaced by repeated
post-hoc redefinition of a favorable scalar mediator.
