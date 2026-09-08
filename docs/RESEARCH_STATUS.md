# Research status

Updated2026-09-09. Results are conditional on task, comparison, measurement and frozen rule. Neither a PASS nor a matching gate identifies a unique mediator.

## Latest completed sequence

| Study | Frozen outcome | Scope |
|---|---|---|
| Issue89 strict translation calibration | TRANSLATION MATCH CALIBRATION FAIL | largest fixed caliper admits799/800steps; no intervention arm or heldout evaluation; planned strict confirmation not run |
| Issue91 fresh overlap-restricted intervention | OVERLAP-RESTRICTED TRANSLATION-BALANCED NOVELTY SUPPORT | matched high-low+2.040585 pp,95%CI[+.861808,+3.219362],p.000685123; H/P-only control+3.070818 pp,CI[+2.040397,+4.101239],p6.13693e-7; both primary conditions and all gates pass |
| Issue83 StageB | PARAMETER-MATCHED GRADIENT NONREDUNDANCY SUPPORT | high-low+3.969922 pp; reference hardness and scalar total diversity matched, factor composition not matched |
| Issue86 spatial scoring | SPATIAL ALLOCATION ALTERNATIVE SUPPORTED | gradient high-low+3.852029 pp; translation-only high-low+1.768644 pp; both change gradient geometry |
| Issue80 loss-stratified schedules | LOSS-STRATIFIED NONREDUNDANCY SUPPORT | high-low+4.66929 pp; physical diversity also increased |

See [strict failure](TRANSLATION_MATCH_CALIBRATION_RESULT.md), [fresh matched result](OVERLAP_TRANSLATION_RESULT.md), [parameter matching](PARAMETER_MATCHED_RESULT.md), and [spatial alternative](SPATIAL_ALLOCATION_RESULT.md).

## What changed

Strict all-step matching failed because some reference states lack an eligible pair, not because a heldout effect was tested and rejected. An independent enumeration of67200subset summaries reproduced the failure. At the worst pilot step, the smallest possible translation difference under the original hardness/diversity calipers was.579288, outside the largest frozen.50caliper.

Issue91 is a DIFFERENT preregistered policy using fresh reps2200-2229 and environments. It fixes translation caliper.20 and uses an identical loss-hard subset in both matched arms where no pair exists. All80updates remain. Pooled matched support2289/2400=95.375%, minimum per-rep91.25%, passed the frozen90%/80%gates before any arm trained. This is not a rescue or relabeling of Issue89.

The matched schedule effect remains positive. However M's translation contrast is+.021821 with90%CI[+.017593,+.026049]: within the±.20equivalence margin, not exactly zero. It retains rotation variance+.202555 and different dx/dy allocation. Both reference gradient separation and eligible support differ from the H/P-only control. Thus translation matching strengthens the procedural evidence, not unique-gradient causal identification.

The U-minus-M performance contrast is+1.030233 pp,95%CI[-.457078,+2.517543],descriptive p.083613. Neither this reduction nor a translation-mediated percentage is established.

## Baselines and measurement

The primary quantities compare high and low OFFLINE reference-defined schedules, not online gradnov versus loss-hard. Issue91 M_high versus reference is a secondary+1.927637 pp,95%CI[+.536827,+3.318447]; M_low versus reference-.112948 pp,CI[-1.443973,+1.218077]. Reference comparisons may cross training CPUs and are not equal-total-compute tests.

A new [algebraic clarification](NOVELTY_COHERENCE_IDENTITY.md) shows that average pairwise novelty at fixed subset size and unit norms is determined by the norm of the mean normalized signature. Equal scores can have different span dimensions. This is not a new mathematical theorem or an empirical performance result. It applies to the intervention's mean-pairwise score, not identically to the original greedy online selector.

## Earlier evidence retained

Five preregistered30-repetition budget blocks remain unchanged:

| Task/model | Low-minus-high attenuation | One-sided p | Scope |
|---|---:|---:|---|
| Digits/MLP first |+2.034 pp|1.06e-7|positive registered contrast|
| Digits/MLP fresh |+2.086 pp|5.16e-7|within-family replication|
| Digits/SmallCNN |+1.265 pp|.001548|cross-architecture within Digits|
| FashionMNIST/Tiny Transformer |+.314 pp|.027264|borderline;95%two-sided CI crosses zero|
| CIFAR-10/ResNet-20 |+.11679 pp|.003804|larger-task replication|

All-candidate model identity is structural control, not standalone causal proof. Excluding that endpoint post-hoc retains positive CIFAR attenuation but not an established SmallCNN contrast. Curves are not universally monotone.

The original CIFAR40-pair online test supports mean+.1206 pp with Holm(5)p.01336; tails are unconfirmed. Optimizer/architecture comparisons, RNG compression and learned-coordinate studies remain in their original documents. Raw rank is not a functionally intrinsic mediator and standardized-rank coupling failed. Pure diversity, accumulated gradient rank and best immediate loss reduction are insufficient explanations. Structured/nuisance matching did not establish reusable-factor causality. MLP full-minus-clean specificity did not transfer to SmallCNN.

## Verification and limits

Issue91:150checkpoint-file/tensor digests,30schedules,201600subset rows,4800reference-pair rows and12000environment rows independently verified. Re-enumeration reproduced choices and fallback. Maximum reference-score, coordinate-reconstruction and environment-aggregation errors0.0. Separate paired t/TOST checks match the frozen decision. No raw-image inference or retraining in the independent audit.

Science used Python3.12.14,torch2.10.0+cpu,numpy2.3.5,pandas2.2.3,scipy1.17.0,sklearn1.8.0,one deterministicCPU thread. Intervention CPUs were AMD EPYC7763/9V74; all four intervention arms share a runner within rep. Clock frequencies are uncontrolled and there is no speed/GPU claim.

The30training blocks, not environments or steps, are statistical repetitions. Images and environment pools are fixed. Paired SE is not total combined uncertainty across datasets/images/backends. Historical cross-hardware drift remains unresolved. New tail safety, calibrated per-run gating, broad task universality and matched-cost practical value are not established.

## Next discrimination

The positive effect survives approximate balance on three reference summaries over available matched support. It does not show whether the useful property is gradient span, directional opposition, remaining physical-factor allocation or another schedule feature. A next experiment should distinguish these properties rather than rename the same average cosine score or convert correlated treatment differences into a mediated fraction.
