# Research status

Updated 2026-11-08.

## Fresh early-switch confirmation and clean guardrail, Issue133 (2026-11-08)

[Full prospective result](SWITCH_TIME_EARLY_RESULT_20261108.md) and [pre-outcome Issue133](https://github.com/Unjuno/seed-guided-optimization/issues/133): **12 new independent transform pools×4 new initializations×4 fixed Muon/AdamW policies=192 models**. Globally seal all checkpoints before constructing heldout environments; 38,400 training updates, 9,216 heldout rows, 192 clean rows. Independent SHA+integer-count audit and all registered decisions checked.

- T60 vs Muon throughout: **+0.035709 CE**, pointwise95%CI[+.015015,+.056404], Holm3 one-sided p=.002954 **SUPPORT**.
- T60 vs T40: **+0.020442 CE**, CI[+.012904,+.027980], p=.000140 **SUPPORT**.
- T60 vs T80: **+0.004231 CE**, CI[−.002634,+.011096], p=.101060 **NOT SUPPORTED**. The preregistered T60 interior peak **fails**.
- **CLEAN guardrail FAILS:** CE_clean(T60)−CE_clean(M)=**+.108709 CE**, CI[+.059690,+.157728], registered one-sided noninferiority test with +.05 margin p=.988422. Shifted-CE gains are **not clean-safe**.
- Sample-only, **post-outcome** mixture w between clean and transformed CE: M preferred for w<.691956, T80 for .691956<w<.899812, T60 for w>.899812. These are not learned or validated switching thresholds.

New transformation keys/init seeds only, the same historical Digits test images; no new-image, mixed-task, GPU, equal production compute, universal “optimizer-favorable region” or **seed-selection** claim. The full state/source archive exists in the owner's conversation ZIP, **not GitHub**; public CI validates summary statistics only.

## Preregistered switch-time boundary, Issue131 (2026-11-07)

[Full result and audit](SWITCH_TIME_BOUNDARY_RESULT_20261107.md), [prospective Issue131 with v1 invalid/v2 amended source hashes](https://github.com/Unjuno/seed-guided-optimization/issues/131). Fresh **12 transform pools ×4 new inits ×4 policy arms = 192 states** with unchanged historical Digits image split, globally sealed before heldout creation; 38,400 applied steps; independent source/hash audit, 9,216 heldout rows, 3 preregistered t11/Holm3 tests all reproduced.

- **T100 vs always Muon: +0.025832 CE**, CI [+0.001760,+0.049905], Holm3 p=.037690 **SUPPORT**.
- **T100 vs switch T80: −0.010962 CE**, CI [−0.015430,−0.006494], directional Holm3 p=.999892 **NOT SUPPORTED**; reversed sign in **all 12 pools**, not a preregistered success of T80.
- **T100 vs switch T120: +0.013302 CE**, CI [+0.008738,+0.017867], Holm3 p=.00007473 **SUPPORT**.

The preregistered conjunction claiming a narrow peak at update100 **FAILED**. Descriptive shifted CE is lowest for T80 (1.177579), but clean CE is lowest when Muon is maintained (0.804520). **Changing the evaluation objective changes the preferred policy.** A post-outcome linear mix of clean/shifted CE produces *sample-only* weight boundaries w=.562438/.623979, not a calibrated universal switch criterion. Source/optimizer-handoff code is in owner full-evidence archive, **not mirrored to GitHub**; GitHub checker verifies aggregate statistics only. Prior failed v1 JSON preflight produced **zero models**, preserved separately; v2 amendment preceded all outcomes. No new-image, unique geometry or cost/speed inference.

## New retrospective local Muon/AdamW optimizer-switching pilot (2026-10-31)

[Full 12-pool analysis and limitations](OPTIMIZER_SWITCH_SEED_RESULT_20261031.md), [Issue129](https://github.com/Unjuno/seed-guided-optimization/issues/129). **Locally frozen before training, but NOT externally preregistered; GitHub archive added after results.** All 192 MLP/Digits training trajectories (12 pools ×4 initializations ×4 policies), 576 checkpoints, final counts and five in-session frozen tests were rechecked on 2026-11-06. Remote GitHub contains the summary CSV/checker and retrospective report; **full weights/raw code are held in the originating conversation's ZIP, not mirrored in this repository**.

- Adaptive D vs fixed AdamW A: **+0.069704 CE decrease**, 95% pointwise CI [+0.048237,+0.091171], Holm5 p=0.00004692 **SUPPORT**.
- Adaptive D vs fixed Muon M: +0.008190 CE, CI [−0.002574,+0.018954], p=0.183245 **NOT SUPPORTED**.
- Adaptive D vs one-way fixed Muon→AdamW switch at 100 updates S: **−0.025667 CE**, CI [−0.040767,−0.010567], registered adaptive-superiority **NOT SUPPORTED**. Descriptively S has the lowest mean CE (S1.146800, D1.172467, M1.180657, A1.242171). This alone is NOT fresh confirmatory proof of a generally best switching time.
- Early step40 seed selection within D: −0.003127 CE, p=1 **NOT SUPPORTED**. D+early-selected seed vs baseline selecting a fixed method/seed early: +0.062071 CE, p=0.00091096 **SUPPORT**, but the baseline selected AdamW **12/12 pools**, so no universal seed-selection gain follows.
- **Post-outcome only:** in 48 matched step100 states, five-update control preference chose continued Muon in 20 cases where 100-update continuation favored switching to AdamW (15 strict rank reversals, five hysteresis-margin cases). The controller used 240 vs200 gradient evaluations, plus extra forward/copy work, and both shadow optimizer proposals were evaluated in all policies; **no equal-cost superiority claim**.

The experiment manipulates **optimizer switching and separately early initial-weight seed selection**, not online transformation-seed SGO. No unique favorable optimizer region, optimizer-state mediator, robust universal seed or GPU speedup has been identified.

## New six-pool registered validation: PR #126

[Full frozen result and integrity audit](ROADMAP_VALIDATION_20260926_RESULT.md). The first complete valid run on 2026-09-26 finished all **840** study states; an independent 2026-10-19 audit verified **840 model tensor digests, 840 selection traces, 68,040 integer-count rows and all 12 registered t5/Holm decisions**. The training/evaluation barriers passed before any heldout generation.

- **Issue124, new Digits clean-versus-shift policy comparison:** gradnov minus hardest-plus-random-three gave **+4.302 pp** finite-pool minimum and **−7.505 pp** clean accuracy; both pre-fixed directional tests pass over **six** pool means. This is not universal worst-case robustness.
- **Issue125 Fashion-MNIST transfer (988/809 selected native 28×28 images):** anchored random minus loss-hard heldout mean **+1.201 pp** (Holm3 PASS), gradnov minus anchor minimum **+3.372 pp** (Holm3 PASS), but gradnov minus anchor heldout mean **−0.019 pp** (**not supported**). Thus positive conditional *minimum* results extend to another image dataset, not a full-data-set benchmark or uniqueness result.
- **Issue125 Digits optimizer factorial:** full-16 versus four-contributor estimator main contrasts PASS for AdamW (**+4.698 pp**) and SGD configuration (**+1.164 pp**), but are **not equal-compute**. AdamW objective×estimator interaction **−1.948 pp** passes Holm7; SGD interaction **−2.054 pp** does **not** pass Holm7; three-way interaction **−0.106 pp** does not pass. Configuration-level comparison does not isolate optimizer-algorithm causes.

See [public pool-level CSVs](../results/roadmap_validation_20260926_pool_contrasts.csv), [frozen primary decisions](../results/roadmap_validation_20260926_primary.csv) and `python scripts/check_roadmap_validation_results.py` for statistic-only replication. The full GitHub Actions artifacts have finite retention and must not be mistaken for permanent storage.

## Latest completed online control experiment

[Issue121 online controls](ONLINE_CONTROL_RESULT.md), first valid run36137245500, used60fresh training blocks across three fixed environment pools. All240states were globally sealed before heldout construction. Each policy selected on its own current model state: original gradnov, loss-hard, hardest-plus-random-three, and gradnov with shuffled gradient/candidate correspondence.

| Gradnov minus comparator | Mean effect (pp) | Two-sided95% CI (pp) | Holm-adjusted one-sided p |
|---|---:|---|---:|
| loss-hard | +2.671688 | [+2.019792,+3.323585] | 3.759003e-11 |
| anchored random | +0.677019 | [-0.055184,+1.409222] | 0.034648 |
| shuffled-gradient gradnov | +2.070277 | [+1.434274,+2.706281] | 1.791759e-8 |

Frozen outcome: **ONLINE GRADIENT-INFORMATION CONTROL SUPPORT**. All3registered one-sided criteria pass after Holm adjustment. The anchored-random contrast is much weaker; its marginal two-sided interval crosses zero and its pool0 point estimate is negative. Do not claim uniform pool superiority or a strictly positive two-sided interval for that comparator.

Correct gradient/environment correspondence contributes to the observed online policy advantage versus the tested sham policy. This does not isolate the unique underlying geometric cause: policies can differ in selected hardness, rank, physical allocation and update trajectories. These data are Digits-specific and use the same historical image split, not new-image or cross-dataset validation.

Independent audit checked440hashes,240state tensor digests,19,200environment rows and19,200training selection records and separately reproduced the paired statistics/Holm decision with zero discrepancy. It did not retrain or reconstruct raw-gradient selector optimality. [Public CSV checker](../scripts/check_online_controls.py) is lighter still: statistics only.

## Mechanism experiments and negative outcomes retained

| Study | Frozen outcome | Scope |
|---|---|---|
| Issue100 centered residual span | NO CENTERED GRADIENT-SPAN PERFORMANCE SUPPORT | centered rank manipulation succeeded; mean effect+0.2072pp,95%CI[-0.6916,+1.1060],one-sided p0.320402; not proof of zero effect |
| Issue103 opposition calibration | OPPOSITION CALIBRATION FAIL | even widest frozen span caliper gave87.25%support and insufficient opposition gap; no performance test |
| Issue105 all-factor moment matching | ALL-FACTOR MATCH CALIBRATION FAIL | only64/800steps supported at widest frozen physical caliper; no performance test |
| Issue104 matched rank/hardness relaxation | MATCHED RANK-RELAXATION SUPPORT | initial30-block mean+0.9066pp; actual selected hardness also changes |
| Issues112/113 rank-versus-hardness separation | respective CALIBRATION FAIL decisions | current natural455-subset family did not provide the required isolated manipulation; not evidence that either factor is irrelevant |
| Issue115 fresh mean-tail replication | NO MEAN-TAIL TRADEOFF REPLICATION | all matching/manipulation gates pass; neither primary directional performance criterion passes |

[Issue115 result and full independent audit](RANK_TAIL_TRADEOFF_RESULT.md):60fresh blocks, mean high-rank minus low-rank+0.297976pp,95%CI[-0.242477,+0.838429],positive-direction p0.137202; minimum-accuracy difference-0.070045pp,CI[-0.925578,+0.785487],negative-direction p0.435213. The initial Issue104 finding is not erased, but its mean benefit did not meet the threshold in this fresh replication. Neither nonsignificance nor an audit PASS establishes equal performance or tail safety.

The Issue115 audit independently reconstructed2,184,000subset rows,4,659optimal pairs and141fallbacks across4,800steps,660source/artifact hashes,180state tensor digests and14,400environment rows. No raw-image inference, raw-gradient reconstruction or retraining was performed in that audit.

## Exploratory discoveries: not promoted to confirmation

Issue115's clean-image accuracy difference was+3.1706pp and environment-accuracy SD difference+0.7196pp despite the failed primary mean/minimum criteria. In the new online trial, gradnov versus anchored-random gave clean accuracy-7.1487pp, minimum over80heldout environments+3.7536pp, and across-environment SD-2.0288pp. These secondary observations motivate a fresh clean-versus-transformation/tail test. They do not retroactively validate Issue115 or establish an adversarial/worst-case guarantee.

## Earlier evidence retained

| Study | Outcome and limits |
|---|---|
| Issue89 strict translation calibration | TRANSLATION MATCH CALIBRATION FAIL; no intervention or heldout performance test |
| Issue91 overlap-restricted schedules | matched high-low+2.040585pp,95%CI[+0.861808,+3.219362],p0.000685123; matching is approximate, not all-factor causal identification |
| Issue83 parameter-matched schedules | high-low+3.969922pp; scalar total diversity and reference hardness matched, factor composition not matched |
| Issue86 spatial-only scoring | gradient high-low+3.852029pp and translation-only high-low+1.768644pp; both alter gradient geometry |
| Issue80 loss-stratified schedules | high-low+4.66929pp; physical diversity also increased |
| Original CIFAR40-pair online comparison | mean+0.1206pp,Holm(5)p0.01336; tail benefit not established |

Issue91 and the other high/low reference-defined histories are OFFLINE schedule interventions, not ordinary online gradnov-versus-loss-hard benchmarks. Numerical effects across these comparisons are not directly comparable. Issue91 retained rotation and individual translation-allocation differences; neither a translation-mediated percentage nor a unique gradient mediator was established. Original failure records and numerical-audit corrections in Issues96/98/100 remain historical evidence, not rescored successes.

Five preregistered30-block budget tests remain unchanged: Digits/MLP first attenuation+2.034pp,p1.06e-7; freshMLP+2.086pp,p5.16e-7; SmallCNN+1.265pp,p0.001548; FashionMNIST/TinyTransformer+0.314pp,p0.027264 (two-sided95%CI crosses zero); CIFAR/ResNet20+0.11679pp,p0.003804. Curves are not universally monotone. The all-candidate identity is a structural control, not causal proof.

[Pairwise novelty algebra](NOVELTY_COHERENCE_IDENTITY.md) distinguishes directional coherence from independent span. It applies to the intervention's mean pairwise statistic, not identically to the original greedy selector. Raw representation rank is not an intrinsic function-level mediator; standardized-rank coupling and other registered explanatory tests failed. Pure diversity, accumulated gradient rank and immediate loss reduction do not suffice as established general explanations.

## Implementation and uncertainty

Scientific runs use Python3.12.14,torch2.10.0+cpu,numpy2.3.5,pandas2.2.3,scipy1.17.0,sklearn1.8.0, one deterministic CPU thread; actual CPU/backend/clock snapshots are archived per run. New online controls use batch128/last92,10epochs,80updates,16candidates/4adopted, float32training and float64statistics. Within-block policy training shares a runner. Clocks are uncontrolled; no speed claim is made.

Blocks, not environments or steps, are statistical repetitions. Images and specified pools are reused/fixed. Paired SE and the t interval are conditional, not combined uncertainty across datasets, unseen pool distributions and hardware. Historical cross-hardware drift remains unresolved. A minimum over80sampled environments is not universal worst-case performance. GPU matched-cost measurement, broad task universality, calibrated per-run gating and unique-mechanism identification remain open.

## Next discrimination and publication

The clean/minimum two-policy tradeoff is now **confirmed only within the registered finite Digits generator/pools**; a new limited Fashion task additionally supports finite-pool minimum contrast but not mean superiority of gradnov over the anchored policy. Next identify policy differences in selected loss/rank, factor allocation and resulting update trajectories with a fresh feasible intervention; the current results do not identify a unique mediator. Broad dataset/task universality, adversarial worst-case guarantees, equal-wall-clock performance and GPU-optimal candidate counts remain unresolved. Follow [the roadmap](RESEARCH_ROADMAP.md). X announcement remains paused at the owner's request.
