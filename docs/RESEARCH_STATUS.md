# Research status

Updated2026-09-25. Results are conditional on task, comparison, measurement and frozen rule. A PASS, a matching gate, or a closed Issue does not identify a unique mediator. [Research roadmap](RESEARCH_ROADMAP.md) separates completed experiments from unresolved mechanisms and resource-dependent measurements.

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

Independently confirm the clean/tail exploratory contrast on fresh environment pools; then test whether the correctly associated gradient policy retains its advantage under a suitable online loss/rank control. Matching failures require a genuinely identifying design, not threshold relaxation. Follow [the roadmap](RESEARCH_ROADMAP.md) and [bounded research scope](AUTONOMOUS_RESEARCH_SCOPE.md). X announcement remains paused at the owner's request.
