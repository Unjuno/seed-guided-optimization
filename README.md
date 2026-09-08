# Seed-Guided Optimization

**Gradient-aware selection of stochastic training environments under a finite update budget.**

Seeds index environments; no intrinsic good seed classes or universal seed quality are assumed. SGO studies whether retaining hard environments while reducing redundancy among model-conditioned gradient signatures improves allocation of a fixed subset-update budget.

> **Status — 2026-09-09:** research code with replicated effects in tested regimes, not a universal optimization law. A new30-repetition schedule intervention retains a positive high-minus-low effect after approximately matching reference hardness, scalar total transformation diversity and translation variance on available matched support. Strict all-step matching first failed calibration and remains a negative result. A unique gradient mediator and the downstream functional mechanism remain unidentified.

## Latest completed experiments

| Study | Result | Scope |
|---|---|---|
| Issue89 strict translation calibration | FAIL; largest caliper supports799/800steps | no intervention arms or heldout evaluation |
| Issue91 overlap-restricted matching,30fresh repetitions | matched high-low **+2.0406 pp**,95%CI[+.8618,+3.2194],p.000685; H/P-only control **+3.0708 pp**,CI[+2.0404,+4.1012],p6.14e-7 | both primary tests and all gates pass; unsupported steps use identical subsets, not looser matching |
| Issue83 parameter-matched schedules | high-low **+3.9699 pp**,95%CI[+3.0506,+4.8892] | hardness and scalar total diversity matched; factor composition not matched |
| Issue86 spatial scoring alternative | gradient high-low **+3.8520 pp**, translation-only high-low **+1.7686 pp** | both scoring procedures create useful contrasts but both alter gradient geometry |

See [strict calibration failure](docs/TRANSLATION_MATCH_CALIBRATION_RESULT.md), [new matched-support result](docs/OVERLAP_TRANSLATION_RESULT.md), [research status](docs/RESEARCH_STATUS.md), and [theory](docs/THEORETICAL_FRAMEWORK.md).

## New result and interpretation boundary

Issue91 fixes translation-caliper.20 and retains all80updates per repetition. Where matching is impossible, both matched schedules use the same reference loss-hard top4. Support is95.375%pooled, minimum91.25%per repetition;111of2400steps use fallback. All schedules seal before intervention training and all150models seal before heldout construction.

The effect survives these constraints. However translation-score difference is+.021821, not zero; rotation variance and individual dx/dy allocations remain different. Gradient separation also shrinks from.206506 to.100945. The control-minus-matched benefit difference is+1.0302 pp with95%CI[-.4571,+2.5175]. Its reduction is not established and cannot be called a translation-mediated fraction.

These are offline shared-reference schedules, not the original online gradnov-versus-loss-hard comparison. Matched-high versus reference is secondary+1.9276 pp; the primary+2.0406 pp compares matched-high with matched-low. Reference comparisons may cross training CPUs and are not equal-total-compute tests.

A [full algebraic note](docs/NOVELTY_COHERENCE_IDENTITY.md) shows that mean pairwise novelty measures normalized directional coherence, not independent span dimension. Opposite gradients on one axis and balanced gradients spanning two axes can have equal scores. This is a measurement clarification, not a performance theorem; the averaged intervention score differs from the original greedy online selector.

## Earlier evidence retained

| Evidence | Result and scope |
|---|---|
| CIFAR primary,40paired online runs | mean+.1206 pp,Holm(5)p.01336; tails unconfirmed |
| Digits budget tests | MLP attenuation+2.034 and+2.086 pp; SmallCNN+1.265 pp |
| FashionMNIST/Tiny Transformer budget test | attenuation+.314 pp,one-sided p0.0273; two-sided CI crosses zero: borderline |
| CIFAR/ResNet-20 budget test | attenuation+.1168 pp,95%CI[+.0335,+.2000],p.00380 |
| Endpoint sensitivity | post-hoc removal of structural zero retains positive CIFAR contrast, not established SmallCNN contrast |
| Fixed-Q Issue80 | high-low+4.669 pp; total physical diversity also shifts |
| Optimizer/architecture comparisons | gains beyond one MLP/AdamW setup in tested regimes |
| Rank interventions | raw rank coordinate-dependent; standardized-rank budget mediation failed |
| Hosted CPU audit | one thread did not remove cross-run CIFAR drift |

All-candidate identity in five budget blocks is a structural control, not standalone causal proof. Curves are not universally monotone. Fresh seeds are not new image identities. Reference equivalence margins do not guarantee exact equality, full factor balance or equality along subsequently diverging arm trajectories. No universal seed quality, unique scalar mediator, calibrated per-run gate, general tail safety, large-Transformer validity, cross-hardware bitwise equality or GPU speedup is established.

## Reproduction and evidence

Accuracy CSV fields are fractions:0.01 is one percentage point. Workflows pin runtime and archive source, subset summaries, schedules, states and manifests. Read-only checks use the extracted complete artifacts:

```bash
python experiments/audit_translation_support.py --evidence-dir CALIBRATION_EXTRACTED --output-dir calibration_audit
python experiments/check_overlap_translation_evidence.py --evidence-dir CONFIRMATION_EXTRACTED --output-dir confirmation_audit
python experiments/check_novelty_coherence_identity.py
```

[Primary paired data](results/translation_matched_primary30.csv), [all aggregate endpoints](results/translation_matched_summary30.csv), [equivalence gates](results/translation_matched_balance30.csv), and [coordinate descriptions](results/translation_coordinate_secondary30.csv) are public. Older studies remain in the [experiment index](experiments/README.md) and [document index](docs/README.md).

## Next discriminating question

Can independent gradient span, directional opposition and raw-update magnitude be distinguished while preserving task-relevant schedule conditions? Another positive average-cosine contrast alone will not resolve that question.

## Citation and license

Citation metadata: [CITATION.cff](CITATION.cff). License: [Apache-2.0](LICENSE).
