# Corrected centered residual-span confirmation — result

Issue #100 Stage B was run only after the corrected Stage-A calibration passed at the frozen `c_G=.05`.

## Frozen decision

**NO CENTERED GRADIENT-SPAN PERFORMANCE SUPPORT**

Authoritative GitHub Actions run: `34584280149`.

The reference barrier passed before either intervention arm trained: pooled supported-step fraction `0.9616667` across 2400 reference steps, minimum replicate support `0.9125`, with 92 identical-subset fallback steps. The arm barrier then sealed 60 intervention states plus 30 reference states before heldout environments were constructed.

## Manipulation and balance

The centered residual spectral effective-rank manipulation was large and consistent:

- mean high-minus-low `S_c` gap: `0.4033064` effective-rank units;
- 95% CI `[0.3941041, 0.4125088]`;
- one-sided paired p=`2.67e-37`;
- positive pairs: `30/30`.

All preregistered H/L/P/T/G balance gates passed. Mean contrasts were:

- hardness H: `-0.0004392`;
- selected-loss variance L: `-0.0001235`;
- scalar physical diversity P: `+0.0001578`;
- translation allocation T: `-0.0008254`;
- mean pairwise gradient novelty G: `+0.0025072`, within the frozen +/-0.05 equivalence margin.

The corrected general centered-energy identity residual was at most `4.88e-15`.

## Primary heldout result

Primary quantity: heldout mean accuracy(high-S_c) minus heldout mean accuracy(low-S_c), one value per independent training block (`n=30`).

- mean: `+0.0020720` accuracy fraction = **+0.2072 pp**;
- paired SE: `0.0043944`;
- two-sided 95% CI: `[-0.0069155, 0.0110596]` = **[-0.6916, +1.1060] pp**;
- preregistered one-sided paired p=`0.320402`;
- positive pairs: `19/30`.

The performance gate therefore failed. The experiment does **not** support the hypothesis that, once mean novelty and the registered physical/loss summaries are balanced, increasing centered residual span is sufficient to reproduce the previously observed positive schedule effect.

This is not evidence that the true effect of span is exactly zero. It is evidence against the registered positive-effect claim at this effect scale and protocol.

## Important post-hoc observation

Both intervention arms were descriptively above the loss-hard reference in mean heldout accuracy:

- high-S_c minus reference: `+2.0739 pp`, 95% CI `[+1.0129, +3.1348] pp`;
- low-S_c minus reference: `+1.8667 pp`, 95% CI `[+0.8309, +2.9025] pp`.

These comparisons were not the primary confirmatory test and reference versus intervention states were produced in different workflow jobs/runners, so they are not treated as a causal result. However, archived schedules show that both high and low arms chose candidates with mean loss rank about `7.13`, while loss-hard top4 has mean rank `2.5`; roughly 62% of selected candidates were outside the top4 hardest candidates. This motivates a new, separately preregistered test of **hardness/rank allocation** rather than reinterpreting the failed span hypothesis.

## Scope

The result is conditional on Digits/SmallCNN, AdamW, Q=4, K=16, the expanded 455-subset family, the fixed head-gradient representation, and the frozen matching policy. It does not rule out other notions of gradient geometry, directional opposition, task-aligned directions, individual physical-factor allocation, or dynamic online effects.

Evidence artifact: `centered-span-confirmation-evidence`, artifact ID `10193177786`, SHA256 `1b7309242ce4f63d0f8f57b965d5ac3e1316491d15eaefa7431fa4257a2b653d`.
