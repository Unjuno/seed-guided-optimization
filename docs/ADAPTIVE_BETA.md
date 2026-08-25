# Adaptive novelty pressure

The gradient-novel selector scores candidates with

`z(loss) + beta * z(gradient novelty)`

while retaining one hard anchor. Earlier fixed-beta sweeps showed a real mean/robustness trade-off: stronger novelty pressure reduces selected-gradient redundancy and raises held-out mean accuracy, but very high beta increases environment-to-environment variance.

## Feedback controller

This experiment asks whether beta must be manually fixed. The controller starts at `beta=1.0`, observes the mean pairwise cosine of the four selected head-gradient directions, and updates

`beta <- clip(beta * exp(0.35 * (selected_pair_cos - target)), 0.10, 5.0)`.

If selected gradients remain too redundant relative to the target, novelty pressure rises; if they become more diverse than necessary, novelty pressure falls. All methods use the same initialization, minibatch order, candidate environments, optimizer budget, and 80 held-out environment seeds within each paired replicate.

## Results: 20 paired replicates

| method | held-out mean | env SD | p10 | minimum | mean beta | selected pair cosine |
|---|---:|---:|---:|---:|---:|---:|
| beta=0 | 48.68% | 8.62 pp | 37.63% | 25.90% | 0.00 | 0.409 |
| beta=1.5 | 54.44% | 8.98 pp | 42.61% | 29.03% | 1.50 | 0.184 |
| beta=3.0 | 55.77% | 9.77 pp | 42.84% | 28.65% | 3.00 | 0.154 |
| adaptive target 0.20 | 53.60% | 8.63 pp | 42.10% | 28.71% | 1.87 | 0.188 |
| adaptive target 0.15 | 54.49% | 9.08 pp | 42.59% | 29.25% | 2.85 | 0.165 |

Adaptive target 0.15 and fixed beta=1.5 are statistically indistinguishable on all five held-out metrics after five-metric Holm correction. This shows that a simple training-side feedback signal can recover the robust fixed-beta operating point without directly optimizing held-out accuracy.

Fixed beta=3 improves held-out mean over beta=1.5 by +1.33 pp (`Holm p=0.0157`) but also increases environment SD by +0.78 pp (`Holm p=0.000686`). Relative to beta=3, adaptive target 0.15 lowers environment SD by 0.69 pp (`Holm p=0.00484`) while p10, minimum and clean accuracy do not differ significantly; its mean is 1.28 pp lower and does not survive five-metric Holm correction (`Holm p=0.0652`).

## Interpretation

`beta` is not just a nuisance hyperparameter. It is a control variable governing a measurable trade-off between local hardness and directional redundancy. A feedback policy can target a desired gradient-space redundancy level, suggesting a control-theoretic formulation of Seed-Guided Optimization.

This does **not** establish a universal cosine target. The target 0.15/0.20 values were tested only on the structured Digits geometric-shift benchmark. Cross-task calibration and a controller that derives its target without prior benchmark knowledge remain open.
