# Relative gradient-redundancy control

An absolute target for selected-gradient pairwise cosine does not transfer cleanly across tasks. On the Digits geometric-shift benchmark, an absolute target near 0.15 was feasible; on the independent synthetic task, selected gradients are intrinsically much more correlated and the same controller saturated at the maximum novelty weight.

This experiment replaces the absolute target with a task- and step-normalized target derived only from the current candidate gradients.

For each step, let `c0` be the pairwise cosine of the four environments selected with `beta=0` (hardness only) and let `c5` be the cosine under strong novelty pressure (`beta=5`). Define

`c_target = c5 + rho * (c0 - c5)`.

We then choose beta from a small fixed grid to make the selected-gradient cosine closest to `c_target`. No held-out environment accuracy is used to choose beta. The same `rho=0.15` is transferred without retuning between Digits and an independently generated synthetic classification task.

## Digits geometric shift — 20 paired replicates

| method | mean | env SD | p10 | minimum | mean beta | selected cosine |
|---|---:|---:|---:|---:|---:|---:|
| beta=0 | 49.83% | 8.78 pp | 38.33% | 25.07% | 0.00 | 0.405 |
| beta=1.5 | 54.63% | 9.00 pp | 43.14% | 29.12% | 1.50 | 0.179 |
| beta=3 | 55.32% | 9.38 pp | 43.02% | 29.40% | 3.00 | 0.149 |
| relative rho=0.15 | 54.46% | 9.28 pp | 41.91% | 28.99% | 1.81 | 0.178 |
| relative rho=0.30 | 53.49% | 9.20 pp | 41.75% | 28.10% | 1.38 | 0.211 |

Relative rho=0.15 vs beta=0 improves mean by +4.63 pp, p10 by +3.58 pp, and minimum by +3.92 pp; all three survive five-metric Holm correction. Relative rho=0.15 is not significantly different from fixed beta=1.5 on any of the five held-out metrics after correction. Relative rho=0.30 is weaker: its mean is 1.14 pp below beta=1.5 (Holm p=0.0278).

## Independent synthetic task — 20 paired replicates

| method | mean | env SD | p10 | minimum | mean beta | selected cosine |
|---|---:|---:|---:|---:|---:|---:|
| beta=0 | 85.94% | 1.45 pp | 84.13% | 82.12% | 0.00 | 0.755 |
| beta=1.5 | 85.97% | 1.44 pp | 84.14% | 82.06% | 1.50 | 0.693 |
| beta=3 | 85.96% | 1.44 pp | 84.11% | 82.03% | 3.00 | 0.685 |
| relative rho=0.15 | 85.98% | 1.45 pp | 84.13% | 82.07% | 2.06 | 0.688 |
| relative rho=0.30 | 85.95% | 1.44 pp | 84.13% | 81.93% | 1.66 | 0.696 |

The synthetic task has much weaker selector effects overall. Relative rho=0.15 is statistically indistinguishable from beta=1.5 and beta=3 across all five held-out metrics after Holm correction, while avoiding the pathological beta=5 saturation produced by the absolute-cosine controller.

## Interpretation

The transferable quantity is not an absolute cosine value. Gradient geometry is task/model dependent. A more defensible control variable is **relative position within the currently attainable hardness-to-novelty redundancy range**. This preserves the control-theoretic interpretation of SGO while removing one task-specific absolute scale.

This is still not evidence that `rho=0.15` is universal. Two tasks support it as a useful cross-task operating point; more architectures and environment generators are required.
