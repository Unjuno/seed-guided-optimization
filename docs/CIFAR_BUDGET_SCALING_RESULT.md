# CIFAR-10 / ResNet-20 finite-budget Q-scaling result

## Status

Preregistered Issue #76 is complete. The authoritative scientific execution is recovery run **34028771645**, which preserved 25 already sealed replicate results from the original run and retrained only the interrupted reps 60-64 before any held-out evaluation was permitted.

Frozen decision: **CIFAR RESNET FINITE-BUDGET COVERAGE REPLICATES**.

This result is a finite-budget Q-scaling test, not a representation-rank mediator test and not a tail-safety confirmation.

## Frozen protocol

- CIFAR-10 deterministic stratified subset: 6000 training images, 3000 test images;
- GroupNorm ResNet-20;
- clean pretrain: 10 epochs, AdamW lr 3e-3, weight decay 1e-4;
- environment finetune: 2 epochs, AdamW lr 1e-3, weight decay 1e-4;
- K=8 candidate environments;
- Q in {2,4,6,8};
- loss-hard versus gradnov, novelty weight 0.6;
- paired reps 50-79, n=30;
- fresh train environment seeds 50000-50063;
- fresh held-out environment seeds 60000-60031;
- four CPU threads, deterministic PyTorch algorithms.

For each replicate and Q, `B[Q] = mean held-out accuracy_gradnov - mean held-out accuracy_loss_hard`.

The preregistered primary statistic was

```text
B_low  = mean(B_Q2, B_Q4)
B_high = mean(B_Q6, B_Q8)
A_B    = B_low - B_high
```

Coverage replication required `mean(A_B)>0` and one-sided paired t-test `p<0.05`. At Q=8 both methods had to be exactly identical in canonical model-state digest, non-timing training diagnostics, held-out metrics and clean accuracy.

## Frozen result

| Quantity | Result |
|---|---:|
| n | 30 paired reps |
| mean low-Q benefit | +0.13675 pp |
| mean high-Q benefit | +0.01997 pp |
| attenuation `A_B` | **+0.11679 pp** |
| paired SE | 0.04071 pp |
| two-sided t29 95% CI | **[+0.03353,+0.20005] pp** |
| one-sided p | **0.00380408** |
| positive attenuation pairs | 22/30 |
| Q=8 identity | **30/30 exact** |
| Q=8 mismatch rows | 0 |

Accuracy values in CSVs are fractions; the table converts them to percentage points.

## Q-wise held-out mean benefit

| Q | coverage Q/K | gradnov - loss-hard mean accuracy | positive pairs |
|---:|---:|---:|---:|
| 2 | 0.25 | **+0.19597 pp** | 23/30 |
| 4 | 0.50 | **+0.07753 pp** | 21/30 |
| 6 | 0.75 | **+0.03993 pp** | 21/30 |
| 8 | 1.00 | **0 exactly** | 0/30 |

Unlike the preceding SmallCNN and Fashion blocks, this CIFAR mean curve is monotone in the observed block. That is descriptive only; universal monotonicity is not claimed.

The selected pairwise-gradient novelty manipulation also contracts as Q grows:

- Q2: +0.10039;
- Q4: +0.03241;
- Q6: +0.01247;
- Q8: 0 exactly.

## Q=K structural control and endpoint sensitivity

Q=8 is an implementation control: both selectors use every candidate in the same sorted order, so exact learned-state identity is expected if the pipeline is correct. It should not be treated as standalone proof that gradient coverage is the causal mediator.

To assess dependence on that structural endpoint, a **post-hoc sensitivity analysis** removes Q=8 and compares the same low-Q mean with Q=6 alone:

```text
mean(B_Q2,B_Q4) - B_Q6 = +0.09682 pp
SE                         = 0.04575 pp
95% t29 CI                 = [+0.00324,+0.19040] pp
one-sided descriptive p    = 0.02152
positive pairs             = 20/30
```

This comparison was not the preregistered primary and cannot replace it. However, unlike the analogous SmallCNN endpoint sensitivity, the CIFAR attenuation remains positive with a two-sided interval just above zero after excluding the forced Q=K identity point. This materially reduces concern that the frozen PASS is only an arithmetic consequence of averaging in an exact-zero endpoint.

## Secondary metrics

Q-wise clean and tail metrics are retained in `results/cifar_budget_q_summary30.csv`. Some uncorrected secondary differences are positive, including several p10/minimum effects. They are not multiplicity-controlled primary endpoints, so this experiment does **not** establish CIFAR tail safety.

## Recovery and verification

The original run 34004715086 timed out in the 60-64 training shard after reps 60-63 had been partially computed. No held-out evaluation had begun. The recovery policy was fixed before evaluation:

1. preserve exactly the 25 complete, sealed reps 50-59 and 65-79;
2. reject the entire unsealed partial 60-64 artifact as a scientific shard;
3. retrain only reps 60,61,62,63,64 in five independent jobs under the unchanged scientific code;
4. require a global 30-replicate / 240-state seal before constructing held-out environments;
5. evaluate all approved shards under the unchanged held-out protocol;
6. apply the unchanged Issue #76 summarizer.

Recovery validation reported:

- 240 checkpoint states verified;
- 30 Q8 training-identity pairs verified;
- 7680 per-environment accuracy rows independently reaggregated;
- maximum environment reaggregation error: **0.0**;
- protocol hash: `e388d13d5890e8b60e939f085403763d5065360bd3cdaecc7aa05de510f144d1`.

The recovery changed execution partitioning only. Replicate count, data subsets, model, optimizer, K/Q values, selectors, seed blocks, held-out environments and frozen statistical decision were unchanged.

## Mechanistic consequence

The evidence hierarchy is now stronger than before this experiment:

- two strong MLP Q-scaling blocks on Digits/geometric;
- one strong SmallCNN block on Digits/geometric, although its non-Q=K endpoint sensitivity is weak;
- one borderline FashionMNIST/Tiny Transformer cross-task block;
- **one strong CIFAR-10/ResNet-20 cross-task block**, with attenuation that remains positive in a post-hoc analysis excluding Q=K.

This supports a narrow finite-budget statement: when selector freedom is binding, hard plus model-conditioned gradient-nonredundancy can allocate a limited update budget more effectively than hardness alone in several tested structured regimes. It still does not identify the downstream learned-function mediator or prove a universal gradient-diversity law.
