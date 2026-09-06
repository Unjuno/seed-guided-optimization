# FashionMNIST Tiny Transformer finite-budget Q-scaling (Issue #73)

Authoritative Actions run: **34003772907**. Scientific head: `0d2277140a0d0e8f7d662302a30f34f03311d080`.

## Frozen decision

**FASHION TRANSFORMER FINITE-BUDGET COVERAGE REPLICATES**.

This is the first direct Q-scaling test outside the Digits dataset/generator family. It reused the established FashionMNIST TinyPatchTransformer protocol: 3000 stratified train images, 1000 stratified test images, dim48 / 2-layer / 4-head Transformer, clean pretrain for 5 epochs, environment finetune for 2 epochs, AdamW, K=8, and the existing geometric/photometric environment generator. Fresh paired reps were 30-59; fresh train environment seeds were 70000-70063 and fresh heldout seeds 80000-80023.

For each replicate and Q, `B[r,Q] = heldout_mean_gradnov - heldout_mean_loss_hard`. The frozen contrast was

```text
A_B = mean(B_Q2, B_Q4) - mean(B_Q6, B_Q8).
```

Results:

- mean low-Q benefit: **+0.43625 pp**;
- mean high-Q benefit: **+0.12250 pp**;
- attenuation: **+0.31375 pp**;
- paired SE: **0.15659 pp**;
- two-sided t29 95% CI: **[-0.00651,+0.63401] pp**;
- preregistered one-sided p: **0.027264**;
- positive attenuation pairs: **20/30**.

The frozen directional rule therefore passes, but the evidence is **borderline**: the two-sided 95% interval narrowly crosses zero. It should not be described as strong cross-task confirmation comparable in precision to the three Digits Q-scaling blocks.

## Per-Q heldout mean benefit

| Q | Q/K | gradnov - loss-hard | paired SE | one-sided p | positive pairs |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.25 | **+0.84639 pp** | 0.29274 pp | 0.003600 | 21/30 |
| 4 | 0.50 | +0.02611 pp | 0.09754 pp | 0.3954 | 17/30 |
| 6 | 0.75 | **+0.24500 pp** | 0.10437 pp | 0.01297 | 17/30 |
| 8 | 1.00 | **0 exactly** | 0 | 0.5 by zero-effect convention | 0/30 |

The curve is clearly not monotonic: Q=6 exceeds Q=4. The cross-task result supports only the frozen low-vs-high directional contrast and exact Q=K disappearance. It does not support a smooth or universal Q response curve.

## Q=K exact identity

At Q=8, both selectors update on the same eight candidates in the same sorted order. Across all 30 paired replicates there were zero mismatches in:

- canonical SHA256 model-state tensor digest;
- every model parameter tensor (independently checked with `torch.equal`);
- selected pairwise-gradient novelty;
- mean candidate and selected losses;
- heldout mean, SD, p10, minimum and clean accuracy.

Thus the exact Q=K control generalizes beyond Digits: when all candidate environments contribute, the two selection rules have no remaining update-allocation degree of freedom.

## Manipulation diagnostic

Mean gradnov-minus-loss-hard selected pairwise-gradient novelty:

- Q2: **+0.16180**;
- Q4: **+0.04756**;
- Q6: **+0.02426**;
- Q8: **0 exactly**.

The non-redundancy manipulation therefore contracts as Q approaches K and vanishes at Q=K, as expected from finite subset coverage.

## Independent verification

The six original shard ZIP SHA256 digests exactly matched GitHub artifact metadata. All **240 checkpoint file hashes** matched sealed manifests. All 240 canonical state-tensor digests recomputed from checkpoint tensors matched the training CSVs. Archived source snapshots matched manifest source hashes. Q8 parameter tensors were independently compared and had zero mismatches. The aggregate 240 scientific rows matched the six shard CSVs exactly after parsing; primary paired means, SE, t29 interval and one-sided p were independently recomputed.

Hosted runners used AMD EPYC 9V74, AMD EPYC 7763 and Intel Xeon Platinum 8573C CPUs, with PyTorch 2.10.0+cpu and four threads. Comparisons remain paired within each replicate. No wall-clock or cross-hardware bitwise claim is made.

This workflow retained aggregate heldout metrics but not per-environment heldout rows, so the heldout aggregation itself cannot be independently reconstructed from the archived CSVs without reloading the fixed FashionMNIST subset and generator. This is an evidence-archive limitation, not a change to the frozen statistical result.

## Mechanistic consequence

Before this test, finite-budget Q-scaling had replicated three times within Digits/geometric: two MLP blocks and one SmallCNN block. This Fashion result provides the first **cross-task directional support** for the same mechanism and reproduces exact Q=K identity.

The appropriate scope is:

> Finite-budget subset allocation is now supported strongly within Digits/geometric across MLP and SmallCNN, and receives a preregistered but statistically borderline cross-task replication on FashionMNIST/Tiny Transformer. The common exact Q=K disappearance strongly supports the necessity of selector freedom, while the magnitude/shape of the Q response is task dependent.

This still does not identify the downstream function-space mediator or establish universal validity.

## Reproduction

Recompute the primary paired statistic from public result rows:

```bash
python experiments/check_fashion_budget_paired.py --input-dir results
```

Original Actions artifacts retain all 240 checkpoints, manifests, training/heldout aggregate CSVs, source snapshots and runtime metadata.