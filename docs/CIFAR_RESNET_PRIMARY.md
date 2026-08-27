# CIFAR-10 / ResNet-20 primary validation

This experiment is the first larger real-image / ResNet-family validation of seed-guided environment selection.

## Design

- CIFAR-10, stratified 6,000-image train subset and 3,000-image test subset
- CIFAR-style ResNet-20 with GroupNorm
- clean pretraining: 10 epochs
- seed-guided fine-tuning: 2 epochs
- 64 training environment seeds
- final held-out environment seeds start at 40000 and are disjoint from tuning/pilot pools
- K=8 candidate environments per update, Q=4 environments used for backward
- paired comparison: loss-hard vs gradient-novelty
- identical initialization, minibatch ordering, candidate schedule, pretraining state, and held-out pool within each replicate

## 40 paired replicates

The initial 20-pair stage was extended with independent replicate IDs 20–39 without changing the protocol.

| metric | loss-hard | gradient-novelty | paired delta | raw p | Holm p (5 metrics) |
|---|---:|---:|---:|---:|---:|
| held-out mean | 42.1407% | 42.2613% | **+0.1206 pp** | 0.002672 | **0.013361** |
| held-out environment SD | 1.1375 pp | 1.1233 pp | -0.0142 pp | 0.319992 | 0.639984 |
| held-out p10 | 40.7208% | 40.8421% | +0.1213 pp | 0.013235 | 0.052939 |
| held-out minimum | 39.2783% | 39.4658% | +0.1875 pp | 0.029383 | 0.088149 |
| clean | 43.9158% | 43.9817% | +0.0658 pp | 0.336312 | 0.639984 |

The held-out mean improvement survives the stated five-metric Holm correction. p10 and minimum are positive-direction but do not cross the corrected 0.05 threshold.

## Interpretation boundary

Safe claim:

> Under this fixed CIFAR-10 / ResNet-20 stochastic fine-tuning protocol, gradient-novelty produced a small corrected-significant improvement in mean held-out environment accuracy over loss-hard selection.

Do **not** claim confirmed tail robustness from this experiment: p10/minimum remain below the pre-specified corrected significance criterion. The effect size is also much smaller than on the structured Digits geometric benchmark, reinforcing that SGO benefit depends on environment structure and heterogeneity.

## Evidence files

- `results/cifar_resnet_primary_all40.csv`
- `results/cifar_resnet_primary_paired40.csv`
- `results/cifar_resnet_primary_summary40.csv`

The extension workflow that generated reps 20–39 completed successfully in PR #11; the combined 40-pair aggregate is the public analysis target.
