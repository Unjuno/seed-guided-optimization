# CIFAR-10 / ResNet-20 primary validation

This experiment is the first larger real-image / ResNet-family validation of seed-guided environment selection.

## Design

- CIFAR-10, stratified 6,000-image train subset and 3,000-image test subset
- CIFAR-style ResNet-20 with GroupNorm
- clean pretraining: 10 epochs
- seed-guided fine-tuning: 2 epochs
- 64 training environment seeds
- final held-out environment seeds start at 40000 and are disjoint from tuning and pilot pools
- K=8 candidate environments per update, Q=4 environments used for backward
- primary paired comparison: loss-hard versus gradient-novelty
- identical initialization, minibatch ordering, candidate-environment schedule, pretraining state, and held-out environment pool within each replicate

## First 20 paired replicates

| metric | loss-hard | gradient-novelty | paired delta | raw p | Holm p (5 metrics) |
|---|---:|---:|---:|---:|---:|
| held-out mean | 42.3135% | 42.4515% | +0.1380 pp | 0.0224 | 0.1120 |
| held-out environment SD | 1.1484 pp | 1.1489 pp | +0.0004 pp | 0.9851 | 0.9851 |
| held-out p10 | 40.9202% | 41.0635% | +0.1433 pp | 0.0324 | 0.1298 |
| held-out minimum | 39.4483% | 39.6383% | +0.1900 pp | 0.2089 | 0.6268 |
| clean | 44.2433% | 44.3167% | +0.0733 pp | 0.4716 | 0.9432 |

Mean and p10 are positive with unadjusted paired p<0.05, but neither survives correction across the five reported metrics. This is therefore **suggestive but not confirmatory** evidence on this benchmark.

Training time was essentially unchanged in this CPU implementation (gradient-novelty 219.9 s versus loss-hard 222.5 s per fine-tuning run on average), but this is not a GPU wall-clock claim.

## Pre-committed extension

Because the first 20 replicates show a small, consistently positive primary effect but do not survive multiplicity correction, the same protocol is being extended with independent replicate IDs 20-39. No selector, optimizer, data subset size, candidate/backward budget, or held-out environment pool definition is changed for the extension.

The 40-replicate combined analysis will be reported regardless of direction.

## Interpretation boundary

This result does not establish that gradient-novelty universally improves ResNets or CIFAR-10 training. It tests a specific structured stochastic-environment fine-tuning regime. The effect size is much smaller than on the structured Digits geometric-shift benchmark, which is evidence that the benefit depends strongly on the structure and heterogeneity of the stochastic environments.
