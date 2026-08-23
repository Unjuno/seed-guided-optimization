# Cross-generator RNG fingerprint validation

This experiment tests whether a learned RNG fingerprint represents a universal property of seed values or a generator/model-conditioned property that must be recalibrated.

## Setup

The second generator reads a 64-value RNG vector but only seven non-contiguous coordinates affect the environment:

`[5, 13, 22, 31, 40, 49, 58]`

Those coordinates control rotation, x/y translation, blur, contrast, brightness and noise strength. Calibration uses only training data and separate calibration environment seeds. Training environments and the 80 held-out evaluation environments are disjoint from calibration.

The previous generator's learned top-7 coordinates (`[1, 0, 2, 4, 3, 10, 26]`) are included as an explicit transfer baseline.

## Coordinate recovery

A Ridge model predicts a PCA-compressed final-layer gradient signature from the 64 standardized RNG outputs. Coefficient norms define coordinate relevance.

With 128 calibration seeds, the learned top-7 was:

`[22, 5, 13, 31, 40, 49, 53]`

Six of seven true coordinates were recovered. The weak noise-strength coordinate `58` ranked 35th.

Repeated calibration subsets show that six coordinates are stable while the weak seventh coordinate is not. More calibration seeds did not monotonically improve hard top-7 recovery; this is evidence against interpreting exact top-k coordinate recovery as the objective.

## 20 paired training replicates

All methods prefilter 16 candidate environments to 8, then use the same hardness + gradient-novelty selector to choose 4 backward environments.

| Prefilter | Held-out mean | Env SD | p10 | Minimum | Clean |
|---|---:|---:|---:|---:|---:|
| Oracle 7 | 53.69% | 11.70 pp | 36.89% | 26.70% | 50.62% |
| Learned hard top-7 | 53.11% | 11.99 pp | 35.43% | 25.38% | 50.12% |
| Old-generator top-7 | 51.54% | 13.26 pp | 33.05% | 19.45% | 48.20% |
| Raw 64 | 52.74% | 13.03 pp | 33.97% | 21.98% | 47.62% |
| Random 8 | 52.63% | 12.61 pp | 34.35% | 25.96% | 49.72% |
| **Relevance-weighted top-12** | **54.11%** | **11.82 pp** | **36.99%** | **25.38%** | **50.06%** |

### Recalibration vs. old fingerprint

Learned hard top-7 vs. the previous generator's top-7 improved:

- mean: **+1.57 pp**, Holm `p=0.000731`;
- p10: **+2.38 pp**, Holm `p=0.00149`;
- minimum environment: **+5.93 pp**, Holm `p=0.000316`.

The old fingerprint therefore does not transfer as a universal semantic class of seed values.

### Weighted relevance vs. raw RNG

Relevance-weighted top-12 vs. raw 64-dimensional RNG distance improved:

- mean: **+1.37 pp**, Holm `p=0.00275`;
- environment SD: **-1.21 pp**, Holm `p=4.40e-6`;
- p10: **+3.02 pp**, Holm `p=3.88e-6`;
- minimum environment: **+3.40 pp**, Holm `p=0.00275`;
- clean: **+2.44 pp**, Holm `p=0.00275`.

### Weighted relevance vs. oracle

No tested metric differed significantly from the oracle seven-coordinate fingerprint after five-metric Holm correction.

Weighted top-12 also outperformed hard learned top-7 on mean (+1.00 pp, Holm `p=0.0103`) and p10 (+1.55 pp, Holm `p=0.00312`).

## Interpretation

The evidence supports a narrower and more useful model:

> An RNG fingerprint should be treated as a generator/model-conditioned relevance representation, not as a semantic class attached to the integer seed itself.

Hard coordinate selection is brittle when some stochastic factors have weak effects on the current gradient representation. Retaining uncertainty as continuous relevance weights can be more robust than forcing an exact top-k classification.

This result is still based on a synthetic geometric environment generator and a small MLP. It does not establish transfer to unrelated real simulators or large neural networks.
