# Gradient-direction mechanism audit

This audit tests a tempting but overly strong explanation of Seed-Guided Optimization:

> selecting gradient-novel environments simply gives a better finite-sample estimate of the true expected gradient and therefore maximizes immediate loss reduction.

The data support a narrower mechanism.

## Design

Thirty independent MLP training replicates are audited at epochs 0, 2, 5 and 8 on the geometric-shift Digits benchmark.

At each audit point:

1. compute final-layer gradient signatures for all 64 training environments on the same 96-example minibatch;
2. define the **mean reference gradient** as the average of all 64 gradients;
3. define the **tail reference gradient** as the average gradient of the 32 highest-loss environments;
4. draw the same 16 candidate environments and select 4 using random, loss-hard, transformation-parameter novelty or hardness + gradient novelty;
5. compare the mean selected gradient with the two reference gradients.

A descriptive `nearest_ref4` method uses knowledge of the all-64 mean reference and is included only as an oracle-like upper-bound diagnostic, not as a trainable selector.

Metrics include cosine alignment, relative vector error and projection onto each reference direction.

## Gradient approximation results

Across the four audit stages, gradient-novelty is much closer than loss-hard to both reference directions.

Gradient-novelty vs. loss-hard, 30 paired replicates:

- mean-reference cosine: **+0.0384**, Holm `p=4.89e-7`;
- mean-reference relative error: **-0.2255**, Holm `p=6.46e-11`;
- tail-reference cosine: **+0.0159**, Holm `p=0.00395`;
- tail-reference relative error: **-0.1424**, Holm `p=2.14e-8`.

Gradient-novelty also improves mean-reference approximation over physical transformation-parameter novelty:

- mean cosine: **+0.0245**, Holm `p=0.000316`;
- mean relative error: **-0.1209**, Holm `p=2.39e-7`.

### Important negative result: mean expected gradient

Gradient-novelty does **not** significantly outperform random-4 sampling as an estimator of the pure all-environment mean gradient:

- mean-reference cosine difference: +0.0050, not significant;
- mean-reference relative-error difference: +0.0368, not significant.

This is expected conceptually: random sampling is a natural unbiased estimator of the environment mean, whereas gradient novelty intentionally changes the sampling distribution.

### Robust/tail direction

The distinction appears strongly for the tail reference. Relative to random-4, gradient-novelty improves:

- tail cosine by **+0.0759**, Holm `p=4.17e-5`;
- tail relative error by **-0.0955**, Holm `p=0.000217`;
- tail projection by **+0.1084**, Holm `p=0.000537`.

The effect becomes especially visible later in training. At epoch 8, mean tail cosine is approximately:

- gradient-novel: **0.815**;
- loss-hard: 0.803;
- parameter-novel: 0.807;
- random: **0.740**.

## Exact one-step loss-drop test

A second audit freezes the network body and applies the selected final-layer gradient as an exact one-step update. It then measures the change in mean loss across all 64 environments and in the pre-update worst 32 environments.

This directly tests whether gradient novelty works by maximizing immediate loss reduction.

It does not.

At step size `eta=0.01`, mean loss drop is:

- loss-hard: **0.001573**;
- parameter-novel: 0.001515;
- gradient-novel: 0.001394;
- random: 0.001389.

At `eta=0.05`:

- loss-hard: **0.007550**;
- parameter-novel: 0.007290;
- gradient-novel: 0.006758;
- random: 0.006731.

Gradient-novelty therefore produces a significantly **smaller** immediate mean and tail loss drop than loss-hard at both tested step sizes. For example, at `eta=0.05`, gradient-novel minus loss-hard is:

- mean loss drop: **-0.000792**, Holm `p=3.13e-6`;
- tail loss drop: **-0.001391**, Holm `p=5.38e-7`.

Relative to random, immediate **mean** loss reduction is indistinguishable, while tail loss reduction is modestly larger for gradient novelty.

## Interpretation

These results reject the strongest version of the "true gradient" story.

Seed/environment selection should not be described as universally finding a closer approximation to the all-environment expected gradient than random sampling, nor as greedily maximizing the next-step loss decrease.

A more accurate interpretation is:

> hardness + gradient novelty changes the optimization trajectory by retaining difficult environments while reducing redundant/overscaled hard directions and allocating finite updates toward broader robust-gradient coverage.

This is consistent with several previous findings in this repository:

- worst-only and overly narrow tail training can damage mean/clean performance;
- adding tail weighting after active hard selection double-counts hardness and hurts;
- short-horizon learned transfer utilities failed to improve final generalization;
- gradient novelty is most useful under structured environment shifts and can be neutral on low-heterogeneity tasks.

The mechanism is therefore closer to **trajectory regularization / robust directional coverage under finite compute** than to simple expected-gradient Monte Carlo variance reduction.

## Claim boundary

Supported:

- gradient novelty is less distorted than loss-hard relative to broad reference gradients;
- model-conditioned novelty is more informative than physical parameter diversity in the tested MLP audit;
- gradient novelty strongly improves tail-gradient alignment relative to random sampling;
- its long-run benefit cannot be reduced to maximizing immediate one-step loss reduction.

Not supported:

- gradient novelty universally estimates the expected mean gradient better than random sampling;
- the selected update is globally optimal;
- one-step loss reduction predicts final held-out robustness;
- these head-gradient diagnostics are already proven for large models.
