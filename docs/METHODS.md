# Methods

## Problem formulation

A seed indexes a stochastic training environment. Depending on the experiment, the seed controls one or more of input noise, masking, geometric transformations, contrast/brightness, blur, minibatch ordering, or stochastic-layer randomness.

For a fixed minibatch and current model parameters, each environment induces an environment loss and a gradient signature. Candidate seed environments are ranked or selected using combinations of:

- **hardness:** environment mean cross-entropy loss;
- **gradient novelty:** cosine distance between normalized gradient signatures;
- **coverage:** preference for rarely selected environments;
- **stagnation:** preference for environments whose EMA loss is not improving.

The cheapest useful gradient signature in the MLP experiments is the gradient of the final linear layer, computed analytically from logits and penultimate features. Full-network gradient signatures were tested and did not improve the reported MLP results.

## Paired experimental design

Methods within a replicate share the initial model seed, minibatch order, candidate-environment schedule, train/test data split, and held-out environment pool. This makes method comparisons paired rather than independent.

## Held-out environment evaluation

Training and evaluation use disjoint environment-seed ranges. Test environments are never used by the selector during training.

Reported metrics include mean accuracy, standard deviation across held-out environments, 10th percentile accuracy (`p10`), minimum environment accuracy, and clean-data accuracy when applicable.

## Geometric-shift benchmark

Dataset: scikit-learn Digits, reshaped to 8×8 grayscale images.

A seed deterministically generates seven environment parameters: rotation angle, x translation, y translation, Gaussian blur, contrast, brightness shift, and additive Gaussian noise. The same environment transformation is applied across examples, producing a shared domain shift rather than independent sample noise.

## RNG prefiltering

RNG prefilter experiments distinguish three objects:

1. the integer seed;
2. a finite RNG-output fingerprint;
3. the model-dependent gradient signature produced by the resulting environment.

The best current prefilter uses the seven RNG draws that actually determine the seven environment parameters. Adding unrelated RNG outputs degraded the distance metric.

A farthest-point procedure retains a diverse subset in RNG-fingerprint space before more expensive model forward/gradient calculations.

## Statistical analysis

Primary experiments use paired replicates. Where multiple outcomes are tested together, Holm correction is used for family-wise error control. Bootstrap confidence intervals are used in several experiment families.

Negative as well as positive results are retained. A result is not treated as general simply because an uncorrected p-value is below 0.05.
