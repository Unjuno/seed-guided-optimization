# Methods

## 1. Problem formulation

A seed is treated as an index into a stochastic training environment, not as a semantic label. Depending on the experiment, the seed controls input noise, masking, geometric transformations, blur, contrast/brightness, minibatch ordering, or stochastic-layer randomness.

For a fixed minibatch and current model, each candidate environment produces:

- **hardness:** mean cross-entropy loss for that environment;
- **gradient signature:** a normalized update-direction proxy;
- **gradient novelty:** cosine distance from already selected gradient signatures;
- optionally **coverage/history** terms in active-selection experiments.

The central comparison is between loss-only selection and selectors that retain hard environments while avoiding redundant gradient directions.

## 2. Models

### MLP

The main geometric benchmark uses a small MLP on scikit-learn Digits. The head-gradient signature is computed from logits and penultimate features.

### CNN replication

A small CNN is evaluated on the same train/test split and the same held-out geometric environment construction. This is an architecture replication, not evidence for modern large-scale CNNs.

Full-network gradient signatures were also tested in the MLP and did not improve over the cheaper final-layer proxy.

## 3. Geometric stochastic environments

Digits images are reshaped to 8×8 grayscale images. A seed deterministically generates seven shared environment parameters:

1. rotation angle;
2. x translation;
3. y translation;
4. Gaussian blur;
5. contrast;
6. brightness shift;
7. additive Gaussian noise.

The same transformation parameters are shared across samples within an environment, so the seed represents a domain-like shift rather than independent per-sample noise.

Training and held-out evaluation use disjoint seed ranges.

## 4. Paired experimental design

Within each replicate, compared methods share:

- initial model parameters;
- train/test data split;
- minibatch order;
- candidate-environment schedule;
- held-out environment pool.

This makes method comparisons paired.

Reported metrics are:

- held-out environment mean accuracy;
- standard deviation across held-out environments;
- 10th-percentile accuracy (`p10`);
- minimum environment accuracy;
- clean-data accuracy when applicable.

## 5. Optimizer ablation and tuning fairness

The optimizer comparison uses AdamW and SGD with momentum.

AdamW uses the established benchmark configuration. SGD+momentum learning rate is selected **before selector comparison**, using a separate loss-hard-only learning-rate sweep. The selected SGD learning rate is `0.2`. This separation prevents selector comparisons from inheriting obvious optimizer mis-tuning.

See:

- `results/sgd_lr_sweep_summary.csv`
- `results/optimizer_ablation_paired20.csv`

## 6. RNG prefiltering

RNG-prefilter experiments distinguish:

1. the integer seed;
2. a finite RNG-output fingerprint;
3. the environment parameters generated from that fingerprint;
4. the model-dependent gradient signature induced by that environment.

The strongest current cheap prefilter uses the seven RNG draws that actually determine the seven geometric environment parameters. Adding unrelated RNG coordinates degrades the distance metric.

A farthest-point procedure reduces the candidate set before more expensive forward/gradient evaluation.

## 7. Statistical analysis

Primary comparisons use paired replicates. When five outcome metrics are tested as one family, Holm correction controls family-wise error.

A result is not promoted to a public claim solely because an uncorrected p-value is below 0.05. Negative results and failed hypotheses are retained.

## 8. Reproducibility assumptions

Published CPU experiments use deterministic PyTorch algorithms and `torch.set_num_threads(1)`. Wall-clock numbers therefore describe the stated hardware/software regime only and should not be transferred directly to GPU execution.
