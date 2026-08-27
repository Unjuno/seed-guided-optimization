# Methods

## 1. Problem formulation

A seed is treated as an index into a stochastic training environment, not as a semantic label. Depending on the experiment, the seed controls input noise, masking, geometric transformations, blur, contrast/brightness, minibatch ordering, or other stochastic components.

For a fixed minibatch and current model, each candidate environment can produce:

- **hardness** — mean loss for that environment;
- **gradient signature** — a normalized update-direction proxy, usually from the final layer;
- **gradient novelty** — cosine distance from already selected gradient signatures;
- optionally **coverage/history/relevance** statistics in supporting selectors.

The central comparison is between loss-only selection and selectors that retain hard environments while avoiding redundant gradient directions.

## 2. Core selector

The main structured selector uses one hard anchor and then greedily adds candidates using a score of the form

```text
score_i = z(loss_i) + beta * z(novelty_i)
```

where novelty is computed from normalized gradient signatures and `z(.)` denotes within-candidate standardization. Pure diversity without a hardness term is retained as a negative control rather than the preferred algorithm.

## 3. Gradient signatures

For the MLP and several supporting studies, the gradient signature is computed analytically from logits and penultimate features for the final linear layer. This is much cheaper than materializing the full-network gradient for every candidate environment.

A direct full-gradient comparison found that the higher-dimensional signature was slower and did not improve held-out selection quality in the tested MLP. Final-layer signatures are therefore the default empirical proxy, not a theoretically guaranteed replacement for full gradients.

## 4. Models and datasets

### Digits MLP

The main structured benchmark uses scikit-learn Digits with a small MLP.

### Small CNN replication

A small convolutional network is evaluated on the same held-out geometric environment family. This is an architecture replication, not evidence for large modern CNNs.

### Synthetic classification

An independent synthetic multiclass task is used for cross-task selector/controller transfer tests.

### Breast Cancer Wisconsin

A real tabular binary-classification task is used for low/high corruption-regime controls and ongoing controller-safety studies.

### CIFAR-10 / ResNet-20

The current external-validation benchmark uses a CIFAR-style ResNet-20 with a stratified 6,000-image training subset and 3,000-image test subset. Clean pretraining is followed by seed-guided fine-tuning. Training/tuning/pilot/final held-out environment pools are separated.

## 5. Structured geometric stochastic environments

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

## 6. Paired experimental design

Within each replicate, compared methods share as much nuisance randomness as possible, including:

- initial model parameters;
- train/test split;
- minibatch order;
- candidate-environment schedule;
- held-out environment pool.

This makes the primary method comparisons paired.

Reported outcome metrics commonly include:

- held-out environment mean accuracy;
- standard deviation across held-out environments;
- 10th-percentile accuracy (`p10`);
- minimum environment accuracy;
- clean-data accuracy.

## 7. Optimizer tuning fairness

Optimizer settings are tuned independently of selector comparison. The main optimizer ablation uses AdamW and SGD+momentum. SGD learning rate was selected using a separate loss-hard-only sweep before the final paired selector comparison; the selected SGD learning rate is `0.2` for that benchmark.

See `results/sgd_lr_sweep_summary.csv` and `results/optimizer_ablation_paired20.csv`.

## 8. RNG prefiltering

RNG-prefilter experiments distinguish:

1. integer seed;
2. finite RNG-output fingerprint;
3. environment generated from that fingerprint;
4. model-dependent gradient signature induced by the environment.

A cheap farthest-point prefilter can reduce the candidate set before expensive forward/gradient evaluation. Raw RNG distance is only useful when the fingerprint represents coordinates that matter to the environment; adding unrelated coordinates can degrade selection.

### Learned RNG relevance

Later experiments remove direct knowledge of the relevant coordinates. A training-only calibration pool is used to fit feature relevance from a longer RNG window to gradient-derived targets. Cross-generator controls move the relevant coordinates and test whether stale relevance transfers. It does not: relevance must be re-estimated for the new generator. Soft relevance weighting is also compared with hard top-k coordinate selection.

## 9. Adaptive and relative novelty control

A fixed `beta` controls the trade-off between hardness and novelty.

An absolute feedback controller attempted to target a selected-gradient pairwise cosine value directly. This failed to transfer when a second task had a different feasible cosine range.

The relative controller instead computes, at each step:

```text
c_target = c_strong_novelty + rho * (c_hardness - c_strong_novelty)
```

where `c_hardness` and `c_strong_novelty` are the selected-pair cosine values under two reference operating points. The controller then chooses the tested beta whose selected cosine is closest to this within-step target. Held-out test performance is not used by the controller.

## 10. Mechanism audits

Separate audits compare selected gradients against:

- the mean gradient over a larger environment reference set;
- a tail/worst-environment reference gradient;
- actual one-step loss reduction after applying the selected update.

These audits are used to test mechanistic explanations rather than to select hyperparameters on held-out environments.

## 11. Statistical analysis

Primary comparisons use paired replicates. When five outcome metrics are tested as one family, Holm correction controls family-wise error.

A result is not promoted to a public claim solely because an uncorrected p-value is below 0.05. In particular, the first 20 CIFAR-10 / ResNet-20 pairs are reported as suggestive rather than confirmatory because the positive mean/p10 raw p-values do not survive the stated correction family.

Negative, null, and failed hypotheses are retained.

## 12. Reproducibility assumptions

Most committed small-scale CPU experiments use deterministic PyTorch algorithms and `torch.set_num_threads(1)`. Long-running CIFAR validation is executed through GitHub Actions workflows with pinned experiment commands.

Wall-clock values describe the stated software/hardware regime only. CPU timing must not be extrapolated directly to GPU execution.
