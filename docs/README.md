# Documentation index

This directory separates the public claim, experimental protocol, evidence, and open questions.

## Start here

1. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md) — compact matrix of supported, negative, suggestive, and in-progress findings.
2. [`METHODS.md`](METHODS.md) — experimental design, pairing, gradient signatures, RNG prefiltering, and statistical rules.
3. [`RESULTS.md`](RESULTS.md) — narrative summary of the evidence.
4. [`LIMITATIONS.md`](LIMITATIONS.md) — current claim boundary and remaining validation gaps.

## Mechanism and control studies

- [`GRADIENT_DIRECTION_AUDIT.md`](GRADIENT_DIRECTION_AUDIT.md) — mean-gradient, tail-gradient, and one-step loss-reduction controls.
- [`ADAPTIVE_BETA.md`](ADAPTIVE_BETA.md) — feedback control of gradient-novelty strength.
- [`RELATIVE_REDUNDANCY_CONTROL.md`](RELATIVE_REDUNDANCY_CONTROL.md) — task-normalized redundancy control and cross-task transfer.

## RNG / candidate compression

- [`RNG_CROSS_GENERATOR.md`](RNG_CROSS_GENERATOR.md) — learned RNG relevance, cross-generator failure of stale fingerprints, and soft relevance weighting.

## External validation

- [`CIFAR_RESNET_PRIMARY.md`](CIFAR_RESNET_PRIMARY.md) — CIFAR-10 / ResNet-20 protocol and first 20 paired primary results.

The CSV evidence snapshots live in [`../results/`](../results/), and executable reproductions live in [`../experiments/`](../experiments/).
