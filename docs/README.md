# Documentation index

This directory separates claim status, protocol, evidence, mechanism, and open questions.

## Start here

1. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md) — compact supported / negative / prospective / open matrix.
2. [`METHODS.md`](METHODS.md) — experimental design, pairing, gradient signatures, RNG prefiltering, and statistical rules.
3. [`RESULTS.md`](RESULTS.md) — narrative summary of the evidence.
4. [`LIMITATIONS.md`](LIMITATIONS.md) — claim boundary and remaining validation gaps.

## Mechanism and control

- [`GRADIENT_DIRECTION_AUDIT.md`](GRADIENT_DIRECTION_AUDIT.md) — mean-gradient, tail-gradient, and one-step loss controls.
- [`TRAJECTORY_MECHANISM.md`](TRAJECTORY_MECHANISM.md) — cross-task trajectory diagnostics and the representation-rank candidate mechanism.
- [`PROSPECTIVE_REPRESENTATION_RANK.md`](PROSPECTIVE_REPRESENTATION_RANK.md) — seven registered direction tests across five datasets and retirement of the failed tail rule.
- [`FASHION_TRANSFORMER_REP_RANK.md`](FASHION_TRANSFORMER_REP_RANK.md) — preregistered FashionMNIST/Tiny Transformer architecture-shift test; directional decision PASS.
- [`ADAPTIVE_BETA.md`](ADAPTIVE_BETA.md) — feedback control of gradient-novelty strength.
- [`RELATIVE_REDUNDANCY_CONTROL.md`](RELATIVE_REDUNDANCY_CONTROL.md) — task-normalized redundancy control.

## RNG / candidate compression

- [`RNG_CROSS_GENERATOR.md`](RNG_CROSS_GENERATOR.md) — learned RNG relevance, cross-generator stale-fingerprint failure, and soft relevance weighting.

## External validation

- [`CIFAR_RESNET_PRIMARY.md`](CIFAR_RESNET_PRIMARY.md) — fixed-protocol CIFAR-10 / ResNet-20 validation through 40 paired replicates.

CSV evidence snapshots live in [`../results/`](../results/); reproduction scripts live in [`../experiments/`](../experiments/).
