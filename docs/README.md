# Documentation index

This directory separates claim status, protocol, evidence, theory, mechanism, reproducibility, and open questions.

## Start here

1. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md) — compact supported / negative / prospective / open matrix.
2. [`THEORETICAL_FRAMEWORK.md`](THEORETICAL_FRAMEWORK.md) — current working explanation: finite-budget coverage of unresolved gradient subspaces and conversion into reusable representation structure.
3. [`METHODS.md`](METHODS.md) — experimental design, pairing, gradient signatures, RNG prefiltering, and statistical rules.
4. [`RESULTS.md`](RESULTS.md) — narrative summary of the evidence.
5. [`LIMITATIONS.md`](LIMITATIONS.md) — claim boundary and remaining validation gaps.

## Mechanism and prospective validation

- [`GRADIENT_DIRECTION_AUDIT.md`](GRADIENT_DIRECTION_AUDIT.md) — mean-gradient, tail-gradient, and one-step loss controls.
- [`TRAJECTORY_MECHANISM.md`](TRAJECTORY_MECHANISM.md) — cross-task trajectory diagnostics and representation-conversion mechanism candidate.
- [`PROSPECTIVE_REPRESENTATION_RANK.md`](PROSPECTIVE_REPRESENTATION_RANK.md) — eight registered condition-level direction tests across six datasets, Fashion independent replication, and retirement of the failed tail rule.
- [`FASHION_TRANSFORMER_REP_RANK.md`](FASHION_TRANSFORMER_REP_RANK.md) — initial FashionMNIST/Tiny Transformer prospective PASS.
- [`FASHION_TRANSFORMER_EXTENSION.md`](FASHION_TRANSFORMER_EXTENSION.md) — exact-protocol independent 20-pair Fashion extension; directional relation replicates.
- [`CIFAR_RESNET_REP_RANK.md`](CIFAR_RESNET_REP_RANK.md) — independent CIFAR-10/ResNet-20 prospective representation-rank PASS.
- [`ADAPTIVE_BETA.md`](ADAPTIVE_BETA.md) — feedback control of gradient-novelty strength.
- [`RELATIVE_REDUNDANCY_CONTROL.md`](RELATIVE_REDUNDANCY_CONTROL.md) — task-normalized redundancy control.

## RNG / candidate compression

- [`RNG_CROSS_GENERATOR.md`](RNG_CROSS_GENERATOR.md) — learned RNG relevance, cross-generator stale-fingerprint failure, and soft relevance weighting.

## External validation and reproducibility

- [`CIFAR_RESNET_PRIMARY.md`](CIFAR_RESNET_PRIMARY.md) — fixed-protocol CIFAR-10 / ResNet-20 validation through 40 paired replicates.
- [`CIFAR_CPU_REPRO_AUDIT.md`](CIFAR_CPU_REPRO_AUDIT.md) — preregistered single-thread hosted-CPU audit; numerical drift persists while aggregate directions remain stable.

CSV evidence snapshots live in [`../results/`](../results/); reproduction scripts live in [`../experiments/`](../experiments/).
