# Experiments

Run commands from the repository root after installing `requirements.txt`. This directory contains reproduction scripts, not a single production training package.

## Recommended reading / reproduction order

### A. Core structured benchmark

1. `mlp_geometric.py` — main Digits MLP geometric-shift benchmark.
2. `gradient_vs_parameter_novelty.py` — model-dependent gradient novelty vs physical transformation-parameter novelty.
3. `cnn_replication.py` — small-CNN architecture replication.
4. `optimizer_ablation.py` — AdamW vs independently tuned SGD+momentum.
5. `sgd_lr_sweep.py` — loss-hard-only SGD tuning used before the optimizer comparison.

Example:

```bash
python experiments/mlp_geometric.py --start 0 --end 20 --output mlp_geometric.csv
python experiments/cnn_replication.py --start 0 --end 20 --output cnn_replication.csv
python experiments/optimizer_ablation.py --start 0 --end 20 --output optimizer_ablation.csv
```

## B. Candidate compression and RNG relevance

- `rng_compression_sweep.py` — candidate-count / coverage trade-off.
- `learned_rng_fingerprint.py` — training-only RNG relevance discovery in the original generator.
- `learned_rng_cross_generator.py` — shifted relevant-coordinate generator, stale-fingerprint control, and soft relevance weighting.

These experiments distinguish seed integers from RNG outputs, generated environments, and model-dependent gradient signatures.

## C. Novelty-strength control

- `gradient_novelty_beta_adaptive.py` — adaptive novelty-strength feedback with an absolute gradient-cosine target.
- `gradient_novelty_relative_control.py` — normalized within-step redundancy target used for cross-task Digits/Synthetic validation.

The absolute controller is retained because its cross-task saturation is an informative negative result.

## D. Efficiency / budget study

- `wallclock_seed_count.py` — seed-count, update-count, gradient-reliability, and CPU wall-clock trade-off.

Do not infer GPU-optimal candidate counts from this script; GPU validation is tracked separately in Issue #2.

## E. CIFAR-10 / ResNet external validation

These scripts form one experiment family and should not be interpreted independently:

1. `cifar_resnet_calibrate.py` — calibration/setup checks.
2. `cifar_resnet_tune.py` — optimizer/pretraining tuning that is separated from selector evaluation.
3. `cifar_resnet_pilot.py` — early end-to-end pilot.
4. `cifar_resnet_finetune_pilot.py` — lightweight pretrain→seed-guided-finetune protocol used to make paired validation tractable on CPU CI.
5. `cifar_resnet_primary.py` — pre-registered primary loss-hard vs gradient-novel comparison.

The first 20 primary paired runs are committed in `results/`. Independent reps 20-39 are tracked in PR #11.

## Shared code

`common.py` contains shared dataset/environment/model/selector utilities for the small structured benchmarks. Not every later experiment imports every helper because some mechanism and external-validation scripts intentionally freeze their own protocol.

## Replicate ranges

Where `--start/--end` is supported, ranges are half-open: `--start 0 --end 20` runs replicates 0 through 19. Split ranges may be concatenated as long as each replicate ID is run once and the protocol is unchanged.

## Experimental discipline

For a public comparison:

- keep training/selection and final held-out environment seed pools disjoint;
- share initialization, minibatch order, candidate schedule, and evaluation pool within paired replicates where possible;
- tune optimizer settings independently of the selector comparison;
- do not choose a selector/controller using final held-out test metrics;
- report negative and null comparisons as well as successful ones;
- use the correction family stated in `docs/METHODS.md` rather than promoting isolated raw p-values.

## Where to find evidence

Committed evidence snapshots are indexed in [`../results/README.md`](../results/README.md). The current claim matrix is [`../docs/RESEARCH_STATUS.md`](../docs/RESEARCH_STATUS.md).
