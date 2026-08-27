# Experiments

Run commands from the repository root after installing `requirements.txt`. These are reproduction scripts, not a single production training package.

## Recommended reproduction order

### A. Core structured benchmark
- `mlp_geometric.py`
- `gradient_vs_parameter_novelty.py`
- `cnn_replication.py`
- `optimizer_ablation.py`
- `sgd_lr_sweep.py`

### B. Candidate compression and RNG relevance
- `rng_compression_sweep.py`
- `learned_rng_fingerprint.py`
- `learned_rng_cross_generator.py`

### C. Novelty-strength control
- `gradient_novelty_beta_adaptive.py`
- `gradient_novelty_relative_control.py`
- `relative_control_breast_cancer.py`

### D. Mechanism
- `trajectory_mechanism_pilot.py` — four-task trajectory audit comparing accumulated gradient and hidden-representation geometry.

### E. Prospective representation-rank validation
The frozen rule is evaluated without using the final held-out pool before prediction registration.

- `prospective_rep_rank_validation.py` — Digits photometric
- `prospective_rep_rank_unstructured.py` — Digits unstructured pixel corruption
- `prospective_rep_rank_bands.py` — Digits band/edge occlusion
- `prospective_wine_rep_rank.py`
- `prospective_iris_rep_rank.py`
- `prospective_diabetes_regression.py`

The corresponding prediction/outcome sequence is documented in `docs/PROSPECTIVE_REPRESENTATION_RANK.md` and Issue #12.

### F. CIFAR-10 / ResNet external validation
- `cifar_resnet_calibrate.py`
- `cifar_resnet_tune.py`
- `cifar_resnet_pilot.py`
- `cifar_resnet_finetune_pilot.py`
- `cifar_resnet_primary.py`

The primary fixed protocol now has 40 paired replicates. See `results/cifar_resnet_primary_*40.csv`.

### G. Efficiency / budget
- `wallclock_seed_count.py`

CPU wall-clock measurements are hardware-specific and must not be interpreted as GPU-optimal.

## Experimental discipline

For public comparisons:

- keep training/selection and final held-out seed pools disjoint;
- share initialization, minibatch order, candidate schedule, and evaluation pool within paired replicates where possible;
- tune optimizer settings independently of selector comparison;
- do not choose selector/controller settings from final held-out metrics;
- for prospective diagnostics, register the predicted direction before evaluating the final held-out pool;
- report negative/null comparisons;
- use the correction family stated in `docs/METHODS.md`.

## Evidence

Committed evidence snapshots are indexed in [`../results/README.md`](../results/README.md). Claim status is in [`../docs/RESEARCH_STATUS.md`](../docs/RESEARCH_STATUS.md).
