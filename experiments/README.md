# Experiments

Run commands from the repository root after installing `requirements.txt`.

## Primary reproduction order

```bash
python experiments/mlp_geometric.py --start 0 --end 20 --output mlp_geometric.csv
python experiments/cnn_replication.py --start 0 --end 20 --output cnn_replication.csv
python experiments/optimizer_ablation.py --start 0 --end 20 --output optimizer_ablation.csv
python experiments/rng_compression_sweep.py --start 0 --end 20 --output rng_compression.csv
```

## Supporting experiments

`sgd_lr_sweep.py` reproduces the independent SGD learning-rate tuning used before the selector ablation.

`gradient_vs_parameter_novelty.py` isolates model-dependent gradient novelty from transformation-parameter diversity.

`wallclock_seed_count.py` studies the seed-count / update-count / wall-clock trade-off.

`common.py` contains the shared dataset split, environment generator, models, selectors, metrics, and deterministic setup.

## Replicate ranges

Scripts use half-open ranges: `--start 0 --end 20` runs replicates 0 through 19. Split ranges can be concatenated later as long as each replicate is run once.

## Statistical summaries

Inspect the committed paired CSVs in `results/`. The published headline numbers are committed snapshots; reruns on different hardware should reproduce metrics under the stated software/determinism assumptions, while wall-clock times may differ.
