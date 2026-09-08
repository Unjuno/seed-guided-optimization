# Parameter-matched gradient nonredundancy: positive result and remaining factor allocation

## Confirmatory result: Issue83 Stage B

Authoritative run34091432033, source a13c903853b2965eb350d0a65bb58546288ad56f.
Frozen decision: **PARAMETER-MATCHED GRADIENT NONREDUNDANCY SUPPORT**.

30 paired repetitions1800-1829, Digits988 training/809 evaluation images, SmallCNN,
AdamW lr0.005/wd0.001,10epochs,batch128,K16/Q4,one-thread CPU.
New environment/replicate seeds, NOT new image identities or a new dataset.
The reference loss-hard trajectory defines high/low gradient-novelty schedules.
This is not the original online gradnov-versus-loss-hard comparison.

| Quantity | Estimate | Interval | Frozen test |
|---|---:|---|---:|
| High-minus-low heldout mean | +3.969922 pp | 95% [+3.050617,+4.889226] pp | one-sided p5.10359e-10 |
| Reference gradient-novelty gap | +0.203313 | 95% [+0.198874,+0.207752] | p7.45537e-38 |
| Reference standardized hardness difference | -0.001386 | 90% [-0.002502,-0.000270] | TOST within +/-0.05 passes |
| Reference standardized total physical-diversity difference | +0.001186 | 90% [-0.000025,+0.002397] | TOST within +/-0.05 passes |

Heldout SE0.449487 pp;28/30 positive pairs. Clean mean+1.017717 pp is secondary.
The within-step caliper applies to reference scalar summaries, not each arm's subsequent gradients/losses.

## New exploratory discovery: factor composition is not matched

Reconstructing all2400 reference selections from the archived schedules shows that
matching mean pairwise Euclidean distance in seven standardized physical parameters
still leaves their allocation among coordinates imbalanced.

| Coordinate | High-minus-low within-subset population variance | 95% descriptive interval |
|---|---:|---|
| x translation | +0.348892 | [+0.315149,+0.382636] |
| y translation | +0.311187 | [+0.268815,+0.353559] |
| blur | -0.222225 | [-0.243526,-0.200923] |
| additive-noise amplitude | -0.220839 | [-0.238381,-0.203297] |
| brightness | -0.148571 | [-0.173747,-0.123395] |

These variance units are dimensionless squared standardized-coordinate units.
All seven coordinate means and seven variances are reported in
[the exploratory summary](../results/parameter_composition_summary_posthoc.csv).
These fourteen descriptions were not registered confirmatory endpoints and are not
independent replications. Coordinate means also differ. No particular coordinate
has been identified as the unique cause of the performance difference.

**Implication:** Issue83 establishes balance for scalar hardness and scalar total-diversity
within its margins, not all environmental-factor differences. Gradient-based selection
may prioritize particular useful physical factors. Do not call it uniquely gradient-caused.

## A mathematical design limitation (not a performance theorem)

### Variable table

| Symbol | Meaning (Japanese) | SI unit | Definition | Domain/assumptions | Type |
|---|---|---|---|---|---|
| q | 選択環境数 |1| number of rows | positive integer, here4 | scalar integer |
| p | 座標数 |1| number of columns | positive integer | scalar integer |
| Z | 標準化された選択環境座標 |1| rows z_i | real q by p matrix | matrix |
| z_i | 行iの座標ベクトル |1| row i of Z | i from1 to q | row vector |
| O | 直交座標変換 |1| orthogonal matrix | O O^T = O^T O = I | matrix |
| I | 単位行列 |1| p by p identity | same dimension as O | matrix |
| D_ij, D'_ij | 元座標・変換後の点間距離 |1| Euclidean norms of row differences | nonnegative | scalar |
| Z' | 変換後の座標 |1| Z O | same size as Z | matrix |
| i, j | 比較する行番号 |1| row indices |1 through q|scalar integers|
| k | 座標番号 |1| column index |1 through p|scalar integer|
| v_k | 座標ごとの母分散 |1| mean squared centered coordinate | nonnegative | scalar |

For any two rows, orthogonality gives

```math
D'_{ij}{}^2 = (z_i-z_j) O O^T (z_i-z_j)^T
             = (z_i-z_j)(z_i-z_j)^T = D_{ij}^2.
```

Distances are nonnegative, hence the distances themselves and their average are equal.
But individual coordinate variances need not be equal. For four rows
(-3,0),(-1,0),(1,0),(3,0), the first-coordinate variance is
(9+1+1+9)/4=5. Swapping the two coordinate axes is an orthogonal transformation,
preserves every distance, and makes the first-coordinate variance0.
Thus even preserving ALL pairwise distances does not preserve factor-specific
variance allocation; matching just their average cannot guarantee it either.
This is an abstract counterexample, not a claim that such a rotation was applied
to the actual generator or that reference losses remain invariant under it.

Unit check: coordinates, squared distances and variances are all dimensionless.
ERROR CHECK: the counterexample's complete distance matrices were exactly equal
in the accompanying numerical check; first-coordinate variances were5 and0.

## Audit and uncertainty

Six evaluation ZIP SHA256 digests matched GitHub metadata. All30 schedule hashes,
60 checkpoint file/tensor digests, archived scientific source hashes, source/split
consistency,2400 reference rows and4800 environment rows were checked.
Maximum independent environment reaggregation error1.11e-16. Paired t/TOST
statistics were recomputed using a separate checker. Raw-image inference was not rerun.
The primary SE is conditional on fixed images/environment pools; it is not a total
combined uncertainty including dataset/backend variation. t29 coverage factor for
95% intervals is2.04523. No cross-hardware bitwise or speed claim.

Runtime: Python3.12.14, torch2.10.0+cpu, numpy2.3.5, scipy1.17.0, sklearn1.8.0;
MKL2024.2/MKL-DNN3.7.1, deterministic algorithms, one thread. Arm jobs used
AMD EPYC7763/9V74 and Intel Xeon6973P-C; same-repetition arm comparisons share a runner.
Clock snapshots are metadata, not fixed-frequency benchmarks.

Recompute: `python experiments/check_parameter_matched_evidence.py`.
The completed follow-up is [Issue86 spatial-allocation control](SPATIAL_ALLOCATION_RESULT.md):
a fresh four-arm comparison of gradient scoring versus translation-variance-only
scoring, with unchanged scalar matching and a descriptive reference baseline.
