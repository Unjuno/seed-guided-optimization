# Loss-stratified gradient-nonredundancy intervention result

## Status

Preregistered Issue #80 is complete. Authoritative GitHub Actions run: **34088665610**.

Frozen decision: **LOSS-STRATIFIED NONREDUNDANCY SUPPORT**.

This is the first completed experiment in the repository that holds subset size fixed and constructs a direct MAXNOV versus MINNOV schedule contrast on one common reference trajectory. It is stronger causal evidence than Q=K disappearance alone, but a post-hoc diagnostic identifies an important remaining competing explanation: the MAXNOV schedules also have substantially greater diversity in the known physical transformation parameters.

## Frozen design

- sklearn Digits, canonical 988-image training split;
- evaluation: all 809 canonical nontraining images;
- existing SmallCNN;
- AdamW lr=5e-3, weight decay=1e-3;
- 10 epochs, batch128;
- K=16 candidate environments, **Q=4 in both arms**;
- paired reps1600-1629, n=30;
- fresh training environment seeds51000-51063;
- fresh heldout environment seeds52000-52079;
- deterministic one-thread CPU.

For every reference step, both arms were defined from the same loss-hard reference model state and same 16 candidates. Both always retained the hardest candidate and selected exactly one candidate from each identical adjacent loss-rank pair `(2,3)`, `(4,5)`, `(6,7)`. Among the same eight feasible Q=4 subsets:

- **MAXNOV** = maximum mean pairwise head-gradient novelty;
- **MINNOV** = minimum mean pairwise head-gradient novelty.

The reference trajectory itself advanced with ordinary loss-hard top4. All MAXNOV/MINNOV schedules were sealed before arm training. Two arm models were then reset to the same initialization and trained on those fixed schedules. All 60 arm states were sealed before the 80 fresh heldout environments were constructed.

## Frozen manipulation

Per replicate, the mean reference-step novelty contrast was

```text
N_r = mean_step(novelty_MAXNOV - novelty_MINNOV).
```

Result:

- mean novelty difference: **+0.231634**;
- SE: 0.002623;
- two-sided t29 95% CI: **[+0.226269,+0.236999]**;
- one-sided p: **4.12e-37**;
- positive: **30/30 reps**.

The novelty manipulation therefore passed decisively.

## Frozen hardness equivalence

At each reference step:

```text
z_hard = (selected_mean_loss - candidate_mean_loss) / (candidate_loss_sd + 1e-8)
```

and `H_r` was the mean MAXNOV-minus-MINNOV difference across steps. The preregistered equivalence margin was ±0.10 candidate-pool loss SD units.

Result:

- mean standardized hardness difference: **-0.009100 SD**;
- SE: 0.002458 SD;
- t29 90% CI: **[-0.013276,-0.004924] SD**;
- TOST lower p: **2.95e-26**;
- TOST upper p: **1.62e-28**;
- 90% interval is fully inside [-0.10,+0.10].

Hardness equivalence therefore passed. The raw reference selected-loss difference was also small: mean **-0.001248** cross-entropy units.

## Frozen heldout result

Primary performance quantity:

```text
B_r = heldout_mean_accuracy_MAXNOV - heldout_mean_accuracy_MINNOV.
```

Result:

- mean benefit: **+4.66929 percentage points**;
- paired SE: **0.72486 pp**;
- two-sided t29 95% CI: **[+3.18679,+6.15180] pp**;
- one-sided p: **2.38909e-7**;
- positive: **26/30 pairs**.

Thus manipulation, hardness equivalence and performance all passed the frozen rule.

## Secondary observations

Descriptive only:

- clean MAXNOV-MINNOV: **-1.75525 pp**;
- p10 heldout: **+5.48908 pp**;
- minimum heldout: **+3.79481 pp**;
- mean schedule overlap: **0.41979** of the four selected environments. The hardest anchor is always shared, so the arms differ substantially beyond that forced overlap.

The p10/minimum metrics were not multiplicity-controlled primary endpoints; this is not a new tail-safety claim. The negative clean effect indicates that the useful shifted-environment change need not improve clean performance.

After the intervention begins, the MAXNOV arm experiences lower mean selected training loss on its own diverged trajectory (MAXNOV-MINNOV approximately -0.06894 cross-entropy units, post-hoc). This is a post-treatment dynamic, not a baseline hardness mismatch, and should not be adjusted away when estimating the schedule intervention effect.

## Post-hoc competing-mechanism diagnostic

The environment generator has seven known physical parameters: rotation, x/y translation, blur, contrast, brightness and noise. For a diagnostic not included in the frozen decision, these seven coordinates were standardized across the 64 training environments. At each reference step the mean pairwise Euclidean distance among the four selected physical parameter vectors was computed.

Across reps:

- MAXNOV mean parameter diversity: 3.97085;
- MINNOV mean parameter diversity: 3.47815;
- MAXNOV-MINNOV: **+0.492697** standardized-distance units;
- SE: 0.009559;
- two-sided t29 95% CI: **[+0.473147,+0.512246]**;
- positive: **30/30 reps**;
- descriptive one-sided p: **2.25e-30**.

At the step level, gradient-novelty difference and physical-parameter-diversity difference are positively associated (Pearson r≈0.276). Therefore the intervention does **not** isolate gradient non-redundancy from physical transformation diversity. It establishes that, among tightly loss-stratified feasible subsets, schedules selected to maximize model-conditioned gradient novelty produce substantially better shifted heldout performance than schedules selected to minimize it; however the treatment also changes a known environment-space covariate.

This matters for interpretation. The existing earlier experiment showing gradient-novel selection outperforming parameter-novel selection makes a pure parameter-distance explanation less attractive, but it does not eliminate this confounding in the present direct intervention.

## Independent verification

A separate post-run audit downloaded all six evaluation artifacts and the aggregate artifact and verified:

- all 6 evaluation ZIP SHA256 digests against GitHub artifact metadata;
- all 30 fixed schedule hashes;
- all 60 checkpoint file hashes;
- all 60 canonical state-tensor digests against both training and heldout tables;
- archived scientific source hashes;
- 2,400 reference-step rows;
- 4,800 per-environment heldout accuracy rows;
- per-environment reconstruction of mean/SD/p10/min with maximum numerical difference **1.25e-16**;
- aggregate paired quantities reconstructed from raw reference/heldout rows with maximum CSV-roundtrip difference below **1.0e-16**;
- the novelty t-test, hardness TOST and heldout t-test independently recomputed.

Aggregate artifact SHA256: `30cc36572b982be2c4c547ad94a8ef254626e66780c53f794e1c707f269c7d7f`.

## Mechanistic consequence

The result materially advances the mechanism beyond Q-scaling:

```text
fixed Q + matched reference loss-rank structure
    + higher reference gradient non-redundancy schedule
    -> different learned function
    -> much better full-strength geometric heldout mean
```

But the most defensible wording is **nonredundancy-associated direct intervention support**, not “gradient novelty is now the unique causal mediator,” because known transformation-parameter diversity moved with the treatment.

The next decisive experiment should therefore preserve the shared-reference / fixed-Q design while additionally matching physical transformation-parameter diversity, then maximize versus minimize gradient novelty inside that matched feasible set. If that manipulation remains strong and reproduces the heldout benefit, the gradient-specific causal interpretation becomes substantially harder to replace with environment-space diversity.
