# Parameter-diversity-matched gradient-nonredundancy result

## Status

Preregistered Issue #83 is complete through both stages.

- Stage A authoritative calibration run: **34090850424**.
- Stage B authoritative confirmatory run: **34091432033**.
- Frozen Stage-B decision: **PARAMETER-MATCHED GRADIENT NONREDUNDANCY SUPPORT**.

This experiment directly follows Issue #80. Issue #80 showed a large fixed-Q MAXNOV>MINNOV held-out effect with reference hardness equivalence, but MAXNOV also had much larger diversity in the known seven-dimensional physical transformation parameters. Issue #83 explicitly matched that parameter-space diversity before comparing gradient novelty.

## Stage A: training-only caliper calibration

Stage A used only common reference trajectories. No intervention arm was trained and no held-out environment was constructed.

- pilot reps: 1700-1709;
- training environment seeds: 53000-53063;
- K=16, Q=4;
- each reference step enumerated all 84 subsets formed by loss-rank1 plus any 3 candidates from loss ranks2-10;
- every subset received three reference diagnostics:
  - mean pairwise head-gradient novelty `G`;
  - standardized hardness `z_hard`;
  - standardized seven-dimensional physical transformation diversity `z_param`.

For a frozen caliper `c`, a subset pair was eligible only when both

```text
|delta z_hard| <= c
|delta z_param| <= c
```

Among eligible pairs, the pair with maximum absolute gradient-novelty gap was selected. The preregistered caliper grid was `{0.05,0.10,0.15,0.20,0.30}` and the smallest caliper with 100% pilot-step feasibility plus mean gradient gap >=0.10 had to be selected.

The smallest candidate already passed:

| Stage-A quantity at c=0.05 | Result |
|---|---:|
| pilot reference steps | 800 |
| feasible steps | **800/800** |
| feasibility rate | **1.000** |
| mean gradient-novelty gap | **+0.199492** |
| mean absolute hardness-z difference | 0.025188 |
| mean absolute parameter-z difference | 0.025584 |

Therefore **c=0.05** was sealed before any Stage-B arm training or held-out evaluation.

The first Stage-A workflow attempt was technically incomplete because the aggregate runner lacked a `torch` dependency. It reached no caliper-selection decision. The retry changed only the aggregate runtime dependency; the scientific code, grid and thresholds were unchanged.

Stage-A protocol hash: `444ed1dd41a200f03e883b0e86583b92e86f4f182226f4f6fac3f7fed3a5b9d3`.

## Stage B: fresh confirmatory intervention

- paired reps: **1800-1829**, n=30 fixed;
- fresh training environment seeds: **55000-55063**;
- fresh held-out environment seeds: **56000-56079**;
- Digits canonical 988-image training split;
- evaluation on all 809 canonical nontraining images;
- existing `SmallCNN`;
- AdamW lr=5e-3, weight decay=1e-3;
- 10 epochs, batch128;
- deterministic one-thread CPU;
- K=16 and Q=4 fixed in both arms;
- Stage-A-selected `c=0.05` embedded without modification.

At every fresh reference step, all 84 feasible subsets were recomputed on one common loss-hard reference model state. A pair was eligible only if both reference hardness and physical parameter diversity were within the fixed c=0.05 caliper. MAXNOV and MINNOV were then the higher- and lower-gradient-novelty members of the eligible pair with the largest novelty gap.

All **2,400 confirmatory reference steps** had at least one eligible pair. All 30 complete schedule pairs were sealed before any intervention arm trained. Two identical SmallCNNs were reset to the same initialization and trained on the fixed MAXNOV/MINNOV schedules. All 60 arm states were sealed before fresh held-out environments were constructed.

## Frozen gradient manipulation

Per replicate,

```text
N_r = mean_step(G_MAXNOV - G_MINNOV).
```

Result:

- mean gradient-novelty gap: **+0.203313**;
- SE: **0.002170**;
- two-sided t29 95% CI: **[+0.198874,+0.207752]**;
- one-sided p: **7.46e-38**.

The gradient manipulation passed decisively.

## Frozen hardness equivalence

The confirmatory equivalence margin was **±0.05 `z_hard` units**.

Result:

- mean MAXNOV-MINNOV hardness difference: **-0.001386**;
- SE: 0.000657;
- t29 90% CI: **[-0.002502,-0.000270]**;
- TOST lower p: **6.79e-35**;
- TOST upper p: **1.37e-35**.

The 90% interval is entirely inside [-0.05,+0.05], so reference hardness equivalence passed.

The raw selected-loss difference at the reference states was also small: mean **-0.000120 cross-entropy units**.

## Frozen physical-parameter-diversity equivalence

The known physical generator coordinates are rotation, x/y translation, blur, contrast, brightness and noise. The seven coordinates were standardized over the 64 training environments. For each feasible Q4 subset, physical diversity was the mean pairwise Euclidean distance among those standardized parameter vectors and was then standardized across the 84 feasible subsets within the reference step.

The confirmatory equivalence margin was **±0.05 within-step `z_param` units**.

Result:

- mean MAXNOV-MINNOV `z_param` difference: **+0.001186**;
- SE: 0.000713;
- t29 90% CI: **[-0.000025,+0.002397]**;
- TOST lower p: **1.61e-34**;
- TOST upper p: **6.31e-34**;
- mean absolute per-step `z_param` difference: **0.025284**.

Physical-diversity equivalence therefore passed.

In raw standardized-parameter distance units, the mean MAXNOV-MINNOV physical-diversity difference was only **+0.000480**, compared with **+0.492697** in the unmatched Issue #80 intervention. The major known parameter-diversity confound was therefore reduced by roughly three orders of magnitude in absolute mean difference.

## Frozen held-out result

Primary performance quantity:

```text
B_r = heldout_mean_accuracy_MAXNOV - heldout_mean_accuracy_MINNOV.
```

Result:

- mean held-out benefit: **+3.96992 percentage points**;
- paired SE: **0.44949 pp**;
- two-sided t29 95% CI: **[+3.05062,+4.88923] pp**;
- one-sided p: **5.10e-10**;
- positive pairs: **28/30**.

Thus all four frozen gates passed:

1. fresh reference matching feasible at every step;
2. gradient-novelty manipulation passed;
3. reference hardness TOST passed;
4. physical parameter-diversity TOST passed;
5. held-out mean performance passed.

Frozen decision: **PARAMETER-MATCHED GRADIENT NONREDUNDANCY SUPPORT**.

## Secondary outcomes

Descriptive only:

- clean MAXNOV-MINNOV: **+1.01772 pp**;
- p10 held-out: **+4.39184 pp**;
- minimum held-out: **+5.25752 pp**;
- mean schedule overlap: **0.46542**.

The p10/minimum metrics were not multiplicity-controlled primary endpoints and do not establish tail safety.

## Independent verification

A separate post-run audit downloaded all six evaluation artifacts and the aggregate artifact and verified:

- all six evaluation ZIP SHA256 digests against GitHub artifact metadata;
- all 30 schedule hashes;
- all 60 checkpoint file hashes;
- all 60 canonical state-tensor digests against training/evaluation tables;
- archived scientific source hashes;
- all 2,400 reference-step rows;
- all 4,800 per-environment held-out accuracy rows;
- every reference-step caliper bound (`|delta z_hard|<=0.05`, `|delta z_param|<=0.05`);
- per-environment reconstruction of mean/SD/p10/min with maximum numerical difference **1.11e-16**;
- aggregate paired quantities with maximum difference below **1.0e-16**;
- gradient t-test, hardness TOST, parameter-diversity TOST and held-out t-test independently recomputed.

Aggregate artifact SHA256: `0503bf1492c7dc8f35232974c91aad5423057b60a7d7e04d86c317756304334d`.

## Mechanistic consequence

Issue #83 materially strengthens the mechanism relative to Issue #80:

```text
same task/model/init/reference state
+ fixed K=16 and Q=4
+ tightly matched reference hardness
+ tightly matched known 7-D physical transformation diversity
+ high versus low model-conditioned gradient nonredundancy
    -> large learned-function difference
    -> +3.97 pp full-strength held-out mean
```

Within this tested Digits/SmallCNN geometric regime, the result is strong direct **causal-support evidence for a contribution from model-conditioned gradient nonredundancy beyond mean hardness and the known physical transformation diversity**.

The wording must remain narrower than “gradient novelty is the unique causal mediator.” The treatment can still differ in unmeasured schedule properties such as gradient norms, higher-order loss-distribution structure, exact rank composition within ranks2-10, or other latent environment properties. The downstream learned-function mediator between schedule allocation and final behavior is also still unidentified.

## Next falsification

The highest-value next test is no longer another parameter-matching refinement on the same SmallCNN. The mechanism should now be replicated under a materially different model parameterization or task while retaining the fixed-Q / common-reference / matched-hardness / matched-parameter-diversity design.

A fast cross-architecture replication on the Digits MLP can test whether this direct intervention is specific to convolutional feature geometry. A stronger but more expensive next step is an analogous intervention in a different task family where an explicit environment-parameter representation is available.
