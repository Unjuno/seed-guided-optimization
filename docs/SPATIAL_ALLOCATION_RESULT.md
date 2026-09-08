# Spatial-factor allocation without gradient scoring: Issue86 result

## Frozen result

**SPATIAL ALLOCATION ALTERNATIVE SUPPORTED**.

Preregistered Issue86; implementation PR87; first complete run34174483847;
source1171dec1230487fa4d52fb7abf829c7650fd51e5;
protocol SHA256953adbf061f4cc3bbe9600a0ca9384ed887f64a77c0136de1b56d24c62814b6b.

The hypothesis arose from an explicitly post-hoc observation in
[Issue83](PARAMETER_MATCHED_RESULT.md): matching scalar total parameter diversity
left the allocation of variance among translation, noise and blur imbalanced.
This follow-up uses new repetitions and environment seeds, not new image identities.

## Design and measurement conditions

Digits canonical train988/nontraining809 images; SmallCNN; AdamW lr0.005,
weight decay0.001;10epochs;batch128;16candidate environments;4selected.
Reps1900-1929, training seeds57000-57063, evaluation seeds58000-58079.
No interim stopping, severity search, added repetitions or outcome-based retuning.

A common loss-hard reference defines the same84 feasible subsets per step:
retain the hardest candidate and choose3 among loss ranks2-10. Eligible subset
pairs must meet both fixed0.05 calipers on standardized reference mean loss and
standardized total seven-coordinate mean pairwise distance.

Four schedules are fixed before intervention training:
- G_high/G_low: largest positive/negative reference gradient-novelty separation;
- T_high/T_low: largest positive/negative x/y translation-variance separation,
  selected WITHOUT using gradient novelty as a score.

The translation score is the sum of the population variances of globally standardized
x/y translation coordinates across the4 selected environments. It does not use
gradient scores; reference-loss eligibility still depends on a trained model.
Gradients are computed for ordinary training and for cross-measurement diagnostics,
so this experiment is NOT a no-gradient-training or measured speedup claim.

All30 reference schedules were globally sealed before the120 intervention models
trained. Those120 states plus30 reference baseline states were globally sealed
before any heldout evaluation. Each intervention starts from the same initialization
and uses the same minibatch/candidate schedule within its repetition.

## Primary results

Benefits are high-minus-low heldout mean accuracy in percentage points (pp).
Two-sided95% t29 intervals and preregistered one-sided p-values:

| Selector used to define the schedules | Benefit | Paired SE | 95% CI | One-sided p | Positive repetitions |
|---|---:|---:|---|---:|---:|
| Reference gradient novelty | +3.852029 pp |0.452239 pp|[+2.927096,+4.776962] pp|1.10003e-9|29/30|
| Translation variance only | +1.768644 pp |0.581166 pp|[+0.580026,+2.957263] pp|0.00246748|23/30|

Both performance conditions pass. Both families also pass both reference-balance
TOSTs inside the frozen±0.05 margins and their respective manipulation tests.
The primary decision was a conjunction: neither endpoint alone could upgrade it.
Full gate estimates are in [spatial_balance30.csv](../results/spatial_balance30.csv).
A balance PASS means equivalence within the stated margins, not exact matching.

## What the reference baseline changes

These are prespecified SECONDARY descriptions, not additional primary claims:

| Schedule relative to loss-hard reference | Difference | 95% descriptive CI |
|---|---:|---|
| G_high | +1.752627 pp |[+0.614872,+2.890382] pp|
| G_low | -2.099402 pp |[-3.068731,-1.130074] pp|
| T_high | +0.639318 pp |[-0.599578,+1.878214] pp|
| T_low | -1.129326 pp |[-2.202424,-0.056229] pp|

Therefore +3.852 pp must NOT be reported as an online-gradnov improvement over
loss-hard. It is a high-versus-low reference-selected schedule contrast, with both
higher high-arm performance and lower low-arm performance relative to the reference.
Translation high-versus-low is supported; translation high-versus-reference is not
established by its interval. Secondary comparisons are not multiplicity-adjusted.

Reference and intervention training can use different hosted CPUs. The four
intervention arms within each repetition share a runner; reference comparisons
have this additional backend caveat and are not matched for total computation.

## Deepening the discovery: the variables remain coupled

Prespecified cross-measurements:

| Schedule contrast | Reference gradient-novelty gap | Translation-variance gap |
|---|---:|---:|
| G_high minus G_low | +0.208500 | +0.616683 |
| T_high minus T_low | +0.105610 | +1.151191 |

All quantities in this table are dimensionless; translation variance is in squared
standardized-coordinate units, not pixel-squared or SI distance units.

Thus a translation-only score also creates a positive gradient-novelty gap.
The observed alternative is **an alternative scoring procedure**, not a demonstrated
causal route that bypasses gradient geometry. Conversely, the larger gradient-family
high/low performance contrast does not show that translation is irrelevant.
The descriptive difference of contrasts is+2.083385 pp,95% CI[+0.971133,+3.195637].
Different low baselines and correlated physical factors prevent interpreting this
as an identified mediated fraction or a head-to-head ranking of the two high arms.

## H / T / D / C / U

H: Useful factor allocation can produce a performance contrast without scoring
candidate subsets by gradient novelty.
T: Fresh30 paired repetitions; four interventions and a reference; fixed hardness
and scalar diversity calipers; global schedule/state barriers.
D: All balance/manipulation gates and both positive primary performance tests pass.
C: Translation score may act through correlated gradient geometry or other physical
factor redistribution; low-arm degradation may contribute. Neither is eliminated.
U: Fixed images and environment pools, stochastic initialization/schedules, reference
versus intervention trajectories, and backend differences limit interpretation.
The primary paired SEs are0.452239 and0.581166 pp; the t29 coverage factor is2.04523.
A total combined uncertainty including datasets/backends has NOT been estimated.

## Independent verification / ERROR CHECK

Six evaluation ZIP digests and the aggregate ZIP digest matched artifact metadata.
The archived experiment source is byte-identical to the locally self-tested source
(git blob04676c73e4d8499ce433d81b5a8799ab1b6f0ffc).
Checked all150 checkpoint-file/tensor digests,30 schedule reconstructions,
4800 reference rows and12000 environment rows. Recomputed frozen aggregate CSVs
matched exactly under round-trip parsing; environment reaggregation error0.0.
Maximum translation-score reconstruction discrepancy4.44e-16.
Independent SciPy paired t/TOST calculations reproduced the frozen decision.
No model was retrained or raw-image inference rerun for the independent check.

Runtime: Python3.12.14,torch2.10.0+cpu,numpy2.3.5,pandas2.2.3,scipy1.17.0,
sklearn1.8.0; deterministic algorithms; one CPU thread. Intervention runners used
AMD EPYC7763,9V74 and9V45. CPU clocks were not controlled; this is not a timing benchmark.

Recompute primary statistics:

```bash
python experiments/check_spatial_allocation_evidence.py
```

[Public primary paired data](../results/spatial_primary_paired30.csv),
[all aggregate endpoints](../results/spatial_summary30.csv),
[decision](../results/spatial_decision30.csv).

The next discriminating target is a fixed-Q intervention that varies reference
gradient novelty while also balancing the spatial allocation score, with feasibility
established on an independent training-only block. It is not performed by Issue86.
