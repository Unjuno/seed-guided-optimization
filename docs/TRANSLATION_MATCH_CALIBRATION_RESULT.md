# Issue89: all-step translation matching failed before performance evaluation

Authoritative run34250390284; implementation PR90; source20be47206d459b3f1052d49557e4eacb16be2f2e.
Frozen decision: **TRANSLATION MATCH CALIBRATION FAIL**.
No confirmatory intervention arm was trained and no heldout environment was evaluated.
The planned strict StageB was not executed. The decision is not replaced by Issue91.

## Frozen calibration and results

Ten new reference repetitions2000-2009;80steps each. Digits988training images;
SmallCNN, AdamW lr.005/wd.001,10epochs,batch128,K16,Q4,one CPU thread.
Training environment seeds59000-59063. The two original calipers remain.05 for
reference standardized mean loss and standardized total physical diversity.
The additional caliper limits the raw sum of standardized dx/dy population variances.
The selected pair maximizes reference mean pairwise gradient novelty within those bounds.

| Translation caliper | Feasible steps /800 | Mean G gap on feasible steps | Frozen PASS |
|---:|---:|---:|---|
|.05|469|.0696697|no|
|.10|649|.0814970|no|
|.20|753|.0990260|no|
|.30|782|.1129192|no|
|.50|799|.1316488|no|

The frozen rule required100%feasibility and mean G gap at least.10. None qualifies.
Reported conditional gaps exclude infeasible steps; they do not override feasibility.
All score units are dimensionless, not physical distances or accuracy differences.

## New support-overlap diagnosis (post-hoc)

Each of800reference steps archived all84eligible subsets, permitting exact finite
enumeration rather than speculation about a missing match. At rep2003,step70,
14subset pairs pass the two original H/P calipers, but their smallest attainable
absolute translation difference is.5792881483. Thus even.50cannot admit a pair.
This is a failure of available matched support in this design, not evidence that
novelty has no performance effect and not a claim of universal non-identifiability.

The median minimum attainable translation difference is.038045; the90th percentile
is.158750 and the maximum.579288. A single all-step rule is limited by its worst
reference state, even when most states allow much tighter matching.

## Separate next design, not rescue

The exploratory c_T=.20support is753/800steps(94.125%). Assigning an identical
loss-hard subset to both arms at the other47steps gives an unconditional mean
reference G gap.0932083; fallback contributes zero contrast without deleting updates.
Issue91 registers a new fresh block using exactly this.20caliper and common fallback,
with frozen pooled90%and per-replicate80%support gates. That changes the intervention
and estimand, so Issue89 remains a failed strict-feasibility experiment.

## Independent verification / uncertainty

Artifact SHA256:0d67964ebeb619a3716eb755a275710629020bfb4d99eb8433212d8ddca60872.
Five manifests, scientific source hashes and67,200subset rows were checked. Independent
pair enumeration reproduced all grid feasibility counts and maximum feasible G gaps.
No new training or raw-image inference was used in the audit.

The policy design uses training-only calibration, not arm outcomes. It still adapts
to one dataset/generator family and therefore needs fresh confirmatory repetitions.
Reproducibility is conditional on pinned software and hardware paths; no speed claim.

Recompute from extracted evidence:

```bash
python experiments/audit_translation_support.py --evidence-dir EXTRACTED_CALIBRATION --output-dir support_audit
```
