# Weight-initialization Seed × Optimizer × Horizon: locally frozen 2026-10-30 pilot

**Evidence status:** Complete in a local ChatGPT experiment archive; **not yet independently retrained outside that environment**, and **not externally preregistered**. The full raw weights, training/evaluation code, original local frozen protocol, status hashes and failing partial attempt are supplied to the owner as review and complete-evidence ZIPs in the originating chat. Do **not** reinterpret this later GitHub commit as a pre-outcome registration. This report is a reviewable scientific log only; remote raw-evidence import remains pending.

Tracked in [Issue #127](https://github.com/Unjuno/seed-guided-optimization/issues/127).

## What differs from existing SGO

Existing SGO mostly controls **image-transformation/environment seeds**. The prior Muon comparison tested **online gradnov selection × optimizer**. This study directly changes the **initial weight seed**, and deliberately uses fixed uniform sampled environments (no gradnov), eliminating policy differences to isolate initial-state compatibility. Seed key integers do not numerically encode quality.

## Prospective-in-session design, H/T/D

- **10 fixed initialization keys** `4000000000+1337*i`, each crossed with **6 independently drawn transformation pools**, under **AdamW / Muon+AdamW / SGD+momentum** = 180 training trajectories. Fixed Digits 988 train/809 evaluation image split, 8×8 MLP64→128→10. 25epochs, 200 optimizer updates, identical within-pool 4-environment exposure and minibatch sequence for all initials/optimizers.
- 3 checkpoints each: **20 / 80 / 200 updates**, 540 globally sealed states. All heldout transforms created *after* global seal. Each checkpoint evaluated on **48 heldout environments** (25,920 integer counts) and clean Digits (540 counts), denominator809.
- AdamW all parameters, lr.005 wd.001; Muon hidden 2D matrix only lr.005 momentum.95 NS5 and AdamW head/bias; SGD momentum.9 lr.05 wd.001 all. These are **optimizer configurations**, not matched learning-rate/compute or Muon-orthogonalization-only causal comparisons.
- Primary **H1:** sample SD across 10 initialization identities of six-pool-mean, within-pool-centered Muon-minus-AdamW heldout mean accuracy at update200. **H2:** corresponding seed-specific SD on the extra Muon-minus-AdamW improvement from update80 to200. Each requires a **point SD≥.005 (0.5 percentage point)** and a Holm2-adjusted p<.05 from **9,999 locked seeded Monte-Carlo identity-label permutations independently inside each pool**, shared across horizons. The permutation null is *no consistent cross-pool seed identity*, NOT SD≤.005. Exchangeability of seed identity under the null is an assumption.

## Frozen local results (not after-outcome redesigned)

| Endpoint | Seed-specific SD (pp) | Nominal permutation p | Holm2 p | Registered label |
|---|---:|---:|---:|---|
| H1: optimizer-specific final initialization effect | **0.610598** | 0.0116 | **0.0116** | **SUPPORT** |
| H2: optimizer-specific change from update80→200 | **0.577523** | 0.0046 | **0.0092** | **SUPPORT** |

These are SDs of across-seed *relative optimizer effects*, **not** mean accuracy gains.

Descriptive heldout mean accuracies over six pools × ten seeds: 

| Steps | AdamW | Muon configuration | SGD configuration |
|---:|---:|---:|---:|
| 20 | 34.56% | 34.55% | 21.77% |
| 80 | 44.17% | 44.58% | 38.65% |
| 200 | 56.98% | 60.98% | 47.02% |

Descriptive ten-seed rank correlation early step20→200 was AdamW −0.358, Muon −0.055, SGD +0.321. This is *not* a preregistered test or evidence that early ranking reliably predicts final ranking.

## Provenance, incident, and independent audit

Local pre-outcome frozen SHA256s:
- protocol `95f1ae9dab6a6b2089d502852586a673b06f6764bbda18fb750144610ea70b4c`
- training/evaluation code `072ef0e8df19f020486f9f496d2b2b56d1d4cafcc4a156ced16abd4e3300aaad`
- reused `common.py` source `ebd2e760dc30f23e1161c0718b192ccfbbf2f70f530f1233e5459f698151bb4a`
- freeze `d354fd4ed48a2506daf9f7efecabde4ad8e413e49f37b7a96909954576f6cff0`
- global seal record digest `76656944941c48d915059af5ac8574e8383691d1ecf3836919792a297510c16f`

An independent raw-integer-count auditor validated 540 checkpoint file/tensor digests, all 36,000 completed-cohort updates, all 25,920 heldout integer counts, all 540 clean counts and both frozen permutation decisions. Six preselected full trajectories were replayed, reproducing 18/18 checkpoint tensor digests bitwise on the same CPU/backend. **An initial foreground training command reached its tool execution timeout** after four pools: 25 incomplete checkpoint files were preserved and excluded; two remaining pools were computed under the identical frozen source, all six passed a global seal *before* heldout construction. The extra incomplete work is **not** part of the 36,000 valid-cohort steps.

Runtime: Intel Xeon Platinum 8573C, 1 deterministic CPU thread/worker, uncontrolled clock, torch2.10.0+cpu, Python3.13.5, MKLDNN enabled. No GPU/equal wall time claim.

## C/U / remaining scientific question

A limited consistent *initial-state × optimizer-configuration* interaction is supported, but **no optimizer-aware initial-seed selection policy was prospectively shown to improve performance**. The results do not identify a universally good seed, superior optimization basin, Muon's polar transform as unique mediator, or original SGO's environment-seed mechanism. Reused images and six generator pools constrain generalization; uncertainty over new architectures/images and hardware (`u_c`) is unknown. Distinguish *experiment complete* from *general optimization law proved*.

**Next:** new independently frozen sample with more initialization identities/new images, normalized update-magnitude/lr calibration and an optimizer-aware seed selector versus uniform and shared optimizer-agnostic controls. X announcement remains on hold.
