# Prospective early Muon→AdamW switching window: T40/T60/T80/M

Registration required BEFORE any fresh model training or heldout outcome. Original source SHA256 and this protocol SHA256 must be attached to GitHub Issue before `freeze`. The first complete valid run is authoritative; malformed states or missing grids are `INVALID`, not scientific failures. No optional sample extensions, changing thresholds, or test-driven tuning.

## H — three directional primary CE contrasts

On a historical Digits/MLP task and 200 applied optimizer updates, **T60** (Muon 1..60, AdamW 61..200 on hidden 128×64 weight) will have lower mean transformed-heldout CE than (H1) never switching `M`, (H2) switch `T40`, (H3) switch `T80`. Improvements are `CE(base)−CE(T60)`. All three one-sided Student t tests with 11 degrees of freedom against zero form one Holm3 family. Each `SUPPORT` iff its adjusted p<0.05 AND its observed mean difference ≥0.01 CE. Otherwise `NOT SUPPORTED`; do not infer equality. A distinct *interior peak at T60 among 40/60/80* requires **both H2 and H3** support. The 0.01 effect threshold is a point-estimate gate, not a test against true effect≥0.01.

### Clean image guardrail — separately registered secondary endpoint

Compare clean CE of T60 against never-switch M. Define clean penalty `CE_clean(T60)−CE_clean(M)`. One-sided Student t11 noninferiority test H0: true penalty ≥0.05 CE, H1: penalty<0.05 CE. A guardrail PASS iff p<0.05 AND mean penalty<0.05; a guardrail failure is not evidence that penalty necessarily exceeds the margin. It is *not* part of the three-way Holm family and cannot rescue a failed H. If claiming joint shift benefit **and** clean protection, both an applicable shift improvement and guardrail must pass. Report all clean losses/accuracies as secondary. Also report the fixed w=0.8 shifted/clean CE mixture as descriptive only, chosen from prior exploratory work, not a registration success endpoint.

## T — fixed design (one bounded experiment)

**12 independently seeded transformation pools×4 per-pool initializations×4 policies=192 trained full states; exactly 200 applied updates per state**. Same sklearn Digits historical 800 fit images /809 *previously reused* test images; no independent new-image claims. 25 epochs, batch100, 4 transforms sampled from 64 training keys each update, 200 updates, 48 heldout keys per pool. Same minibatch and training-transform schedule across all four policies and all four initials within each pool. Exact initial tensors shared across four arms per `(pool,initialization)`; initialization seeds differ between pools (not crossed across pools, thus does not estimate stable seed identity interactions).

- Policies: `('M','T40','T60','T80')`, switch numbers `M=201, T40=40, T60=60, T80=80`; actual zero-index loop `active='M' if t<switch_at else 'A'`. Head/bias AdamW in all four policies.
- Initialization keys: `13800000000 + 10103*(4*pool+seed)`, pool0..11, seed0..3. No outcome-based choice.
- Training transformation keys: `2100000+20000*pool` through `+63`.
- Heldout transformation keys: `2105000+20000*pool` through `+47`, constructed **only after GLOBAL_SEAL**.
- Schedule base: `20900000000+1423*pool`, use same Torch permutation/NumPy four-env selection pattern as prior v2 sealed study. No tuning of model or optimizer.
- AdamW on hidden proposals and head/bias: lr 0.005, weight_decay0.001, betas (0.9,0.999), eps1e−8, foreach=False,fused=False.
- Muon hidden only: lr0.005, weight_decay0.001, momentum0.95,nesterov=True,ns_steps=5, eps1e−7, `adjust_lr_fn='match_rms_adamw'`; float32 weights/gradients and PyTorch2.10 CPU Muon Newton–Schulz bfloat16 backend. Reuse exactly the prior tested `code/switch_engine.py`.
- Every arm updates **both** AdamW and Muon hidden **shadow histories on the currently realized shared gradient** and applies only the active proposal. No reset/transfer/frozen-momentum artifact. The four arms pay identical two-shadow update costs but **do not represent single-optimizer production compute**. No GPU performance extrapolation.
- One deterministic CPU thread per worker, uncontrolled CPU frequency. Report actual Python, torch, NumPy, sklearn, SciPy, CPU/MKLDNN flags and source SHA. No quantization zero point.
- **Global train/test barrier:** all 192 full checkpoint file/tensor state digests, 38,400 applied updates, source/protocol/data split hashes and exact grid must be sealed before even constructing heldout environments. Final eval has 192×48=9,216 integer accuracy and float64 CE-sum rows, plus 192 clean rows, each denominator809. No performance peeking at intermediate pool outputs.

## D/C/U — reporting and scope

Aggregate 4 seeds×48 heldout environments within each pool, then take *12 pool-level* CE differences. These 12 (not 48 seeds or9,216 image-environment evaluations) are the inferential units. Preserve all 12 raw pool contrasts per primary test, pointwise two-sided95% t11 confidence intervals, Holm3 p and fixed PASS/FAIL decisions, clean guardrail one-sided p + interval and sign-flip as optional exploratory sensitivity. Any zero SE, nonfinite, missing grid, source/hash drift, evaluation-before-seal -> INVALID.

A result for fixed switch-time *policies* cannot prove a universal weight-space favorable region, a per-seed online switching controller, optimal time outside40/60/80, optimizer's polar geometry uniquely causes gain, equal-wallclock/FLOP speedup, robust behavior under new test images, or clean-safe switching. Alternative explanations include lr/update norms/momentum and differing future gradient trajectories. These must remain unresolved unless separately isolated.

## Variables / SI unit check

| Symbol | Meaning (日本語) | SI unit | Definition | Domain / premise | Type |
|---|---|---|---|---|---|
| p | 独立訓練変換群 | 1 | integer 0..11 | 12 groups | integer scalar |
| i | 初期化キー群内ID | 1 | 0..3 | fixed independent per pool | integer scalar |
| t | 適用更新番号 | 1 | 1..200 | integer | integer scalar |
| s | 切替更新番号 | 1 | 40,60,80,201 never | one-way | integer scalar |
| L_j | 評価CEのプール平均 | 1 | transformed test 4 seeds×48env×809 images | nonnegative | real scalar |
| D | shift CE改善 | 1 | CE(base)−CE(T60) | real | real scalar |
| C | clean CEの悪化 | 1 | CE_clean(T60)−CE_clean(M) | real | real scalar |
| m | clean非劣性マージン | 1 | 0.05 CE | fixed pre-outcome | real scalar |
| n | 独立プール数 | 1 | 12 | positive integer | integer scalar |
| SE | プール差平均の標準誤差 | 1 | sample SD of12 differences /sqrt12 | positive for tests | real scalar |
| w | 混合目的の変換比率 | 1 | 0.8 (descriptive) | [0,1] | real scalar |
| u_c | 画像/HWを含む合成標準不確かさ | 1 | not estimable here | unknown | unavailable scalar |

Unit check: CE is unitless natural-log loss (SI1); CE contrasts, means, SE, CI, guardrail margin and weighted CE have SI1; accuracy percentage points are different. Step counts/keys dimensionless; seconds/FLOP energy are not CE without an explicit conversion or equal-resource test.
