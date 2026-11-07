# Early Muon→AdamW switch timing, separate clean guardrail (2026-11-08)

**Status:** [Prospective GitHub Issue #133](https://github.com/Unjuno/seed-guided-optimization/issues/133) registered before all new outcomes; first complete valid run. This summary is committed **after** the outcomes, and does not retroactively alter a decision. Complete training code/checkpoints and invalid-attempt handling are in the owner's conversation evidence ZIP, not in this statistics-only GitHub branch.

## H / T — fixed design

The prior Issue131 failed to support a peak at100 because earlier T80 was better on transformed CE; this study prospectively compared **M (hidden Muon throughout) versus Muon→AdamW at 40,60,80 applied updates**. Digits historical image split: 800 fit and the **same reused 809 test images**; all training initialization keys, transformation keys and schedules are fresh. New environments are **not new independent test images**.

Exactly **12 independent transformation pools×4 distinct initial-weight keys/pool×4 methods=192 final states**, 200 applied updates per state, same initial weights/mini-batches/four selected fit transformations within a 4-arm block, hidden weight matrix 128×64. Both hidden optimizer shadow histories are maintained with each realized gradient for every arm; only the selected update is applied. AdamW for head/bias throughout, 1 CPU thread per worker, 4 parallel workers. This isolates *policy configurations under dual-shadow computation*, not equal-single-optimizer production cost. Muon uses PyTorch2.10.0+cpu, NS5/bfloat16 internally and float32 training; AdamW uses float32 lr=.005, wd=.001. No seed optimizer or online gradnov is being tested.

All **192 model+two-optimizer state files** and **38,400 applied updates** were globally SHA256-sealed before any heldout transformation was generated. Each model then evaluated on48 fixed heldout transformations×809 images and clean809 images. **Inference unit is 12 pool-average differences**, not 48 model seeds or 9,216 environment rows.

## D — frozen 3 Holm3 one-sided t11 tests

CE gain = CE(baseline)−CE(T60), positive favors T60; test zero, with **registered point gain >=.01 CE** AND adjusted p<.05. All intervals below are pointwise two-sided95% t11.

| Primary contrast | Mean CE gain | 95%CI | Holm3 p | Frozen outcome |
|---|---:|---|---:|---|
| T60−M | **+0.0357093** | [+0.0150150,+0.0564037] | 0.00295387 | **SUPPORT** |
| T60−T40 | **+0.0204419** | [+0.0129043,+0.0279795] | 0.000139991 | **SUPPORT** |
| T60−T80 | +0.0042310 | [−0.0026339,+0.0110958] | 0.101060 | **NOT SUPPORTED** |

**The preregistered conjunctive T60 local peak (T60>both T40 and T80) is NOT SUPPORTED.** Do not infer equality with T80 or that T60 is globally optimal.

### Separately registered CLEAN noninferiority guardrail

For T60−M clean CE penalty, noninferiority margin **+.05 CE**. H0: true clean penalty >=.05; H1: penalty <.05, one-sided Student t11.

Mean penalty **+0.1087087 CE**, pointwise95%CI [+0.0596897,+0.1577277], p=.988422: **CLEAN NONINFERIORITY NOT SUPPORTED**. The observed 95% lower CI even exceeds the margin. A shifted-performance improvement is **not** a clean-safe joint improvement.

[Public 3 comparisons](../results/switch_early_20261108_primary.csv), [36 independent pool differences](../results/switch_early_20261108_pools.csv), [12 separate clean-guardrail differences](../results/switch_early_20261108_clean_guardrail_pools.csv) and `python scripts/check_switch_early_20261108.py`. This checker recomputes **only** public CSV statistics, not hashes, weights, inference or pre-outcome registration validity.

## Secondary factual observations

| Policy | Shifted CE (low best) | Shifted accuracy | Clean CE (low best) | Clean accuracy |
|---|---:|---:|---:|---:|
| **M** | 1.201560 | 61.442% | **0.906225** | **67.228%** |
| T40 | 1.186292 | 61.005% | 1.067510 | 61.351% |
| **T60** | **1.165851** | **61.802%** | 1.014934 | 62.925% |
| T80 | 1.170082 | 61.698% | 0.976935 | 64.040% |

Thus no one policy has the lowest sample CE on **both** clean and transformed sets. Earlier transformation optimization may impair clean performance.

**Exploratory (POST-OUTCOME, not a confirmed controller):** with the convex sample-average mixed CE weighted `w` to shifted images and `1−w` to clean images, M has the lowest mean at `w<0.691956`, T80 at `0.691956<w<0.899812`, and T60 at `w>0.899812`. T40 is sample-mean Pareto dominated. Breakpoints come from crossings of four observed loss lines; they were calculated **after final test results** and are not generalized parameter-region boundaries or deployed optimizer switch rules. At the pre-specified *descriptive* `w=.8`, T80 has the lowest sample mixed CE (1.131452). A 12-pool bootstrap picks T80 in 73.35% of 10,000 resamples at this weight; its conditional preference near w=.9 is unstable.

## ERROR CHECK and provenance

- First pre-outcome GitHub Issue133 supplies 6 exact source hashes and fixed H/T/D. Local frozen `freeze.json` SHA256: `6f8565b4566d6607df0d4213ccaf3b5069a11714df763e58d8381f8036f5196d`, config digest `12be5313229fa15e1894e63ccba18bce550421983571b743896f2b2f258ec616`, `GLOBAL_SEAL.json` SHA256 `bfdffb9d947e1eb826ddc98a11f8492726ee740d8c6f50f366f570b255b36aee`. Sealing occurred **before all heldout construction**.
- Independent raw evidence verifier rerun 2026-11-08: **192 full state/tensor/file SHA256s, 38,400 correctly active optimizer steps, 15,360 matched common-prefix comparisons, 9,216 heldout integer counts+CE sums and192 clean counts; 3 t/Holm and clean guardrail** all match. 8/8 unit tests PASS.
- Preserved original replay report: **3 full 200-update bitwise training reruns (including both optimizer histories) and two separate raw-image reinferences**, passing on this CPU/backend. This is NOT full 192-model independent retraining.
- Review evidence ZIP **2,007,924 bytes**, SHA256 `947a9e9d04466d3d7b6297f0800a823828321da323d1181cb678d4d52ee25b42`; full 192-state evidence ZIP **29,134,326 bytes**, SHA256 `b25d2dba51dd4419dc92bfc039a62696328243ad9506b857e288f1a315c22f15`. ZIPs are owner-visible conversation artifacts, **not mirrored into GitHub**.
- Runtime AMD EPYC9V74, clock uncontrolled, Python3.13.5, torch2.10.0+cpu, NumPy2.3.5, SciPy1.17.0, scikit-learn1.8.0; torch one CPU thread/worker and MKLDNN enabled.

## C / U — claims excluded

The study has not identified a unique optimizer-favorable region, a stable universal transition time, an optimal initial-seed selector, a clean-safe switch controller, new-image/task transfer, or equal-FLOP/GPU advantages. Alternative explanations include optimizer hyperparameters, update norms, momentum history, and how trajectories diverge after switch. Small samples of 12 generator pools are fixed to reused test images; combined standard uncertainty `u_c` across images, tasks and backend is **not estimated**.

**Next experiment suggested:** test smoothly blended hidden Muon/AdamW proposals instead of hard switching, with **new fixed seed pools**, and jointly require an improvement in transformed CE plus a prespecified clean noninferiority margin. Alternatively fresh-image transfer. Ensure total proposals, gradient evaluations and heldout image barrier are matched. Do not tune on this same test.

## Variable table / SI unit audit

| Symbol | 意味（日本語） | SI単位 | 定義 | 定義域・前提条件 | 型 |
|---|---|---|---|---|---|
| p,i | 環境群と初期化識別子 | 1 | 12群, 各4seed | 正整数範囲 | 整数 |
| t,s | 適用更新番号、固定切替時刻 | 1 | t∈1..200,s∈{40,60,80,never} | 固定方策 | 整数 |
| S_j | 変換画像の平均CE | 1 | 48環境×809画像×4seed平均 | 有限・非負 | 実数 |
| C_j | clean画像の平均CE | 1 | 809画像×4seed平均 | 有限・非負 | 実数 |
| w | 変換損失の重み | 1 | 0..1 | 実験後の重み境界は探索的 | 実数 |
| Δ | 対照−対象CE改善 | 1 | 正なら対象で損失減少 | 有限実数 | 実数 |
| m | clean非劣性マージン | 1 | +0.05CE | 事前固定 | 実数 |
| SE | 12群差の標準誤差 | 1 | 標本SD/√12 | 有限・非負 | 実数 |
| u_c | 異なる画像・装置を含む合成不確かさ | 1 | 本試験で推定不能 | 未知 | 未確定 |

Unit check: natural-log CE, difference, weight, margin and standard error are all dimensionless (SI1). A +0.01 CE gain is **not** a +1 percentage-point accuracy gain. CPU time measured in seconds, FLOPs as operations count: no equal-time/energy comparison was registered.
