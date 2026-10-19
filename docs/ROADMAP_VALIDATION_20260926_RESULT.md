# Frozen six-pool roadmap validation: completed results and independent audit

Experiment: GitHub Actions [run 36202615155](https://github.com/Unjuno/seed-guided-optimization/actions/runs/36202615155), completed **2026-09-26 UTC**. Independent read-only numerical and integrity audit: **2026-10-19**. This is the first valid completed run under Issue #124 and Issue #125; no new training was performed during the October audit. The original experiment protocols and earlier negative results remain unchanged.

All 18 training jobs, three global pre-evaluation seals, evaluation shards and three aggregation jobs succeeded. All **840 final model states** were sealed before their study's heldout environments were constructed. The official scientific decisions below are separate from GitHub Actions' technical success.

## Study T: Digits online clean-versus-minimum tradeoff (Issue #124)

60 fresh training blocks, two policies (120 models), six independently seeded environment pools with 10 paired blocks each, 80 heldout transformations per pool. **Primary unit: six pool averages**, t distribution with five degrees of freedom. Comparisons are gradnov minus hardest-plus-random-three anchor selection.

| Preregistered endpoint | Mean difference (pp) | Pointwise 95% CI (pp) | One-sided p | Frozen result |
|---|---:|---|---:|---|
| Minimum accuracy across 80 heldout environments | **+4.3016** | [+2.3192, +6.2840] | 0.0012762 | PASS |
| Clean-image accuracy | **−7.5052** | [−10.4424, −4.5679] | 0.0006135 | PASS |

**Decision: CLEAN-TAIL POLICY TRADEOFF SUPPORT.** Both conjunctive preregistered tests pass. This is not universal worst-case or adversarial robustness; the minimum is over a prespecified set of 80 generator draws.

[Complete hashed raw evidence](https://github.com/Unjuno/seed-guided-optimization/actions/runs/36202615155/artifacts/10893229927). Artifact SHA256: `9984916780cad0cb34b3a22b86027e870b3b1a0caba4e5c2676bb7cb3e77c643`.

## Study F: native-resolution Fashion-MNIST transfer (Issue #125)

Official Fashion-MNIST image files/checksums; **988 selected training and 809 selected test images**, 28×28 native input, SmallCNN with adaptive pooling, seven-coordinate transformation generator with spatial scale adjusted to 28/8. This is **not** a full-data-set benchmark or a universal new-task replication. Six new environment pools, 60 paired blocks, four methods, **240 states**. For all three preregistered one-sided hypotheses, test whether the difference is **greater than +0.5 percentage points**; Holm3 is one family.

| Directional contrast and endpoint | Effect (pp) | Pointwise 95% CI (pp) | Holm-adjusted p | Decision |
|---|---:|---|---:|---|
| anchor_random − loss_hard, mean heldout accuracy | **+1.2008** | [+0.7362, +1.6653] | 0.014873 | PASS |
| gradnov − anchor_random, mean heldout accuracy | −0.0186 | [−0.4810, +0.4437] | 0.982773 | **NOT SUPPORTED** |
| gradnov − anchor_random, minimum heldout accuracy | **+3.3725** | [+1.5451, +5.1998] | 0.014873 | PASS |

A simple gradient-free anchored random policy improves average accuracy relative to loss-hard on this protocol, while gradnov improves the prespecified finite-pool minimum relative to the anchored policy. The corresponding mean advantage for gradnov is **not** established. Because the methods generate their own trajectories and loss/physical allocation, no unique gradient-information mechanism follows.

[Complete hashed raw evidence](https://github.com/Unjuno/seed-guided-optimization/actions/runs/36202615155/artifacts/10893078811). Artifact SHA256: `19a3f62e378a9a0129f5475f5c3b76845188be7db69a0152ab277f199a5a0093`.

## Study O: objective × estimator × optimizer configuration (Issue #125)

Digits/SmallCNN, six environment pools, 60 blocks, eight fixed arms = **480 states**. AdamW is configured with lr=0.005, decoupled weight decay; SGD with momentum is configured with lr=0.05, coupled weight decay. Four candidate-contributor arms are deliberately contrasted with sixteen-contributor arms; **not an equal-compute comparison**. The seven prespecified effects/interactions use two-sided pool-level t5 tests, Holm7 correction.

| Registered contrast | Effect (pp) | Pointwise 95% CI (pp) | Holm7-adjusted p | Decision |
|---|---:|---|---:|---|
| AdamW objective main | −0.4328 | [−1.0164, +0.1507] | 0.297626 | NOT SUPPORTED |
| AdamW estimator main | **+4.6983** | [+4.3918, +5.0047] | 0.00000139 | PASS |
| AdamW objective × estimator | **−1.9480** | [−3.1333, −0.7626] | 0.041468 | PASS |
| SGD objective main | −0.9937 | [−2.2574, +0.2700] | 0.297626 | NOT SUPPORTED |
| SGD estimator main | **+1.1642** | [+0.6061, +1.7223] | 0.018201 | PASS |
| SGD objective × estimator | −2.0543 | [−3.7816, −0.3269] | 0.112779 | **NOT SUPPORTED after Holm7** |
| Three-way configuration difference | −0.1063 | [−1.9744, +1.7618] | 0.889417 | NOT SUPPORTED |

Although the nominal two-sided SGD interaction CI excludes zero, it fails **registered familywise correction**. A CI alone cannot override the Holm7 decision. No difference in the interactions between the two optimizer configurations is established. Cross-configuration observations do not isolate optimizer algorithms because lr/weight decay also vary.

[Complete hashed raw evidence](https://github.com/Unjuno/seed-guided-optimization/actions/runs/36202615155/artifacts/10893249721). Artifact SHA256: `b39ad01ab00f527614eec479aae06d244d5e6ac7ebaed439daa671fefc1ea59b`.

## Independent October audit and reproduction

The **independent** audit did not call the original aggregation function to derive primary outcomes. It recomputed means, finite-pool minima, p10, lower-decile means, SD, clean fractions, pair/block contrasts, pool contrasts, t5 p-values, pointwise CIs, and Holm corrections **from integer correct counts**. All **12 registered primary tests** match the frozen aggregate. The auditor verified:

- **840/840** stored model tensor digests; **840/840** per-model 80-step selection traces, including exact float32 re-selection of the gradnov greedy policy, 16 unique candidates per step, weight normalization, anchoring/top-loss rules, and same candidate schedule across within-block methods.
- **68,040/68,040** count rows, exactly 809 images per row, correct pool/rep/method/environment grids; **1,788** file/source/hash checks and six global-seal bindings per study.
- Maximum absolute error reconstructing six published evaluation metrics from integer counts: **0**; maximum pool contrast discrepancy: **2.78e-17** in accuracy-fraction units.
- Training manifests explicitly say `heldout_constructed: false`, and the pre-evaluation global seals were produced before evaluation jobs according to Actions job dependencies. The audit did not independently retrain models or repeat image inference.
- Scientific runner: Python 3.12.14, PyTorch 2.10.0+cpu, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, deterministic one CPU thread, MKLDNN enabled. Hosted CPUs included AMD EPYC 7763 and 9V74; **clock frequencies uncontrolled**, no hardware speed results.

**Low-cost public verification:**

```bash
python scripts/check_roadmap_validation_results.py
```

This checks only the public six-pool CSVs and twelve statistical decisions, **not** model/source hashes, inference or retraining. Raw complete artifacts are subject to GitHub Actions' retention limits; do not treat the artifact links as permanent archives.

## H / T / D / C / U interpretation

- **H**: Frozen directional/interaction hypotheses in Issues #124/#125.
- **T**: A fixed 60-block design per study, grouped into six independent environment pools. Training images and transformed environments are shared within each pool. Repetitions/pools were not added after inspection.
- **D**: Original preregistered t5 and Holm criteria; a software workflow success is distinct from a scientific PASS.
- **C**: Physical-factor allocation, selected-hardness/rank, downstream learning trajectories, optimizer state, and other geometry remain rival explanations. Unique-mediator identification is **not** achieved.
- **U**: Six pools are few. Reported paired uncertainty is conditional on fixed images, generator, model and CPU backend; it is **not** combined uncertainty across unseen datasets/accelerators. No worst-case, universal, equal-wall-clock or GPU-speedup claim.

## Variable table and unit check

| Symbol | Meaning (日本語) | SI unit | Definition | Domain / premise | Type |
|---|---|---|---|---|---|
| `rep` | 学習反復識別子 | 1 | registered block ID per study | study-specific 60 integers | integer scalar |
| `pool` | 独立変換環境群 | 1 | ten blocks per pool | integers 0–5 | integer scalar |
| `a` | 正答率 | 1 | correct count / 809 | real in [0,1] | real scalar |
| `B` | paired performance difference | 1 | accuracy(policy A) − accuracy(policy B) | [−1,1] per endpoint | real scalar |
| `n` | 主解析の群数 | 1 | fixed 6 | positive integer | integer scalar |
| `SE` | 六群平均差の標準誤差 | 1 | sample SD of six pool contrasts / sqrt(6) | positive finite | real scalar |
| `p` | 仮説検定のp値 | 1 | preregistered one/two-sided t5 | [0,1] | probability scalar |
| `u_c` | データ・環境・HWを通じた合成不確かさ | 1 | not identified in these fixed studies | cannot be numerically inferred | unavailable scalar |

Unit check: accuracy, contrast, SE and confidence intervals are dimensionless fractions. **1 percentage point = 0.01 accuracy fraction**; multiply fractional effects by 100 to display pp. No physical duration/energy efficiency is measured.

## Post-hoc six-pool inference sensitivity

The registration's **t5/Holm decisions above are frozen**. An additional after-outcome [exact sign-flip sensitivity analysis](SIX_POOL_SIGNFLIP_SENSITIVITY_20261019.md) found that the two Digits tradeoff components individually yield p=0.015625, but the Fashion t5/Holm3 PASS contrasts yield sign-flip Holm3 p=0.09375 and the optimizer t5/Holm7 PASS contrasts yield sign-flip Holm7 p>=0.21875. With only six independent pools, the latter two-sided Holm7 sign-flip family has minimum possible adjusted p=0.21875. Interpret this as **small-n model/discreteness sensitivity**, not a revision of preregistered outcomes or proof of a null effect. More independent environments/pools would be necessary for a stronger distribution-robust claim.

## Next research gate

The original *online* policy's gradient-information specificity, conditional losses/ranks, and physical-factor allocation still need an identifying design. Do not recast this result as a proof that head-gradient novelty alone causes the tail benefit. A new confirmatory intervention must be preregistered with fresh blocks and an independent matching/support audit. GPU matched-cost value remains blocked on authorized CUDA hardware. The owner requested **no X announcement yet**.
