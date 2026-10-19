# Exploratory policy-trace audit: why the tested online selectors differ

Date: 2026-10-19. **Post-outcome exploratory analysis** of the frozen Issue124/125 [run 36202615155](https://github.com/Unjuno/seed-guided-optimization/actions/runs/36202615155). No new model was trained, no outcome changed and no retrospective confirmatory gate was added. [Preregistered scientific decisions](ROADMAP_VALIDATION_20260926_RESULT.md) remain authoritative.

## Data and observable procedure

For each of 60 training blocks in each of two studies (Digits and native Fashion-MNIST), read the complete 80-step float32 losses, 16-candidate Gram matrix and selected-weight trace for each policy, directly from the frozen Actions artifacts. Use the exact local loss ranking at **each policy's own current model state** (rank 1 = hardest); the selected average rank is the weighted sum over Q=4 selected candidates. Compute selected loss standardized within that policy's own 16 candidates, and the mean of six pairwise cosine novelties among four selected normalized head-gradient signatures.

Both gradnov and anchored-random **always retain the current hardest candidate**. Anchored random samples the remaining three uniformly from the other fifteen without replacement. Gradnov computes a greedy score mixing current loss and the normalized nonredundancy of head-gradient direction. Candidate order and initial schedules were matched within a training block, but the models' **later states diverge**, so these are not fixed-state matched contrasts.

## Observed policy-specific treatment differences (descriptive)

| Observable | Digits: gradnov | Digits: anchor | Fashion: gradnov | Fashion: anchor |
|---|---:|---:|---:|---:|
| Selected mean loss rank (1=hardest) | 3.498385 | 7.006458 | 3.400104 | 6.996719 |
| Standardized selected CE mean | 1.096333 | 0.374416 | 1.176780 | 0.443473 |
| Mean selected pairwise gradient novelty | 0.573450 | 0.502299 | 0.540294 | 0.447168 |
| Selected raw CE averaged over 80 updates | 1.898968 | 1.799716 | 1.741124 | 1.657082 |

Across every one of 60 blocks in each study, gradnov's selected mean loss rank was smaller (harder) than anchor. The average gradnov-minus-anchor rank difference was **−3.508073** in Digits and **−3.596615** in Fashion. Standardized selected loss increased by **+0.721917** and **+0.733307** respectively. Head-gradient novelty also increased by **+0.071151** and **+0.093125**.

These were derived only from training-side traces, but the hypothesis to examine them was selected **after** seeing the experimental outcomes. The consistency is an important design clue, **not** a registered causal test or a new independent replication.

## Next identifying question

Can a policy using **loss ranks/hardness only, without gradients**, approximate gradnov's realized loss-rank/hardness *distribution* on fresh reference trajectories while leaving enough gradient-direction separation for a registered heldout comparison?

Stage A should audit candidate support and selected-factor overlap **without evaluation outcomes**, fixing the rank-only policy from historical traces first. A future Stage B, if eligible, should use fresh blocks/pools and predeclared simultaneous mean/finite-minimum/clean endpoints with independent t5 or another justified analysis. If sufficient overlap cannot be obtained, report **identification failure**, not absence of a gradient mechanism. Per-step identical ranks are not an achievable control in a strict unique-rank candidate list: selecting exactly the same ranks selects exactly the same candidates. A distribution-level policy is a distinct, weaker identification claim.

## H / T / D / C / U

- **H**: Exploratory possibility that hardness allocation and gradient directional information jointly explain policy behavior.
- **T**: Existing frozen records, 60 blocks and 6 environment pools per task, no new training or heldout access.
- **D**: No confirmatory PASS/FAIL on this exploratory trace analysis. Future experiments require preregistration and fresh data.
- **C**: Different downstream model states, physical environment selection and AdamW update paths are uncontrolled.
- **U**: These are descriptive trace means. No population CI or hardware-general uncertainty is claimed.

## Variable table and dimensional audit

| Symbol | Meaning (日本語) | SI unit | Definition | Domain / assumption | Type |
|---|---|---|---|---|---|
| Q | 更新採用候補数 | 1 | exactly 4 for both policies | integer 4 | integer scalar |
| r | 選択した候補の損失順位 | 1 | rank of each of 16 within-policy losses | integers 1..16 | integer scalar |
| H | 選択損失の候補内zスコア | 1 | selected CE mean standardized within policy/step candidate losses | finite | real scalar |
| G | 選択勾配の平均方向novelty | 1 | mean of 1 minus selected pairwise head-gradient cosine (6 pairs) | typically [0,2] | real scalar |
| CE | cross-entropy損失 | 1 | policy-specific mean CE over selected environments | nonnegative | real scalar |

All quantities here are dimensionless; rank and normalized CE cannot be subtracted as if they shared meaning. A larger CE is not automatically a more informative update. The original registered performance results remain in [the result report](ROADMAP_VALIDATION_20260926_RESULT.md).
