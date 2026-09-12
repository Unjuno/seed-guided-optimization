# Matched loss-rank relaxation confirmation — result

Issue #104 Stage B tested whether, while retaining the single hardest candidate and matching the registered loss-shape, physical and gradient-geometry summaries, moving the other selected slots toward lower-loss candidate ranks improves fresh heldout mean accuracy.

## Frozen decision

**MATCHED RANK-RELAXATION SUPPORT**

Authoritative GitHub Actions run: `34696556251`.

All 30 reference schedules were sealed before intervention training; all 60 intervention states plus 30 reference states were sealed before heldout construction. Fresh reference overlap was `0.9800`, with minimum replicate support `0.9375` and 48 identical-subset fallback steps.

## Registered manipulation and balance

High-R minus low-R mean selected rank gap:
- mean `1.769375` rank units;
- paired SE `0.018928`;
- 95% CI `[1.730662, 1.808088]`;
- one-sided p `7.94e-38`;
- positive pairs `30/30`.

The six registered matching gates all passed:
- selected-loss variance z mean difference `+0.002293`, equivalence +/- .05;
- physical-diversity z `-0.000513`, equivalence +/- .05;
- translation allocation `-0.013553`, equivalence +/- .20;
- mean gradient novelty `+0.000676`, equivalence +/- .05;
- centered residual span `-0.000077`, equivalence +/- .10;
- maximum pairwise opposition `+0.001542`, equivalence +/- .15.

The corrected centered-energy identity residual was at most `4.44e-15`.

Hardness H was intentionally not matched because it is an expected correlate of rank allocation. High-R minus low-R selected hardness was `-0.310040 z` (95% CI `[-0.318670,-0.301411]`): the high-R intervention is consistently less hard.

## Primary heldout result

Primary quantity: mean heldout accuracy(high-R) minus mean heldout accuracy(low-R), one value per independent training block (`n=30`).

- mean `+0.0090662` accuracy fraction = **+0.9066 pp**;
- paired SE `0.0038789`;
- two-sided 95% CI `[+0.0011330,+0.0169995]` = **[+0.1133,+1.6999] pp**;
- preregistered one-sided paired p=`0.01327`;
- positive pairs `18/30`.

Thus the frozen performance criterion passed.

Raw mean heldout accuracies averaged over training blocks were approximately:
- high-R: `0.536255`;
- low-R: `0.527188`;
- loss-hard reference: `0.506166`.

## Secondary tail/clean observation

The same intervention shows a potentially important mean-versus-tail tradeoff:

- high-R minus low-R clean accuracy: `+2.1879 pp`, 95% CI `[+0.1764,+4.1994] pp`;
- high-R minus low-R environment SD: `+0.004641`, 95% CI `[+0.000115,+0.009167]`;
- high-R minus low-R p10: `+0.1566 pp`, CI crosses zero;
- high-R minus low-R minimum accuracy: `-1.0589 pp`, 95% CI `[-2.4697,+0.3519] pp`.

Descriptively against the loss-hard reference, both relaxed arms improve mean and clean accuracy, but their environment-level SD is larger and their minimum accuracy is lower. These reference contrasts and tail endpoints were not the primary registered claim and must be treated as secondary/exploratory.

## Interpretation

This is the first fresh confirmatory support in the current mechanism sequence after centered span failed and pairwise opposition/all-factor isolation were infeasible. It supports a **hardness/rank-allocation explanation**: always concentrating the Q=4 budget on the hardest candidates is not optimal for mean heldout performance under this Digits/SmallCNN protocol.

However, rank and selected mean hardness co-move by construction here. The result does **not** establish candidate rank itself as the causal mediator. The next discriminating experiment must separate rank allocation from actual selected loss/hardness. The secondary tail pattern also motivates a preregistered mean-versus-tail tradeoff test.

Evidence artifact: `rank-allocation-confirmation-evidence`, artifact ID `10298882546`, SHA256 `a2fdbadcd40604c74537a38d7375c417d28489642e0369cd8a5628030e9c1e5c`.
