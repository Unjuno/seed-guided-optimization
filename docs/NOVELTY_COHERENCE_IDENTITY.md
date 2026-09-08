# Pairwise novelty measures directional coherence, not span dimension

This is an algebraic clarification of a measurement used by Issues80/83/86/89/91,
not a newly discovered mathematical theorem, causal identification result, or
replacement for their frozen decisions. It applies to the **mean pairwise**
head-signature score used in the schedule interventions. The original online
selector uses a greedy minimum-distance rule and a hardness term; it is not
identical to this averaged score.

## Variables and implementation assumptions

| Symbol | Meaning | SI unit | Definition | Domain / assumption | Type |
|---|---|---|---|---|---|
| q | number of selected signatures |1| row count |integer at least2, here4|integer scalar|
| d | signature dimension |1| number of head coordinates |positive integer|integer scalar|
| g_i | raw head-gradient signature |1| final-layer weight and bias gradient in the fixed neural-network coordinates |finite d-vector|real vector|
| epsilon | normalization floor |1|1e-12 in common.py|positive|real scalar|
| v_i | implemented normalized signature |1|g_i/max(norm(g_i),epsilon)|norm at most1 in exact arithmetic|real vector|
| v_bar | mean normalized signature |1|sum_i v_i/q|same coordinates|real vector|
| V | stacked normalized signatures |1|row i is v_i|q by d|real matrix|
| G | mean pairwise novelty |1|average of1-v_i dot v_j over i<j|finite; fixed q|real scalar|
| i,j | selected-signature indices |1|row indices|1 through q|integer scalars|

The gradient signature is formed after a float32 forward pass and uses a norm
floor. Therefore the unit-norm simplification below requires that each raw norm
exceeds that floor. The general identity remains valid even for zero/sub-floor
signatures in exact arithmetic. All quantities are dimensionless in this fixed
network parameterization; none is an SI distance, physical variance or accuracy.

## Complete derivation

By definition,

```math
G=\frac{2}{q(q-1)}\sum_{i<j}(1-v_i^\top v_j)
 =1-\frac{2}{q(q-1)}\sum_{i<j}v_i^\top v_j.
```

Expanding the square of the sum includes all diagonal terms and each unordered
cross-term twice:

```math
\left\|\sum_i v_i\right\|^2
=\sum_i\|v_i\|^2+2\sum_{i<j}v_i^\top v_j.
```

Subtract the diagonal sum, divide by q(q-1), and substitute into the definition:

```math
G=1+\frac{\sum_i\|v_i\|^2-q^2\|\bar v\|^2}{q(q-1)}.
```

When all signatures have norm1, the diagonal sum is q. Substituting and collecting
terms yields

```math
G=\frac{q}{q-1}\left(1-\|\bar v\|^2\right).
```

Thus, at fixed q and unit norms, maximizing mean pairwise novelty is exactly
minimizing the squared norm of the mean normalized signature. It does not specify
how many independent directions span the selected set. This is **not** an identity
for the actual AdamW update: raw gradient norms, all network layers and adaptive
optimizer state are absent from the normalized head-signature score.

## Explicit counterexample

Use q=4,d=2. Set A has rows (1,0),(-1,0),(1,0),(-1,0). Set B has rows
(1,0),(-1,0),(0,1),(0,-1). Every row has norm1 and both row sums are zero.
The proved formula therefore gives G=4/3 for both. The columns of A span a
one-dimensional space, whereas B contains the two independent coordinate axes
and has rank2. Identical novelty can coexist with different span dimensions.
Equivalently, opposition in a single direction and dispersion across two
directions can have the same score.

Unit check: dot products, squared normalized-vector norms and G are dimensionless;
all additions and the q/(q-1) factor are dimensionally consistent.

## Numerical ERROR CHECK

`python experiments/check_novelty_coherence_identity.py` checks the general identity
on1000 synthetic float64 draws, including zero and sub-floor raw gradients.
Maximum discrepancy was4.44e-16; the explicit rank1/rank2 sets both gave4/3.
This is a synthetic algebra check, not a new trained-model replication.

## Relation to existing research

Yin et al., *Gradient Diversity: a Key Ingredient for Scalable Distributed
Learning*, AISTATS2018, PMLR84, arXiv:1706.05699, studies relationships between
concurrent gradient similarity and minibatch learning. Yu et al., *Gradient
Surgery for Multi-Task Learning*, NeurIPS2020, arXiv:2001.06782, studies detrimental
gradient interference and a projection-based remedy. These provide context for
distinguishing diversity, cancellation and conflict, but neither proves the SGO
intervention mechanism. No empirical SGO conclusion is imported from them.
