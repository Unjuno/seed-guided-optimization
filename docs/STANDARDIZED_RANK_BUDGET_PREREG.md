# Scale-invariant mediator Q-scaling preregistration

The authoritative preregistration is GitHub Issue #38. This file mirrors the frozen scientific contract inside the repository so the implementation and later result record are auditable together.

## Question

Does channel-standardized hidden-representation effective rank attenuate with the finite-budget gradient-novelty benefit in a fresh Q-scaling replication?

## Frozen design

- Digits geometric MLP, `64 -> 128 ReLU -> 10`.
- Training environments: seeds `13000-13063`.
- Fresh held-out environments: seeds `17000-17079`.
- AdamW lr `1e-2`, weight decay `1e-3`, 10 epochs, batch 128.
- Candidate count `K=16`.
- Update counts `Q={2,4,8,12,16}`.
- loss-hard vs gradnov, novelty weight `0.6`.
- paired reps `300-329`, n=30.
- fixed training-only 192-example probe across train-environment indices `[1,9,17,25,33,41,49,57]`.
- all training-only diagnostics are sealed before held-out construction.

## Primary invariant statistic

Concatenate hidden features over the fixed training-only probe/audit environments. For each channel, subtract its mean and divide by population SD; drop channels with SD <= `1e-12`. Compute the repository effective-rank statistic from the standardized matrix.

Raw effective rank is secondary only.

## Frozen contrasts

For held-out mean benefit `B = gradnov - loss_hard`:

`A_B = mean(B_Q2, B_Q4) - mean(B_Q12, B_Q16)`.

Coverage replication passes iff mean `A_B > 0` and one-sided paired-across-replicates t-test `p < 0.05`.

For standardized-rank delta `Z = gradnov - loss_hard`:

`A_Z = mean(Z_Q2, Z_Q4) - mean(Z_Q12, Z_Q16)`.

Standardized-mediator passes iff mean `A_Z > 0` and one-sided paired-across-replicates t-test `p < 0.05`.

At `Q=16`, all non-timing scientific fields that should be method-identical must match exactly for all 30 pairs.

## Frozen decision

- `SCALE-INVARIANT MEDIATOR CANDIDATE PASS`: identity + coverage replication + standardized-mediator pass.
- `COVERAGE REPLICATION ONLY`: identity + coverage replication only.
- `COVERAGE THEORY FAIL`: identity passes but benefit attenuation fails.
- `INVALID / REPRO FAILURE`: Q=16 identity fails.

Q=8, tails, raw-rank attenuation, per-Q significance, slopes and attenuation correlations are descriptive only and cannot upgrade the decision.
