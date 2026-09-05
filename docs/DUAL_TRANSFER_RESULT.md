# Dual-evaluation transfer-specificity result (Issue #59)

Authoritative run: 33665025633. Scientific head: 946ea78a9094eab1a0068e3b87bf0808a4c09031.

## Frozen result

**NO SHARED REPLICATION**. This is not a transfer-specificity PASS.

All 30 paired replicates 1100-1129 completed. The loss-hard-only calibration selected lambda=0.010 and alpha=0.005. Confirmatory loss-hard mean accuracy was 0.43097097463905804 on shared evaluation and 0.4288464428236088 on nuisance evaluation. The difference, 0.0021245318154492487 (0.212453 percentage points), passed the fixed 0.02 difficulty tolerance.

| Quantity | Mean (accuracy fraction) | Paired standard error | One-sided p |
|---|---:|---:|---:|
| Shared gradnov minus loss-hard | 0.007816478225092105 | 0.005954155419463779 | 0.09977630221358615 |
| Nuisance gradnov minus loss-hard | 0.007749064254264028 | See replicate CSV | Descriptive only |
| Shared minus nuisance benefit | 0.00006741397082807669 | 0.00035728616297763693 | 0.4258275993086948 |

The shared effect did not meet the preregistered one-sided 0.05 criterion. The specificity contrast did not meet it either. A failed significance criterion is not proof that the population effect is exactly zero.

## Design limitation, not a revised decision

Both selected strengths were the smallest positive grid points. The shared evaluation is an image interpolation containing only 1% of the geometric-transformed image; nuisance contains 0.5% of a permuted image. Matching baseline accuracy near the clean-input limit does not establish a strong factor manipulation. This observation cannot upgrade the frozen null result or justify changing its thresholds.

The result concerns these weak evaluation mixtures; it is not a failure to replicate the earlier full-strength geometric benchmark. The internal mediator and reusable-factor causal explanation remain unidentified.

Calibration is **evaluation-only validation**, not training-only: it uses labels from the fixed test-image split, although not gradnov models. Fresh environment seeds and initialization seeds do not imply fresh image identities. This reuse limits generalization claims.

## Reproduction and evidence

Use experiments/summarize_transfer_dual_specificity.py with sealed lambda 0.010, alpha 0.005 and the six original confirmatory shard artifacts. Original aggregate artifact 9860627130, SHA256 ca874d992f8b3f4d8651ff5b180d1593de996bd97ce747ce5573be25551b12eb.

Runtime was hosted Ubuntu CPU, one PyTorch thread, batch128; Python3.12, NumPy2.3.5, pandas2.2.3, SciPy1.17.0, scikit-learn1.8.0, PyTorch2.10.0. CPU model/clock were not logged by this workflow; no speed claim is made.

Next experiment must use a new preregistration. A fixed-dose functional-response diagnostic, including clean and full-strength geometric endpoints, can test whether the benefit changes with shift strength without further difficulty-matching searches. Such a diagnostic is not a difficulty-matched specificity or causal-mediation test.
