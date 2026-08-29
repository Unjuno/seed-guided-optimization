# Results summary

The repository separates **supported**, **prospective**, **negative/null**, and **open** findings. Headline comparisons are paired, use disjoint held-out environment seeds, and apply Holm correction when five outcome metrics are one family.

## Supported findings

### 1. Structured geometric benchmark
Gradient novelty improves held-out performance over loss-hard selection, and in the controlled MLP test it outperforms physical transformation-parameter novelty by about **+1.45 pp** held-out mean.

### 2. Architecture and optimizer replication
A small CNN replicated mean/minimum gains. Independently tuned AdamW and SGD+momentum both retained positive corrected-significant MLP gains, ruling out a simple AdamW-only explanation.

### 3. RNG compression and learned relevance
Moderate candidate prefiltering can reduce expensive gradient evaluations, while aggressive 16->4 compression damages tail coverage. Training-only gradient information can learn useful RNG relevance in the tested generators; stale cross-generator fingerprints fail, while re-learning/soft weighting recovers performance.

### 4. Relative gradient-redundancy control
Absolute cosine targets can be infeasible across tasks. A within-step normalized target transfers better across the tested Digits/Synthetic conditions. This does not establish a universal operating point or task-level safety guarantee.

### 5. CIFAR-10 / ResNet-20 primary, 40 paired replicates
Under the unchanged primary protocol, gradient-novel minus loss-hard gives:

- held-out mean **+0.1206 pp**, raw `p=0.002672`, Holm(5) **`p=0.013361`**;
- p10 **+0.1213 pp**, Holm `p=0.052939`;
- minimum **+0.1875 pp**, Holm `p=0.088149`;
- clean +0.0658 pp, Holm `p=0.639984`;
- environment SD -0.0142 pp, Holm `p=0.639984`.

The mean is supported under the stated correction. CIFAR tail robustness is **not confirmed**.

## Mechanism findings

Direct audits rule out simple stories:

- gradient novelty is not simply a better estimator of the mean expected gradient than random sampling;
- loss-hard can have larger immediate one-step loss decrease;
- raw accumulated gradient effective-rank expansion does not separate helpful and harmful regimes.

A four-task trajectory audit found:

| task | held-out mean delta | gradient effective-rank delta | representation effective-rank delta |
|---|---:|---:|---:|
| Digits geometric | +4.394 pp | +0.823 | +0.454 |
| Synthetic | +0.089 pp | +6.231 | +0.113 |
| Breast low | +0.040 pp | +3.362 | -0.165 |
| Breast high | -0.034 pp | +7.148 | -0.361 |

Breast-high is the key counterexample: gradient rank expands the most, yet mean benefit is absent and representation rank falls.

The current working theory is therefore **finite-budget coverage plus representation conversion**, not generic diversity maximization:

```text
hard + non-redundant gradient directions
    -> broader coverage of unresolved task-relevant directions
    -> trajectory change
    -> reusable representation change when conversion succeeds
    -> held-out mean benefit.
```

See [`THEORETICAL_FRAMEWORK.md`](THEORETICAL_FRAMEWORK.md).

## Prospective representation-rank tests

Frozen rule:

> `sign(delta representation effective rank) -> sign(mean held-out benefit)`

Registered condition-level direction tests:

| test | delta rep rank | observed mean benefit | match |
|---|---:|---:|---|
| Digits photometric | +0.0402 | +0.289 pp accuracy | yes |
| Digits unstructured pixel | +0.0727 | +0.136 pp accuracy | yes |
| Digits band occlusion | +0.2076 | +0.715 pp accuracy | yes |
| Wine | +0.1365 | +0.336 pp accuracy | yes |
| Iris | +0.0307 | +0.113 pp accuracy | yes |
| Diabetes regression | **-0.0249** | **-0.001346 MSE benefit** | yes |
| FashionMNIST / Tiny Transformer | **+0.0542** | **+0.736 pp accuracy** | yes |
| CIFAR-10 / ResNet-20 | **+0.03205** | **+0.1703 pp accuracy** | yes |

The registered directional record is now **8/8 across six datasets**. It is not eight independent datasets because three tests use Digits. The primary evidence is the frozen condition-level direction decision, not a fitted magnitude model or per-run classifier.

### FashionMNIST / Tiny Transformer

Initial prospective test, reps 0-9:

- representation effective rank: **+0.05415**, SE 0.02355, raw paired `p=0.04708`;
- held-out mean accuracy: **+0.7363 pp**, SE 0.2990 pp, raw paired `p=0.03603`;
- decision: **PASS**.

Independent preregistered extension, reps 10-29:

- representation effective rank: **+0.07853**, SE 0.02349, descriptive `p=0.003414`;
- held-out mean accuracy: **+0.4377 pp**, SE 0.1918 pp, descriptive `p=0.03420`;
- decision: **REPLICATES** under the frozen condition-average direction rule.

After the extension decision was frozen, all 30 pairs were combined for precision only:

- delta representation rank: **+0.07041**, descriptive `p=0.000354`;
- held-out mean accuracy: **+0.5372 pp**, descriptive `p=0.002367`.

Per-run behavior is not a reliable gate: the independent extension contained 13 sign matches, 5 mismatches, and 2 uncertain cases under the same `0.01` rank tolerance.

See [`FASHION_TRANSFORMER_EXTENSION.md`](FASHION_TRANSFORMER_EXTENSION.md).

### CIFAR-10 / ResNet-20 prospective representation-rank audit

A separate frozen 10-pair audit used reps 40-49 and a fixed 256-example training-only representation probe. Before held-out evaluation, mean delta representation effective rank was **+0.03205**, above the preregistered `0.01` tolerance, sealing a positive prediction.

Held-out mean gradient-novel minus loss-hard was **+0.1703 pp**. **Decision: PASS.**

Descriptive statistics:

- rank delta SE 0.02016, `p=0.1464`;
- held-out mean SE 0.06775 pp, `p=0.03310`;
- p10 +0.2747 pp, descriptive `p=0.0966`;
- minimum +0.1600 pp, descriptive `p=0.3467`;
- clean +0.1800 pp, descriptive `p=0.1006`.

No tail claim is promoted. At the replicate level there were 6 direction matches, 3 mismatches, and 1 uncertain case; again, the supported use is condition-average.

See [`CIFAR_RESNET_REP_RANK.md`](CIFAR_RESNET_REP_RANK.md).

A secondary rule based on per-environment representation-rank SD failed on Wine and remains retired.

## Hosted-CPU execution reproducibility

A preregistered execution audit reran CIFAR reps 45 and 46 twice under single-thread OMP/MKL/OpenBLAS/PyTorch controls. The frozen result was:

- bitwise equality over all scientific fields: **false**;
- max representation-rank drift: **0.427255**;
- max accuracy drift: **0.014667**;
- aggregate rank direction stable: positive in both repeats;
- aggregate held-out mean direction stable: positive in both repeats;
- decision: **DRIFT PERSISTS**.

Rep45 was bitwise identical across AMD EPYC 7763 and AMD EPYC 9V74. Rep46 drifted between AMD EPYC 7763 and Intel Xeon 6973P-C under the same software/thread controls. This makes hardware-dependent numerical paths a strong candidate, but the audit does not isolate a sole cause.

See [`CIFAR_CPU_REPRO_AUDIT.md`](CIFAR_CPU_REPRO_AUDIT.md).

## Negative and null results retained

- Worst-only training can damage average/clean performance.
- Pure diversity without hardness can hurt.
- Full-network signatures were slower without tested benefit over the head proxy.
- Single-task selector/meta-policy weights did not transfer universally.
- Absolute gradient-cosine targets fail cross-task.
- Relative redundancy control does not guarantee benefit on a new task.
- Gradient-rank expansion alone does not predict benefit.
- Tail prediction from representation-rank SD failed.
- The representation-rank rule is not a calibrated per-run gate.
- Hosted-CPU bitwise reproducibility across heterogeneous hardware is not established.
- CPU wall-clock optima are not GPU claims.

## Public claim boundary

The evidence supports:

> Gradient-aware stochastic-environment selection can improve held-out performance in several structured tested regimes; a small mean improvement also survives correction in the fixed CIFAR-10 / ResNet-20 study. Across eight registered prospective conditions on six datasets, training-only representation effective-rank direction has matched the sign of **condition-average mean benefit**, with an independent FashionMNIST/Tiny Transformer replication and a CIFAR/ResNet PASS. The current mechanism hypothesis is finite-budget coverage of unresolved task-relevant gradient directions followed by conversion into reusable representation structure.

It does not support a universal seed family, universal selector/controller, causal representation-rank law, universal calibrated/per-run gate, confirmed CIFAR tail robustness, general large-model validity, bitwise cross-hardware reproducibility, or GPU efficiency advantage.
