# Results summary

The repository separates **supported**, **prospective**, **negative/null**, and **open** findings. Headline comparisons are paired, use disjoint held-out environment seeds, and apply Holm correction when five outcome metrics are one family.

## Supported findings

### 1. Structured geometric benchmark
Gradient novelty improves held-out performance over loss-hard selection, and in the controlled MLP test it outperforms physical transformation-parameter novelty by about **+1.45 pp** held-out mean.

### 2. Architecture and optimizer replication
A small CNN replicated mean/minimum gains. Independently tuned AdamW and SGD+momentum both retained positive corrected-significant MLP gains, ruling out a simple AdamW-only explanation.

### 3. RNG compression and learned relevance
Moderate candidate prefiltering can reduce expensive gradient evaluations, while aggressive 16→4 compression damages tail coverage. Training-only gradient information can learn useful RNG relevance in the tested generators; stale cross-generator fingerprints fail, while re-learning/soft weighting recovers performance.

### 4. Relative gradient-redundancy control
Absolute cosine targets can be infeasible across tasks. A within-step normalized target transfers better across the tested Digits/Synthetic conditions. This does not establish a universal operating point or task-level safety guarantee.

### 5. CIFAR-10 / ResNet-20, 40 paired replicates
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

## Prospective representation-rank tests

Frozen rule:

> `sign(delta representation effective rank) -> sign(mean held-out benefit)`

Registered direction tests:

| test | delta rep rank | observed mean benefit | match |
|---|---:|---:|---|
| Digits photometric | +0.0402 | +0.289 pp accuracy | yes |
| Digits unstructured pixel | +0.0727 | +0.136 pp accuracy | yes |
| Digits band occlusion | +0.2076 | +0.715 pp accuracy | yes |
| Wine | +0.1365 | +0.336 pp accuracy | yes |
| Iris | +0.0307 | +0.113 pp accuracy | yes |
| Diabetes regression | **-0.0249** | **-0.001346 MSE benefit** | yes |
| FashionMNIST / Tiny Transformer | **+0.0542** | **+0.736 pp accuracy** | yes |

The directional record is now **7/7 across five datasets**. It is not seven independent datasets because three tests use Digits. The Diabetes negative prediction remains the strongest negative falsification case. FashionMNIST supplies a new positive falsification test that changes dataset, stochastic generator, and representation architecture simultaneously.

### FashionMNIST / Tiny Transformer

This test was preregistered in Issue #20 before the CIFAR/ResNet prospective result was available. Ten paired runs used a 2-layer, 48-dimensional CLS patch Transformer, 3,000 training examples, 1,000 test examples, 64 training environments, and 24 disjoint held-out environments.

Gradient-novel minus loss-hard:

- representation effective rank: **+0.05415**, SE 0.02355, raw paired `p=0.04708`;
- held-out mean accuracy: **+0.7363 pp**, SE 0.2990 pp, raw paired `p=0.03603`.

The frozen practical tolerance was 0.01, so the rank diagnostic registered a positive aggregate prediction. The observed mean effect was positive. **Decision: PASS.**

All 10 paired held-out mean differences were positive. Secondary p10/minimum/clean results are retained descriptively, but no tail-safety rule was registered and no tail claim is promoted.

See [`FASHION_TRANSFORMER_REP_RANK.md`](FASHION_TRANSFORMER_REP_RANK.md).

A secondary rule based on per-environment representation-rank SD failed on Wine and is retired.

## Negative and null results retained

- Worst-only training can damage average/clean performance.
- Pure diversity without hardness can hurt.
- Full-network signatures were slower without tested benefit over the head proxy.
- Single-task selector/meta-policy weights did not transfer universally.
- Absolute gradient-cosine targets fail cross-task.
- Relative redundancy control does not guarantee benefit on a new task.
- Gradient-rank expansion alone does not predict benefit.
- Tail prediction from representation-rank SD failed.
- CPU wall-clock optima are not GPU claims.

## Public claim boundary

The evidence supports:

> Gradient-aware stochastic-environment selection can improve held-out performance in several structured tested regimes; a small mean improvement also survives correction in the fixed CIFAR-10 / ResNet-20 study. Training-only representation effective-rank change is a promising predictor of the sign of **mean** benefit, now including a prospective FashionMNIST/Tiny Transformer architecture-shift PASS.

It does not support a universal seed family, universal selector/controller, universal calibrated gate, general Transformer/large-model validation, confirmed CIFAR tail robustness, or GPU efficiency advantage.
