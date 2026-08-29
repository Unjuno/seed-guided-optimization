# Prospective representation-rank validation

## Purpose

A four-task discovery analysis suggested that gradient novelty helps when additional optimization-direction diversity is converted into a richer hidden representation. The candidate training-only indicator was frozen as:

> **sign(delta representation effective rank) predicts the sign of condition-average held-out mean benefit.**

The rule is tested prospectively: train models, compute and seal the diagnostic without evaluating the final environment pool, register the prediction, and only then evaluate untouched held-out seeds.

The wording is now explicitly **condition-average**. FashionMNIST and CIFAR diagnostics show that the same threshold is not a reliable per-replicate gate.

## Registered prospective tests

| test | task | delta rep rank | delta grad rank | pre-registered direction | observed mean benefit | match |
|---|---|---:|---:|---|---:|---|
| photometric | Digits classification | +0.0402 | +0.6174 | positive, small | +0.289 pp accuracy | yes |
| unstructured pixel | Digits classification | +0.0727 | -0.0252 | positive, small/moderate | +0.136 pp accuracy | yes |
| band occlusion | Digits classification | +0.2076 | +0.4527 | positive, larger | +0.715 pp accuracy | yes |
| Wine | real classification | +0.1365 | +2.1473 | positive | +0.336 pp accuracy | yes |
| Iris | real classification | +0.0307 | +1.1776 | positive, very small | +0.113 pp accuracy | yes |
| Diabetes | real regression | **-0.0249** | **+3.6764** | **negative** | **-0.001346 MSE benefit** | yes |
| FashionMNIST / Tiny Transformer | image classification | **+0.0542** | not measured | **positive** | **+0.736 pp accuracy** | yes |
| CIFAR-10 / ResNet-20 | image classification | **+0.03205** | not primary | **positive** | **+0.1703 pp accuracy** | yes |

For regression, benefit is `MSE(loss-hard) - MSE(gradnov)`, so a negative value means gradient novelty produced higher MSE.

The registered condition-level directional record is now **8/8 across six datasets**. Three tests share the Digits dataset, so this should not be interpreted as eight independent datasets or eight independent Bernoulli trials.

The strongest negative falsification case remains Diabetes: gradient effective rank increased strongly while representation effective rank decreased, and the frozen prediction correctly switched to negative held-out benefit.

FashionMNIST and CIFAR add materially different representation architectures beyond the original small-MLP discovery family.

## FashionMNIST / Tiny Transformer: initial test and independent replication

Issue #20 registered the initial protocol before the CIFAR/ResNet prospective result was known. Across the original 10 paired replicates:

- mean delta representation effective rank: **+0.05415** (SE 0.02355; raw paired `p=0.04708`);
- mean held-out accuracy benefit: **+0.7363 pp** (SE 0.2990 pp; raw paired `p=0.03603`);
- frozen rank tolerance: 0.01;
- registered aggregate prediction: positive;
- observed aggregate mean direction: positive;
- decision: **PASS**.

Issue #26 then preregistered an exact-protocol independent extension using reps 10-29. Across those 20 new pairs:

- mean delta representation effective rank: **+0.07853** (SE 0.02349; descriptive `p=0.003414`);
- mean held-out accuracy benefit: **+0.4377 pp** (SE 0.1918 pp; descriptive `p=0.03420`);
- frozen aggregate prediction: positive;
- observed aggregate mean direction: positive;
- decision: **REPLICATES**.

After the extension-only decision was frozen, reps 0-29 were combined for precision only:

- rank delta **+0.07041**;
- held-out mean benefit **+0.5372 pp**.

Per-replicate diagnostics do not justify a gate: among the 20 extension pairs, using the same `+/-0.01` rank tolerance descriptively gave 13 matches, 5 mismatches, and 2 uncertain cases.

See [`FASHION_TRANSFORMER_REP_RANK.md`](FASHION_TRANSFORMER_REP_RANK.md) and [`FASHION_TRANSFORMER_EXTENSION.md`](FASHION_TRANSFORMER_EXTENSION.md).

## CIFAR-10 / ResNet-20 prospective falsification

Issue #12 fixed the independent CIFAR audit before the held-out result:

- replicate IDs 40-49, n=10;
- exact separated CIFAR primary protocol;
- final 64-D pooled hidden feature;
- fixed 256-example training-only probe;
- fixed training-environment audit indices;
- both methods trained and both training-only ranks emitted before held-out construction/evaluation;
- `|mean delta rank| < 0.01` -> uncertain;
- positive delta -> predict positive mean benefit;
- nonpositive delta -> predict nonpositive mean benefit.

Observed training-only mean delta representation effective rank was **+0.0320505**, which sealed a **positive** prediction. Only after that seal was the held-out pool evaluated.

Observed held-out mean accuracy delta was **+0.001703125 = +0.1703 pp**. **Decision: PASS.**

Descriptive uncertainty:

- rank delta SE 0.02016, raw paired `p=0.1464`;
- held-out mean SE 0.06775 pp, raw paired `p=0.03310`;
- approximate 95% interval for rank delta: about `-0.0136` to `+0.0777`;
- approximate 95% interval for mean benefit: about `+0.0171` to `+0.3236 pp`.

The rank difference itself is noisy and not individually significant at conventional levels. The prospective evidence is therefore the **predeclared directional decision**, not a claim that rank magnitude is precisely estimated.

Secondary p10/minimum/clean outcomes do not establish a tail rule. At the individual-replicate level, the same `0.01` tolerance gave 6 matches, 3 mismatches, and 1 uncertain case.

See [`CIFAR_RESNET_REP_RANK.md`](CIFAR_RESNET_REP_RANK.md).

## What the current record supports

The narrow supported predictor statement is:

> In the tested conditions, the sign of the **condition-average** training-only representation effective-rank difference has prospectively tracked the sign of mean held-out novelty benefit.

It does not yet support:

- a universal magnitude mapping;
- a calibrated decision probability;
- a reliable per-run gate;
- a causal interpretation of effective rank itself;
- a tail-safety rule;
- large-model/general-Transformer validity.

## Tail rule: failed

A secondary post-hoc hypothesis proposed that increasing the across-environment SD of representation effective rank would predict tail/dispersion risk. It partially matched one Digits generator but failed on the independently tested Wine dataset. The tail rule is therefore retired rather than retuned.

The current mechanism candidate is deliberately narrow:

> Gradient-space expansion is not itself sufficient. Benefit appears when useful optimization-direction coverage is converted into reusable representation structure, for which representation effective-rank direction is a training-only proxy.

No reliable training-only predictor of p10/worst-case benefit has yet been established.

## Protocol integrity

For the prospective sequence:

1. training and selection environment seeds are fixed;
2. final held-out seed pools are separate;
3. paired methods share initialization, minibatch order, and candidate schedule;
4. training-only diagnostics and model states are sealed before held-out evaluation;
5. the direction rule/tolerance is fixed before the final comparison;
6. held-out results are reported regardless of direction;
7. failed secondary rules are retired rather than retuned on the same evidence.

Prediction/outcome provenance is preserved in Issues #12, #20, and #26 and the corresponding result PRs.
