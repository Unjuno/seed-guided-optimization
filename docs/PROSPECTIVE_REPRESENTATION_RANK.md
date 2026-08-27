# Prospective representation-rank validation

## Purpose

A four-task discovery analysis suggested that gradient novelty helps when the additional optimization-direction diversity is converted into a richer hidden representation. The candidate training-only indicator was frozen as:

> **sign(delta representation effective rank) predicts the sign of held-out mean benefit.**

The rule was then tested prospectively: train models, compute and seal the diagnostic without evaluating the final environment pool, register the prediction, and only then evaluate untouched held-out seeds.

## Prospective tests

| test | task | delta rep rank | delta grad rank | pre-registered direction | observed mean benefit | match |
|---|---|---:|---:|---|---:|---|
| photometric | Digits classification | +0.0402 | +0.6174 | positive, small | +0.289 pp accuracy | yes |
| unstructured pixel | Digits classification | +0.0727 | -0.0252 | positive, small/moderate | +0.136 pp accuracy | yes |
| band occlusion | Digits classification | +0.2076 | +0.4527 | positive, larger | +0.715 pp accuracy | yes |
| Wine | real classification | +0.1365 | +2.1473 | positive | +0.336 pp accuracy | yes |
| Iris | real classification | +0.0307 | +1.1776 | positive, very small | +0.113 pp accuracy | yes |
| Diabetes | real regression | **-0.0249** | **+3.6764** | **negative** | **-0.001346 MSE benefit** | yes |
| FashionMNIST / Tiny Transformer | image classification | **+0.0542** | not measured | **positive** | **+0.736 pp accuracy** | yes |

For regression, benefit is `MSE(loss-hard) - MSE(gradnov)`, so a negative value means gradient novelty produced higher MSE.

The directional record is now **7/7**. This should not be interpreted as seven independent Bernoulli trials because three tests share the Digits dataset. The seven tests span five datasets. FashionMNIST is the first prospective test here to move the rule to a Transformer-style representation rather than the small MLP family used for the original discovery analysis.

The strongest negative falsification-oriented result remains Diabetes: gradient effective rank increased strongly while representation effective rank decreased, and the pre-registered prediction correctly switched to a negative held-out benefit. FashionMNIST supplies a complementary positive architecture-shift test: mean representation rank increased by about +0.0542 and mean held-out accuracy improved by about +0.736 pp across 10 paired runs.

Among the first five classification prospective tests before FashionMNIST, task/generator-level delta representation effective rank correlated with held-out mean accuracy gain (Spearman rho about 0.90; Pearson r about 0.896; n=5). Those correlations were descriptive and should not be retroactively recalibrated using the new Fashion result. The primary prospective evidence is the frozen directional decision, not a fitted magnitude model.

## FashionMNIST / Tiny Transformer detail

Issue #20 registered the protocol before the CIFAR/ResNet prospective result was known. The experiment changed dataset, stochastic generator, and architecture simultaneously. Across 10 paired replicates:

- mean delta representation effective rank: **+0.05415** (SE 0.02355; raw paired p=0.04708);
- mean held-out accuracy benefit: **+0.7363 pp** (SE 0.2990 pp; raw paired p=0.03603);
- frozen rank tolerance: 0.01;
- registered aggregate prediction: positive;
- observed aggregate mean direction: positive;
- decision: **PASS**.

All 10 paired held-out mean differences were positive. Tail metrics are secondary; no tail rule was registered for this test.

See [`FASHION_TRANSFORMER_REP_RANK.md`](FASHION_TRANSFORMER_REP_RANK.md) for protocol and result details.

## Tail rule: failed

A secondary post-hoc hypothesis proposed that increasing the across-environment SD of representation effective rank would predict tail/dispersion risk. It partially matched one Digits generator but failed on the independently tested Wine dataset. The tail rule is therefore retired rather than retuned.

The current supported mechanism candidate is deliberately narrow:

> Gradient-space expansion is not itself sufficient. A positive conversion into representation effective rank is a candidate predictor of **mean** held-out benefit.

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

Prediction/outcome provenance for the original six tests is preserved in Issue #12. FashionMNIST preregistration and outcome provenance are preserved in Issue #20 and PR #21.
