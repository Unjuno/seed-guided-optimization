# Prospective representation-rank validation

## Purpose

A four-task discovery analysis suggested that gradient novelty helps when the additional optimization-direction diversity is converted into a richer hidden representation. The candidate training-only indicator was frozen as:

> **sign(delta representation effective rank) predicts the sign of held-out mean benefit.**

The rule was then tested prospectively: train models, compute the diagnostic without generating/evaluating the final environment pool, register the prediction in GitHub Issue #12, and only then evaluate the untouched held-out seeds.

## Prospective tests

| test | task | delta rep rank | delta grad rank | pre-registered direction | observed mean benefit | match |
|---|---|---:|---:|---|---:|---|
| photometric | Digits classification | +0.0402 | +0.6174 | positive, small | +0.289 pp accuracy | yes |
| unstructured pixel | Digits classification | +0.0727 | -0.0252 | positive, small/moderate | +0.136 pp accuracy | yes |
| band occlusion | Digits classification | +0.2076 | +0.4527 | positive, larger | +0.715 pp accuracy | yes |
| Wine | real classification | +0.1365 | +2.1473 | positive | +0.336 pp accuracy | yes |
| Iris | real classification | +0.0307 | +1.1776 | positive, very small | +0.113 pp accuracy | yes |
| Diabetes | real regression | **-0.0249** | **+3.6764** | **negative** | **-0.001346 MSE benefit** | yes |

For regression, benefit is `MSE(loss-hard) - MSE(gradnov)`, so a negative value means gradient novelty produced higher MSE.

The directional record is 6/6. This should not be interpreted as six independent Bernoulli trials because three tests share the Digits dataset. The strongest falsification-oriented result is Diabetes: gradient effective rank increased strongly while representation effective rank decreased, and the pre-registered prediction correctly switched to a negative held-out benefit.

Among the first five classification prospective tests, task/generator-level delta representation effective rank correlates with held-out mean accuracy gain (Spearman rho about 0.90; Pearson r about 0.896; n=5). Delta gradient effective rank has essentially no association with the gain in the same five tests. These small-n correlations are descriptive and should be expanded before being treated as a calibrated predictor.

## Tail rule: failed

A secondary post-hoc hypothesis proposed that increasing the across-environment SD of representation effective rank would predict tail/dispersion risk. It partially matched one Digits generator but failed on the independently tested Wine dataset. The tail rule is therefore retired rather than retuned.

The current supported mechanism candidate is deliberately narrow:

> Gradient-space expansion is not itself sufficient. A positive conversion into representation effective rank is a candidate predictor of **mean** held-out benefit.

No reliable training-only predictor of p10/worst-case benefit has yet been established.

## Protocol integrity

For every prospective test:

1. training and selection environment seeds were fixed;
2. final held-out seed pools were separate;
3. paired methods shared initialization, minibatch order, and candidate schedule;
4. model states were saved before held-out evaluation;
5. training-only diagnostics were computed first;
6. the predicted sign was posted to Issue #12;
7. only then was the final held-out pool generated/evaluated.

Prediction/outcome timestamps are preserved in the Issue #12 discussion.
