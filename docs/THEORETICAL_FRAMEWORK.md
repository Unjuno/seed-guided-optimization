# Theoretical framework: subset allocation, overlap and unresolved mediation

Updated2026-09-09. This is a working theory, not a generalization theorem.

## 1. Distinguish procedural effects from mediation

Seeds index environments; no intrinsic good seed classes are assumed. Controlled comparisons support some changes of environment-selection policy. They do not automatically identify gradient novelty as the unique cause or identify a downstream learned-function mediator.

The five budget studies and fixed-Q schedule interventions answer different questions. At full candidate adoption, identical initial/optimizer/RNG states, ordered inputs and deterministic operations imply identical updates. Induction over steps preserves identity provided selector computations have no unequal side effects. This is an implementation control, not an explanation of smaller-subset benefits or a guarantee across hardware.

## 2. Evidence progression

| Study | What is controlled | What is observed | What remains unresolved |
|---|---|---|---|
| Five Q-scaling blocks | paired schedules/initialization; varying adopted count | positive registered low-minus-high contrasts with varying strength | causal role of geometry; structural zero endpoint |
| Issue80 | fixed count and reference hardness strata | high-low+4.669 pp | total physical diversity also shifts |
| Issue83 | reference hardness and scalar total diversity | high-low+3.969922 pp | individual factor allocation shifts |
| Issue86 | same constraints, gradient versus translation scoring | positive contrasts from both scoring procedures | translation-only scoring also changes gradient geometry |
| Issue89 | additionally match translation in every step | strict feasibility FAIL; no heldout test | lack of matched support in some states |
| Issue91 | three-summary matching on supported steps, identical fallback otherwise | matched+2.040585 pp and control+3.070818 pp; conjunction PASS | approximate balance, other factors, reduced manipulation and downstream mediation |

These are offline shared-reference schedule interventions; their high-low differences are not the original online gradnov-over-loss-hard effect.

## 3. Available support changes the question

A comparison requiring every reference state to contain a suitable matched pair may be infeasible. Issue89's worst pilot step had14hardness/diversity-eligible pairs, yet minimum translation difference.579288 exceeded the registered maximum.50. The failure is not evidence of zero performance benefit.

Issue91 retained a fixed.20translation caliper and used identical reference top4 subsets in both matched arms at unsupported steps. There were111fallback steps among2400, with all updates and adopted counts retained. The causal procedural target is the complete policy that intervenes only on available matched support. It is not a claim about all reference states or the unexecuted strict Issue89 design.

The original calibration FAIL remains recorded. Changing the policy required a new preregistration and fresh repetitions/environment seeds; it was not an outcome-dependent rescue of a failed performance test.

## 4. What the new positive result permits

The matched high-minus-low effect is+2.040585 pp,95%CI[+.861808,+3.219362],p.000685123. The control matched only in hardness/total diversity gives+3.070818 pp,CI[+2.040397,+4.101239],p6.13693e-7. Both pass the registered conjunction and manipulation/equivalence gates.

Translation-score separation is reduced from.760933 to.021821; gradient-score separation is also reduced from.206506 to.100945. The translation equivalence90%CI[.017593,.026049] is inside±.20 but excludes zero. Equivalence within a tolerance is not exact equality. Rotation variance remains+.202555 and individual dx/dy variances redistribute even when their sum is balanced.

Therefore the effect survives three scalar constraints over supported steps, but a unique-gradient cause still has not been isolated. Residual translation sensitivity, rotation/other factor changes and the relationship between reference and arm-state geometry remain possible explanations.

The control-minus-matched performance difference has95%CI[-.457078,+2.517543]pp. Even a precise positive difference would not be a mediated fraction: the intervention's support and gradient-dose magnitude change too.

## 5. Mean pairwise novelty is not gradient-span dimension

[Full derivation, variable table and counterexample](NOVELTY_COHERENCE_IDENTITY.md) establish an exact algebraic limitation. With unit-normalized head signatures and fixed subset size, their average pairwise1-cosine is a function of the squared norm of the mean normalized signature. Opposing vectors along one axis and a balanced set spanning two axes can have the same score.

This clarification distinguishes directional dispersion/cancellation from independent span. It does not prove cancellation is why performance improves, describe full-network AdamW updates, or negate observed benefits. Zero/sub-floor gradients require the general norm-floor identity, also derived in that note. The averaged intervention score is not identical to the original greedy minimum-distance-plus-hardness rule.

Prior studies of gradient diversity (Yin et al.,AISTATS2018,arXiv:1706.05699) and gradient interference (Yu et al.,NeurIPS2020,arXiv:2001.06782) motivate distinguishing these objects; they do not establish the SGO mechanism.

## 6. Candidate explanation and retained falsifications

A candidate theory is useful allocation of limited updates across model-conditioned error directions and reusable variation factors. Gradient signatures can guide that allocation without explicit physical-factor names, but a physical score can sometimes approximate part of it. Factor allocation and gradient geometry need not be rival, mutually exclusive causal descriptions.

Raw hidden effective rank is coordinate-dependent under function-preserving reparameterizations; standardized-rank budget mediation failed. These failures do not imply all representation effects are irrelevant. They reject those particular strong scalar-mediator interpretations.

Neither best one-step descent, random mean-gradient estimation, pure diversity nor accumulated gradient rank alone explains all recorded comparisons. MLP shift-specificity did not reproduce in SmallCNN; universal monotone budget/shift laws are unsupported. Tail safety, generality across datasets of the matched interventions and practical matched-cost value remain open.

## 7. Next falsifiable distinction

Measure and manipulate normalized directional coherence separately from raw update magnitude, independent span and task-relevant alignment, using a fixed model/optimizer and matched comparison budget. A positive scalar-novelty effect is insufficient to choose among them. Any new mediator claim needs a prospective intervention and cannot follow merely from correlation with these successful schedules.

## ERROR CHECK and scope

All reported empirical differences are linked to [the Issue91 result](OVERLAP_TRANSLATION_RESULT.md) or retained historical documents. Score algebra is an exact mathematical statement under its explicit assumptions; performance conclusions remain conditional empirical results. Steps and environments are not independent training replicates. Pinned software does not establish cross-hardware bitwise equality, and no total uncertainty or universal performance theorem has been supplied.
