# Seed-Guided Optimization

**Gradient-aware selection of stochastic training environments under a finite update budget.**

Seeds index stochastic environments; they are not assumed to have intrinsic semantic classes or universal quality. SGO studies whether retaining hard environments while reducing redundancy among model-conditioned gradient signatures improves how a fixed subset-update budget is allocated.

> **Status — 2026-09-08:** experimental research code. Performance gains and finite-budget dependence have been observed across several tested tasks. Two new results narrow the explanation: a fixed-Q shared-reference gradient-novelty contrast remains positive after matching reference hardness and scalar total transformation diversity; a fresh translation-variance-only scoring intervention also creates a positive high-versus-low contrast. Gradient novelty is therefore not established as the unique causal variable. Factor-specific allocation and gradient geometry remain coupled; the downstream mediator is unidentified.

## Latest completed experiments

| Experiment | Result | Interpretation |
|---|---|---|
| Parameter-matched intervention, Issue83,30 pairs | high-minus-low gradient-novelty schedules **+3.9699 pp**,95% CI[+3.0506,+4.8892],one-sided p5.10e-10 | reference hardness and scalar total physical diversity pass equivalence; individual factor composition is NOT matched |
| Spatial-allocation alternative, Issue86,30 fresh repetitions | gradient-family high-low **+3.8520 pp**; translation-only family high-low **+1.7686 pp**; both primary tests and all gates pass | a physical-factor score can create a useful contrast without using gradient novelty to score subsets; this does NOT show a pathway independent of gradient geometry |
| Loss-stratified intervention, Issue80,30 pairs | high-low **+4.669 pp**,95% CI[+3.187,+6.152] | reference hardness matched, but total physical diversity shifted; historical result unchanged |
| CIFAR-10/ResNet-20 budget scaling, Issue76,30 pairs | low-minus-high attenuation **+0.1168 pp**,95% CI[+0.0335,+0.2000],p0.00380 | cross-task support under its frozen contrast; exact all-candidate identity is a structural control |

See [parameter-matched result](docs/PARAMETER_MATCHED_RESULT.md), [spatial alternative](docs/SPATIAL_ALLOCATION_RESULT.md), [research status](docs/RESEARCH_STATUS.md) and [theoretical framework](docs/THEORETICAL_FRAMEWORK.md).

## The new discovery: diversity amount is not factor allocation

In Issue83, matching mean pairwise distance in seven standardized transformation coordinates still left high-novelty schedules with more x/y translation variance and less noise/blur variance. This was an exploratory discovery, not a registered endpoint. Issue86 then prospectively tested a translation-only scoring alternative on fresh repetitions/environment seeds.

The gradient-family contrast independently reproduced at+3.852 pp,95% CI[+2.927,+4.777],one-sided p1.10e-9. The translation-family contrast was+1.769 pp,95% CI[+0.580,+2.957],p0.00247. Both families passed the original reference-hardness and scalar-diversity equivalence gates.

However, translation-only selection ALSO increased reference gradient novelty. The result supports an alternative scoring procedure, not proof that gradients are irrelevant. Nor does it isolate translation from every correlated physical factor.

### Do not confuse the baselines

These fixed schedules are constructed on a separate shared reference trajectory, then replayed after resetting the models. They are not the original online gradnov-versus-loss-hard comparison.

In Issue86 the descriptive high-gradient schedule versus the loss-hard reference was+1.753 pp, whereas high-gradient versus low-gradient was+3.852 pp. The translation-high versus reference estimate was+0.639 pp with95% CI[-0.600,+1.878],so superiority over the reference is not established. Reference comparisons are secondary, not matched for total computation, and can span different training CPUs.

## Earlier evidence retained

| Evidence | Result and scope |
|---|---|
| CIFAR-10/ResNet-20 primary,40 pairs | online gradnov-loss-hard mean+0.1206 pp,Holm(5)p0.01336; tails unconfirmed |
| Digits finite-budget tests | two MLP blocks attenuate+2.034 and+2.086 pp; SmallCNN+1.265 pp |
| FashionMNIST/Tiny Transformer budget test | attenuation+0.314 pp,one-sided p0.0273; two-sided CI crosses zero: borderline |
| CIFAR budget endpoint sensitivity | post-hoc exclusion of forced-zero endpoint leaves+0.0968 pp,95% CI[+0.0032,+0.1904] |
| SmallCNN endpoint sensitivity | analogous post-hoc contrast+0.476 pp,95% CI[-0.537,+1.490]: not established |
| Optimizer and architecture comparisons | gains occur beyond one MLP/AdamW setup in tested regimes; not universal validity |
| Rank interventions | raw rank changes under function-preserving reparameterization; standardized-rank budget mediation also failed |
| Hosted CPU audit | one-thread execution did not remove cross-run CIFAR drift; cross-hardware bitwise reproducibility unestablished |

At full candidate selection both methods use the same complete ordered subset. Exact state identity was verified in all five budget-scaling blocks, but this is not by itself a causal explanation of improvements at smaller subsets. Budget curves are not universally monotone.

## Claim boundary

The evidence supports useful, task-dependent stochastic-environment selection and specific preregistered schedule contrasts. It does not establish a unique gradient-novelty mediator, a universal monotone budget or shift-strength law, intrinsic good seed classes, causal representation-rank laws, reliable per-run gating, general tail safety, large-Transformer validity, cross-hardware bitwise identity, or GPU efficiency.

Equivalence tests concern REFERENCE scalar summaries within explicit margins. They do not guarantee identical losses on the subsequently diverging intervention trajectories, or equal factor-specific means/variances. Fixed image sets are reused; fresh seeds do not constitute a new dataset.

## Reproduction

Accuracy CSV fields are fractions:0.01 equals one percentage point. Per-experiment workflows pin the relevant CPU runtime and archive source, schedules, states and manifests.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/check_parameter_matched_evidence.py
python experiments/check_spatial_allocation_evidence.py
python experiments/check_cifar_budget_paired.py --input-dir results
```

The two new independent checkers reproduce the public paired tests without retraining. Full workflow artifacts also permit schedule/hash/environment-level validation. See [experiment index](experiments/README.md) and [document index](docs/README.md) for the older studies.

## Next discriminating question

Can reference gradient novelty be varied while holding not only hardness and scalar total diversity, but also spatial-factor allocation, within preregistered equivalence margins? That requires an independent training-only feasibility stage before further heldout testing. Issue86 did not perform that isolation.

## Citation and license

Citation metadata: [CITATION.cff](CITATION.cff). License: [Apache-2.0](LICENSE).
