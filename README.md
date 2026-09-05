# Seed-Guided Optimization

**Gradient-aware selection of stochastic training environments under a finite update budget.**

Random seeds index stochastic environments, which induce different gradient directions. Seed-Guided Optimization (SGO) studies whether a fixed subset-update budget can be allocated more effectively by retaining hard environments while reducing redundancy among model-conditioned gradient directions. Seed integers are not assumed to have intrinsic semantic classes or universal quality.

> **Status — 2026-09-06:** experimental research code. Benefits are supported in some structured small-scale benchmarks and narrowly for CIFAR mean accuracy. The internal causal mediator remains unidentified. The completed dual-transfer experiment returned **NO SHARED REPLICATION**. A new preregistered 30-pair fixed-dose audit returned **DOSE-DEPENDENT BENEFIT PASS**, but its full-minus-clean interaction is borderline: one-sided p=0.048974 and two-sided 95% CI crosses zero. A frozen PASS must not be presented as conclusive mechanism evidence.

## Latest completed experiments

| Experiment | Result | Interpretation |
|---|---|---|
| Dual evaluation, Issue #59, 30 pairs | Shared benefit +0.78165 pp, one-sided p=0.099776; shared-minus-nuisance +0.00674 pp, p=0.425828 | **NO SHARED REPLICATION**; baseline difficulty matched but both evaluation mixtures were near clean |
| Fixed-dose full geometric effect, Issue #61, 30 new pairs | +2.84223 pp; two-sided 95% CI [2.23536, 3.44910] pp; one-sided p=8.68e-11; positive in 28/30 pairs | Supported under this fixed protocol |
| Fixed-dose full-minus-clean interaction | +1.62125 pp; two-sided 95% CI [-0.31784, 3.56035] pp; one-sided p=0.048974; positive in 19/30 pairs | Narrowly passes the preregistered directional rule, not strong causal evidence |

See [dual-transfer result](docs/DUAL_TRANSFER_RESULT.md) and [fixed-dose result, protocol, uncertainty and artifact manifest](docs/FIXED_DOSE_RESPONSE.md). The latter retains checkpoint/source hashes and environment-level evidence. Recompute the primary fixed-dose statistics without retraining:

```bash
python experiments/check_fixed_dose_paired.py --input-dir results
```

Fresh training replicates and environment seeds do **not** imply fresh image identities. These Digits audits reuse a fixed 988-training/445-test image split. Earlier loss-hard-only evaluation calibration used test-image labels and was evaluation-only validation, not training-only calibration.

## Research claim and evidence boundary

The method is **model-conditioned stochastic-environment selection / trajectory shaping**, not seed-number optimization. A hard anchor is selected first, then additional candidates balance standardized loss and novelty of final-layer gradient directions relative to the selected set. The final-layer signature is a cheap proxy, not a full-network gradient.

Under some structured shifts, a limited subset-update budget can favor hard, non-redundant directions over hardness alone. Budget-scaling evidence is consistent with a subset-allocation explanation. The stronger claim that the benefit is mediated by reusable latent factors is still a hypothesis; neither representation rank nor the latest dose audit identifies that mechanism.

## Evidence at a glance

Historical comparison details and protocols remain in [RESULTS](docs/RESULTS.md), [RESEARCH_STATUS](docs/RESEARCH_STATUS.md) and the linked experiment documents. Holm correction is used where five outcome metrics form a tested family.

| Evidence | Result | Scope |
|---|---|---|
| Digits MLP, geometric shifts | Gradient-novel exceeds parameter-novel by +1.45 pp held-out mean | Tested small-scale structured regime |
| Small CNN replication | +2.25 pp mean and +2.57 pp minimum vs loss-hard, both Holm-significant | Tested CNN protocol |
| Optimizer replication | Gains survive independently tuned AdamW and SGD+momentum | Not explained by AdamW alone in the tested MLP |
| RNG candidate compression / learned relevance | Moderate prefiltering helps; aggressive compression loses tail coverage; learned relevance can fail after generator changes | Not a universal seed family |
| Relative redundancy control | Within-step normalization transfers better than an absolute cosine target in tested tasks | Not a task-level safety guarantee |
| CIFAR-10 / ResNet-20 primary, 40 pairs | Mean +0.1206 pp, Holm(5) p=0.01336 | Mean supported; tail robustness unconfirmed |
| Finite-budget Q-scaling, two n=30 blocks | Frozen benefit attenuation +2.034 then +2.086 pp; exact method identity when all candidates contribute | Within the Digits/geometric family; not an identified internal mediator |
| Raw representation-rank prospective record | Registered condition-average direction matches, including CNN and Transformer conditions | Fixed-parameterization marker, not a calibrated per-run gate |
| Function-preserving rank intervention | Raw effective rank changes while predictions remain identical | Raw rank is not functionally intrinsic |
| Standardized-rank budget test | Frozen mediator criterion failed while benefit attenuation replicated | Normalization did not rescue this mediator hypothesis |
| Hosted-CPU reproducibility | One thread did not remove cross-run drift | Bitwise cross-hardware reproducibility is not established |

## External validation: CIFAR-10 / ResNet-20

The primary 40-pair gradient-novel minus loss-hard mean difference was +0.1206 pp, raw p=0.002672 and Holm(5) p=0.01336. The p10 difference was +0.1213 pp with Holm p=0.05294; minimum +0.1875 pp with Holm p=0.08815; clean +0.0658 pp with Holm p=0.63998. **Do not describe these as confirmed CIFAR tail improvements.**

A separately preregistered 10-pair raw-rank audit gave a +0.03205 training-only rank difference and +0.1703 pp held-out mean benefit, matching the frozen directional rule. Rank's descriptive p was 0.1464, so this was not a claim of significant rank expansion. Per-replicate signs were imperfect.

See [primary protocol](docs/CIFAR_RESNET_PRIMARY.md) and [rank audit](docs/CIFAR_RESNET_REP_RANK.md).

## Representation mechanism: marker versus cause

Raw hidden effective rank tracked condition-average benefit in prospective tests. FashionMNIST/Tiny Transformer also replicated in 20 independent training pairs: rank difference +0.07853 and held-out mean +0.4377 pp. Those directional results remain evidence for a marker under fixed parameterizations, not for rank causing accuracy.

A function-preserving intervention changed raw rank without changing predictions. Channel-standardized rank subsequently failed its preregistered budget-coupling test. Larger accumulated gradient rank is also insufficient; loss-hard can achieve a larger immediate one-step loss decrease, and novelty is not simply a superior estimator of the mean gradient.

See [theoretical framework](docs/THEORETICAL_FRAMEWORK.md), [rank intervention](docs/FUNCTION_PRESERVING_RANK_INTERVENTION.md), [standardized-rank result](docs/STANDARDIZED_RANK_BUDGET_RESULT.md), [prospective record](docs/PROSPECTIVE_REPRESENTATION_RANK.md), and [Fashion extension](docs/FASHION_TRANSFORMER_EXTENSION.md).

## Reproducibility and limitations

The CIFAR single-thread CPU audit returned **DRIFT PERSISTS**: maximum rank drift 0.427255 and accuracy-metric drift 0.014667, although aggregate directions remained positive. Hardware-dependent numerical paths are a candidate cause, not an isolated sole cause. Do not combine scientific rows from separate reruns. See [CPU audit](docs/CIFAR_CPU_REPRO_AUDIT.md).

The latest fixed-dose audit explicitly uses PyTorch 2.10.0+cpu, one-thread deterministic execution and paired methods on the same runner. All 60 states are preserved with hashes. Hosted CPU families still differ between shards. Clock snapshots are not controlled benchmark clocks; no GPU or wall-clock efficiency advantage is established.

Do not claim a universal selector, universally good seeds, causal rank law, calibrated per-run gate, confirmed CIFAR tail robustness, general large-Transformer validity, or exact-zero effects from nonsignificant tests. No safety guarantee follows from average accuracy improvement.

## Reproduction and repository map

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/mlp_geometric.py --start 0 --end 20 --output mlp_geometric.csv
```

Use [experiments/README.md](experiments/README.md) for individual workflows and [docs/README.md](docs/README.md) for the evidence archive. Accuracy-like CSVs use fractions: 0.01 is one percentage point. Requirements pin PyTorch 2.10.0, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 and scikit-learn 1.8.0. CIFAR/Fashion workflows additionally specify torchvision. Follow the exact workflow for CPU-wheel and backend conditions rather than assuming identical hardware execution.

## Current research priorities

The immediate statistical target is an independent challenge to the borderline full-minus-clean interaction, without selecting new strengths after outcomes. The mechanistic target is an intervention distinguishing reusable task structure from baseline difficulty and other learned-function changes. Repeated generator-matching searches have not identified this mediator.

Other open work includes an actionable early-trajectory diagnostic, materially different tasks/modalities, a new tail-safety theory, pinned-hardware reproducibility, and GPU comparisons with genuinely matched costs. Full-candidate identity and limited-budget benefits alone do not establish a universal optimization law.

## Citation and license

Citation metadata: [CITATION.cff](CITATION.cff). License: [Apache-2.0](LICENSE).
