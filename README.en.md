# Seed-Guided Optimization (SGO)

**What changes when we select random training transformations using loss and gradient directions?**

A personal hobby research project on machine-learning experiments. The repository includes positive results, failed hypotheses, preregistered decision rules, result CSVs, and verification code.

[日本語](README.md) · [Results and documents](docs/README.md) · [Check without retraining](docs/QUICKSTART.md) · [Research status](docs/RESEARCH_STATUS.md)

## The idea

Suppose there are 16 candidate image transformations, but only 4 can contribute to an update. **Loss-hard** prioritizes transformations with high current loss. **Gradnov** considers difficulty and how similar the candidate head-gradient directions are to directions already selected.

Seeds are reproducible environment identifiers, not integers with intrinsic or universally good properties. Matching the number of selected environments is **not** the same as matching wall-clock time or total computation.

## What has been observed

[Experiments on Digits, FashionMNIST, and CIFAR-10](docs/RESEARCH_STATUS.md) have found conditions where the selection procedure improves mean accuracy. Effects depend on the task and model. The causal explanation is still under investigation.

| Study | Actual comparison | Result and scope |
|---|---|---|
| [CIFAR-10 / ResNet-20, 40 pairs](docs/CIFAR_RESNET_PRIMARY.md) | Online gradnov versus loss-hard | Mean accuracy **+0.1206 percentage points**; a small improvement, not established worst-case robustness. |
| [Digits / SmallCNN, 30 pairs](docs/OVERLAP_TRANSLATION_RESULT.md) | High- versus low-novelty reference-defined schedules, approximately matched for reference loss, total diversity, and translation variance | **+2.041 percentage points**, 95% CI **[+0.862, +3.219]**. This is a different intervention, not the online gradnov/loss-hard benchmark. |
| [Strict translation-matching calibration](docs/TRANSLATION_MATCH_CALIBRATION_RESULT.md) | Can every step meet the matching constraints? | **Failed**; no intervention-arm training or heldout performance evaluation followed that failure. |

Percentage points are absolute accuracy differences: 80% to 81% is +1 percentage point. Each result document specifies the data, model, optimizer, hardware, and statistical procedure. These different contrasts are not a leaderboard.

Recent experiments investigate **which variation factors receive learning opportunities**, not just total diversity. A [translation-only scoring intervention](docs/SPATIAL_ALLOCATION_RESULT.md) produced a positive high/low contrast, but also changed gradient geometry. Neither physical factors nor gradient novelty have been isolated as the unique cause.

## Try a read-only CSV check

No GPU, image dataset, or checkpoint download is needed. These commands recompute published statistics without training a model.

```bash
git clone https://github.com/Unjuno/seed-guided-optimization.git
cd seed-guided-optimization
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-check.txt
python scripts/check_public_results.py
```

On Windows, create the environment with `py -3.12 -m venv .venv` and use `.venv\Scripts\python.exe` for the two Python commands. More checks and the separate full-audit route are in [QUICKSTART](docs/QUICKSTART.md).

`PUBLIC STATISTICS VERIFIED` means that the public paired means, standard errors, intervals, p-values, and counts agree with the committed summary. It does not certify the original model runs, all matching gates, or a new independent replication.

## Find your way around

| Topic | Entry point |
|---|---|
| Supported findings and open questions | [Research status](docs/RESEARCH_STATUS.md), [working theory](docs/THEORETICAL_FRAMEWORK.md) |
| Study reports, including failures | [Document index](docs/README.md) |
| Implementation and workflows | [Experiment index](experiments/README.md) |
| CSVs and comparison definitions | [Evidence index](results/README.md) |
| Questions and reproduction reports | [Issues](https://github.com/Unjuno/seed-guided-optimization/issues), [contributing](CONTRIBUTING.md) |

## Limits

This is **not** a universal optimization law, proof of a unique gradient-novelty mechanism, worst-case guarantee, or GPU speedup claim. Later mechanism experiments reuse the Digits image identities; fresh seeds are not a new dataset. Same-seed numerical identity across CPU/backend changes is not established.

The repository contains experimental scripts, not a finished drop-in optimizer library. Historical positive findings and failed tests are retained rather than rewritten as one successful story.

Code license: [Apache-2.0](LICENSE). Citation metadata: [CITATION.cff](CITATION.cff). External datasets retain their own terms.
