# CIFAR/ResNet hosted-CPU reproducibility audit

## Purpose

This is an **execution reproducibility audit**, not a new selector-benefit experiment. The scientific CIFAR-10 / ResNet-20 representation-rank prospective result was already sealed separately; this audit tests whether cross-run numerical drift can be removed by forcing common CPU math libraries and PyTorch to one thread.

Preregistration: Issue #28.

## Frozen protocol

- scientific protocol: unchanged CIFAR-10 / ResNet-20 representation-rank audit;
- replicate IDs: 45 and 46;
- independent repeats: A and B for each replicate;
- `OMP_NUM_THREADS=1`;
- `MKL_NUM_THREADS=1`;
- `OPENBLAS_NUM_THREADS=1`;
- PyTorch intra-op threads = 1;
- deterministic algorithms enabled;
- model/data/seed/optimizer/environment/selector/probe settings unchanged;
- `train_seconds` excluded from equality checks;
- representation-rank and accuracy numerical-stability tolerance: `1e-8`.

The first workflow execution failed before scientific computation because the CIFAR download timed out and a fallback URL returned non-archive content. The workflow was repaired operationally by using an MD5-verified CIFAR archive path. The authoritative scientific execution is Actions run `33192491959`; all four scientific jobs and the frozen aggregate completed successfully.

## Preregistered decision

| quantity | result |
|---|---:|
| all non-timing scientific fields bitwise equal | **false** |
| metadata exact under the comparison rule | true |
| max absolute representation-rank drift | **0.4272553899** |
| max absolute accuracy-metric drift | **0.0146666667** (1.4667 pp) |
| numerical-stability tolerance | `1e-8` |
| aggregate rank direction stable | true |
| aggregate held-out mean direction stable | true |
| frozen decision | **DRIFT PERSISTS** |

Single-thread execution therefore did **not** restore practical cross-run numerical reproducibility across the tested hosted runners.

## Directional stability

Directional stability was a preregistered secondary diagnostic and does not upgrade the failed reproducibility decision.

| repeat | mean delta representation rank | mean delta held-out accuracy | rank direction | mean direction |
|---|---:|---:|---|---|
| A | +0.040253 | +0.004078 (+0.4078 pp) | positive | positive |
| B | +0.035434 | +0.003250 (+0.3250 pp) | positive | positive |

Thus the scientific aggregate direction was stable even though exact/numerical values were not.

## Per-replicate localization

### Replicate 45

Repeat A ran on **AMD EPYC 7763** and repeat B on **AMD EPYC 9V74**. Despite different AMD CPU models, every compared non-timing scientific field was bitwise identical.

### Replicate 46

Repeat A ran on **AMD EPYC 7763** and repeat B on **Intel Xeon 6973P-C**. Eighteen compared scientific fields differed.

Examples:

- loss-hard pretrain representation effective rank: `4.1602111` vs `3.8471536`;
- loss-hard representation-rank-from-pretrain: `-0.1767047` vs `+0.2505507`;
- gradnov representation-rank-from-pretrain: `-0.0696754` vs `+0.3479418`;
- loss-hard held-out mean accuracy: `0.40184375` vs `0.403875`;
- loss-hard minimum accuracy: `0.3673333` vs `0.3820000`;
- gradnov minimum accuracy: `0.3746667` vs `0.3890000`.

The software/runtime versions were the same in all four jobs:

- Python 3.12.14;
- NumPy 2.3.5;
- SciPy 1.17.0;
- PyTorch 2.10.0+cu128;
- torchvision 0.25.0+cu128;
- OMP/MKL/OpenBLAS/PyTorch thread count = 1.

## Interpretation

The frozen conclusion is narrow:

> **Thread scheduling alone is not a sufficient explanation for the previously observed hosted-CPU drift.**

The pattern makes CPU/microarchitecture-dependent numerical paths a strong remaining candidate: the two AMD runs for rep45 were bitwise identical, while the AMD-vs-Intel pair for rep46 drifted substantially under the same pinned software and single-thread controls.

This observation does **not** prove that CPU vendor or microarchitecture is the sole cause. Training kernels, low-level math libraries/oneDNN behavior, SVD/eigendecomposition paths, and other platform-dependent floating-point details have not been separately isolated.

## Scientific boundary

- Do not mix replicate rows from independent Actions executions into one scientific analysis.
- The completed CIFAR/ResNet prospective PASS remains the authoritative scientific result for that experiment.
- Hosted-runner bitwise reproducibility is **not** established.
- Aggregate scientific direction was stable in this audit, but that is weaker than numerical reproducibility.

For future exact reproducibility studies, use a pinned hardware/runtime environment (for example a fixed self-hosted CPU image) and record CPU/runtime provenance with every artifact.

## Evidence files

- `results/cifar_cpu_repro_decision.csv`
- `results/cifar_cpu_repro_directions.csv`
- `results/cifar_cpu_repro_comparison.csv`
- `results/cifar_cpu_repro_runtime.csv`
- `experiments/cifar_cpu_repro_entry.py`
- `experiments/summarize_cifar_cpu_repro.py`
- `.github/workflows/cifar_cpu_repro_audit.yml`
