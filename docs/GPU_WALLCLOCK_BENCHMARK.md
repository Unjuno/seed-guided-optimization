# GPU wall-clock benchmark harness

Issue #2 cannot be answered from CPU GitHub-hosted runners. The repository now contains a fail-closed CUDA harness and a `workflow_dispatch` workflow targeting a self-hosted runner labelled `[self-hosted, linux, x64, gpu]`.

## What is measured

For candidate/environment counts `m={2,4,8,16}` and both `loss_hard` and gradient-aware selection, the harness records two regimes:

1. **fixed updates** — default 80 optimizer updates;
2. **fixed wall clock** — default 30 synchronized seconds per condition.

Each update explicitly calls `torch.cuda.synchronize()` before the wall-clock boundary is evaluated. This is conservative but prevents asynchronous kernel queuing from masquerading as throughput.

The raw CSV records:
- wall seconds and optimizer updates;
- updates/s and candidate-images/s;
- median/p10/p90 GPU utilization sampled from `nvidia-smi`;
- median NVIDIA-reported memory, power and SM clock samples;
- PyTorch peak allocated/reserved GPU memory;
- selected pairwise gradient novelty, candidate CE and selected CE;
- fresh heldout mean/SD/p10/min accuracy and clean accuracy.

The workflow also archives device name/UUID, driver, CUDA/PyTorch versions, total memory, compute capability, static `nvidia-smi -q`, `pip freeze`, and OS metadata.

## Important implementation boundary

The 64 training environments and 80 heldout environments are deterministically generated before timing and transferred to GPU. The timed region therefore measures **candidate batch gather + model forward + gradient-signature/selection + optimizer update**, not CPU image-warp generation or host-to-device environment construction. This boundary must be stated with any result.

The benchmark intentionally refuses CPU fallback. No GPU optimum or equal-wall-clock claim is valid until a real CUDA artifact is produced on identified hardware.

## Invocation

Use the Actions workflow **GPU wall-clock benchmark (self-hosted)** after attaching an NVIDIA CUDA runner with the required labels. The workflow inputs control repetitions, wall-clock seconds, fixed update count and candidate counts. Preserve the raw artifact before interpreting an optimum.

## Decision rule for Issue #2

A hardware-specific optimum is supported if repeated measurements identify a stable Pareto point in heldout performance versus actual synchronized wall time/throughput. If increasing candidate count removes the statistical benefit under equal wall clock, the GPU-overhead hypothesis fails on that hardware. Results must not be generalized across accelerator models without replication.
