# CIFAR Q-scaling: technical recovery, not a scientific result

Issue #76 / PR #77. Original scientific source: `c3fdba4be6fac9a54e578607978889e876cb10e6`. Original run: `34004715086`.

## Interruption

Five train shards completed (50-59 and 65-79: 25 paired replicates). Shard 60-65, job 101409705441, was cancelled at the configured 180-minute job limit. Its log recorded completed training for reps60-63; it did not publish the complete training CSV/manifest. The downstream evaluation and aggregate jobs were skipped. No heldout outcome is available and no scientific PASS/FAIL is assigned.

## Recovery policy

The pre-evaluation amendment is recorded in Issue76 comment5558704554. Preserve the five completed original artifacts; never retrain or choose among them. Exclude the entire unsealed partial artifact9982838690, rather than reconstructing a retrospective seal. Retry only reps60-64 as five one-replicate jobs. Within each replicate, initialization, pretraining, cached minibatches, every Q and both methods stay together on one runner. The original scientific scripts, all seeds, the four-thread runtime, sample size30 and statistical decision are unchanged.

The first complete valid recovery execution, including exactly the25 original and5 retried replicates, is authoritative. Technical retries are never selected by observed benefit. There is no optional stopping or significance-driven extension.

| Preserved artifact | Replicates | Artifact ID |
|---|---|---:|
| cifar-budget-train-50 | 50-54 | 9982782017 |
| cifar-budget-train-55 | 55-59 | 9982604257 |
| cifar-budget-train-65 | 65-69 | 9982045116 |
| cifar-budget-train-70 | 70-74 | 9982594822 |
| cifar-budget-train-75 | 75-79 | 9982627560 |

## Mandatory integrity barrier

`check_cifar_budget_recovery.py` verifies the exact registered protocol hash, original scientific-source and summarizer hashes, training runtime metadata, complete replicate/Q/method grids, every checkpoint file hash and every canonical model-tensor digest. It rejects the excluded60:65 range, duplicate or missing rows, nonfinite metrics, and Q8 mismatches. Preserved shards must retain their original run identity.

A global all-240-states seal is required before ANY evaluation job constructs heldout inputs. Each evaluation job revalidates its unchanged original manifest against that global seal. Final validation reconstructs mean/SD/p10/min from all7680 environment rows before invoking the unchanged frozen summarizer. Clean metrics have no per-example archive and are not independently reaggregated from individual predictions by this checker.

Artifacts are fetched from the original run using repository-scoped `actions:read`; the workflow has no write permission. CIFAR archive bytes are checked against the installed torchvision CIFAR10 checksum before extraction. Recovered training uses Python3.12.14, torch2.10.0+cpu, torchvision0.25.0+cpu, NumPy2.3.5, pandas2.2.3, SciPy1.17.0 and scikit-learn1.8.0. Four CPU threads and deterministic PyTorch match Issue76. Evaluation runtime/backend metadata are saved separately from the original training metadata.

## Limits

This is operational recovery, not new mechanism evidence. Heterogeneous CPU paths can differ; no cross-hardware bitwise equivalence is claimed. The Q8 identity is an implementation control: both selectors use the same complete candidate set. Even a future frozen low-vs-high PASS would not by itself identify gradient coverage as a causal mediator or establish compute savings.

The original heldout image subset is reused. Replicate/environment seeds, not image identities, are fresh. Report sampling uncertainty conditional on those fixed samples, and distinguish it from hardware and dataset uncertainty. No new outcome or speed benchmark is reported here.
