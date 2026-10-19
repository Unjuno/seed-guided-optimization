# Research roadmap: evidence, not desired positive findings

Updated2026-10-19. The owner requested autonomous experiment progress, including informative unexpected findings. [AUTONOMOUS_RESEARCH_SCOPE](AUTONOMOUS_RESEARCH_SCOPE.md) defines operational bounds. X announcement remains paused.

A completed experiment is not necessarily a resolved mechanism. Preserve its registered decision, interval and alternatives. Do not alter thresholds, metrics, repetitions or pools to turn a valid negative result into PASS.

| Question / experiment | Recorded status | Evidence and remaining limitation |
|---|---|---|
| Centered residual spectral rank at matched novelty, Issue100 | Completed; primary support absent | Successful manipulation; +0.2072pp CI[-0.6916,+1.1060]. Not proof of zero effect. |
| Opposition separated from mean novelty/span, Issue103 | Completed; calibration failure | Largest frozen caliper yielded87.25%support and opposition gap0.07356 below required0.15; no performance test. |
| Novelty matched on all7physical-coordinate means/variances, Issue105 | Completed; severe overlap failure | At most64/800supported steps; natural candidate-family identification remains unresolved. |
| Rank/hardness relaxation, Issues104/115 | Initial support; fresh primary non-replication | Initial+0.9066pp; fresh60-block mean+0.2980pp CI[-0.2425,+0.8384]. |
| Rank versus actual selected hardness, Issues112/113 | Completed; calibration failures | Required independent manipulation could not be produced with this family. Preserve joint interpretation. |
| Mean improvement with minimum deterioration, Issue115 | Completed; NO MEAN-TAIL TRADEOFF REPLICATION | Neither registered performance endpoint passed, despite matching/manipulation. Clean/SD findings remain secondary. |
| Correct online gradient correspondence beyond simple controls, Issue121 | Completed; ONLINE GRADIENT-INFORMATION CONTROL SUPPORT | Gradnov versus loss-hard+2.6717pp, versus shuffled-gradients+2.0703pp, versus anchored-random+0.6770pp; all Holm3 one-sided criteria pass. Anchored-random two-sided interval crosses zero. Not a unique-mediator result. |
| Online clean-versus-tail policy tradeoff, Issue124 | **Completed; preregistered two-part SUPPORT** | First complete run36202615155: six fresh pools, gradnov versus anchored-random minimum+4.3016pp and clean−7.5052pp; both directional conditions pass. Finite 80-environment minimum only; [frozen result](ROADMAP_VALIDATION_20260926_RESULT.md). |
| Online loss/rank-matched policy specificity | Open; design required | Existing online policies differ in loss/rank. Require an identifying control/support audit before unique mechanism claims. |
| Native Fashion-MNIST (Issue125), limited cross-dataset transfer | **Completed; two of three directional contrasts PASS** | On a new image dataset (988 train/809 test), anchor−loss-hard mean+1.2008pp and gradnov−anchor minimum+3.3725pp pass Holm3; gradnov−anchor mean−0.0186pp does not. Broader full-dataset/model transfer still open. |
| Objective × estimator × optimizer-configuration validation, Issue125 | **Completed; mixed evidence** | Full16 estimator contrasts PASS in AdamW and SGD configuration, but unequal contributor count forbids cost inference. AdamW interaction PASS after Holm7; SGD interaction and configuration difference unconfirmed. [Frozen result](ROADMAP_VALIDATION_20260926_RESULT.md). |
| GPU matched-cost value, Issue2 | Measurement resource-dependent | CPU evidence and a runnable harness are not GPU evidence. Use only actual authorized hardware; no automatic paid provisioning. |

Small-sample interpretation: see the [post-hoc exact six-pool sign-flip sensitivity](SIX_POOL_SIGNFLIP_SENSITIVITY_20261019.md); the original t5/Holm successes were not re-scored. Future high-stakes Fashion and optimizer claims need a justified sample-size plan with more independent environment pools, rather than expanding the existing registered sample after seeing its outcome.

## Current next execution

The existing Issue124/Issue125 finite queue has been **executed and independently audited**. PR #126 archives the implementation and pooled results; [six-pool scientific outcomes](ROADMAP_VALIDATION_20260926_RESULT.md) distinguish scientific support/non-support from technical job completion. Do **not** rerun valid experiments or change frozen gates in response to mixed outcomes.

Next identification problem: disentangle **correctly associated gradient information** from online selected-hardness/rank and physical-factor allocation using a *new preregistered* intervention with overlap feasibility. It must preserve a baseline and strong random controls, use independent heldout pools, and prevent performance inspection before the selection policy is sealed. Existing natural-subset matching failures warn that naïvely adding calipers is unlikely to work. No fresh confirmatory CPU run was launched as part of the October 19 artifact audit.

The GPU matched-cost branch (Issue #2) remains hardware-blocked until authorized CUDA compute is available; never fill that gap with CPU elapsed-time extrapolation. X publication remains paused.

## Completion criteria

Each finite experiment requires a committed protocol, synthetic checks, complete execution or documented failure, outcome evidence, independent numerical checks and an explicit scope/limitations report. An interval crossing scientifically relevant alternatives remains inconclusive; closing its Issue does not silently solve the mechanism.

The mechanism roadmap requires distinguishing named alternatives with adequate support and fresh confirmation, and connecting the explanation back to the original online policy. Issue121 provides a policy-specificity result against the registered sham and simple controls; it does not alone establish how all intermediate properties cause learning improvements.

## Continuation and stop rule

First finish/audit any active registered experiment. Old data are for labeled exploration; new confirmatory claims use prospectively fixed new blocks/pools. Submit at most one bounded CPU experiment per continuation cycle. When remaining tasks require external hardware/data/owner policy choices or no identifying design is specified, report and pause that branch rather than repeat valid failures, weaken thresholds or claim success. A finite queue can be completed while the broader scientific problem remains open.
