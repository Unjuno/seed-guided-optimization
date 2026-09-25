# Issue124 invalid execution attempts

## Run 36201702060 — invalid before scientific execution

- Head: `b44f16d98ccaeaa38999697d942297ff85767153`
- Outcome: workflow failure in the synthetic pre-run selftest.
- Failure: the intended zero-standard-error fixture used six floating `0.1` values. Floating representation produced a tiny nonzero sample SD, so the exact zero-SE branch was not exercised and the fail-closed selftest stopped.
- Scientific boundary: no training command completed and no heldout environment was constructed; reps6000-6059 therefore produced no scientific outcome.
- Correction allowed: replace only that synthetic fixture with exact zeros.
- Frozen Issue124 repetitions, pools, policies, thresholds, endpoints, statistical unit and historical scientific sources are unchanged.
- The first subsequent complete valid run remains authoritative.
