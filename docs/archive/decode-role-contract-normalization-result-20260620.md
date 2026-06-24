# Decode Role Contract Normalization

Date: 2026-06-20

## Verdict

`PASS_DECODE_ROLE_CONTRACT_NORMALIZATION_NATIVE_BLOCKED`

This pass normalizes decode into a single role-contract table. It does not change routing, build kernels, launch
models, or make a new performance claim. The goal is to stop mixing four different decode paths:

- promoted default tinygrad decode;
- imported/source-contract Q4 MMVQ;
- hardened opt-in q8 FFN artifact;
- blocked native q8/MMVQ renderer work.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_role_contract_normalization.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_role_contract_normalization_result.json
```

## Normalized Rows

| row | status | decision |
|---|---|---|
| default decode stack | promoted default, W==D authority | keep as baseline |
| imported Q4 `attn_output` | q8 producer + imported consumer lifecycle passes | continue graph-safe source-import track |
| imported Q4 `ffn_gate` | shape matrix correct, standalone consumer fast | candidate for graph-safe Q4 route |
| imported Q4 `ffn_up` | shape matrix correct, standalone consumer fast | candidate for graph-safe Q4 route |
| Q6 selected roles | current tinygrad default for selected roles | keep default, add Q6 source/import coverage |
| q8 FFN artifact | hardened opt-in, lossy, external | keep default-off |
| native q8 scheduler/renderer | roadmap-only | blocked; no `>=30us` attributed feature |
| MMVQ project option | source-import or native contract preservation | funded project only |

## Current Decode Decision

The next buildable decode track is **graph-safe Q4 source-import routing**, not native renderer work.

Why:

- Q4 imported consumer correctness and shape coverage are already proven for `attn_output`, `ffn_gate`, and `ffn_up`.
- Q4 still needs graph-safe routing and W==D policy before it can be a real model path.
- Q6 source/import parity remains open and should run as a parallel coverage track.
- Native q8/MMVQ renderer is still blocked because the readiness artifact is `ROADMAP_ONLY` and the largest
  timing-grade feature movement remains below the `30us` start gate.

## Native Renderer Boundary

Native renderer start remains blocked by:

- max timing-grade native feature movement below `30us`;
- q8 FFN artifact is externally owned and default-off;
- Q4 source-import graph route is not promoted;
- Q6 source/import coverage is still open.

So the decode sequence is now:

```text
primitive transfer
-> schedule-object metadata
-> role-contract normalization
-> graph-safe Q4 source-import route
-> Q6 source/import coverage
-> native renderer only if attribution clears or broad backend work is accepted
```
