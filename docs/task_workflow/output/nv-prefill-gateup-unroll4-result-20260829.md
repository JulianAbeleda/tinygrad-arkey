# NVIDIA prefill gate/up unroll-4 result (2026-08-29)

## Decision

**PASS / retain as a research route; measured combined whole-lifecycle win.**
The compiler-generated gate/up K64 reduction loop was compiled with `#pragma unroll 4`.
The switch remains default-off. No claim is booked from the earlier gate/up-only bracket alone.

## Matched test

Both runs used the same current tree, model, 180-row census, gate/up + K + Q/O routes,
multi-queue placement, cut policy, 3 warmups, 9 timed rounds, and 20-cycle deep replay.
The only changed variable was `NV_COMPILER_Q4_IMMA_UNROLL=4`.

| run | status | min ms | median ms | token | logits |
|---|---:|---:|---:|---:|---|
| matched base (unroll disabled) | PASS | 73.958260 | 74.147875 | 198 | exact |
| unroll 4 | PASS | 69.205220 | 69.342578 | 198 | exact |

Measured improvement: **4.753040 ms minimum (6.43%)** and **4.805297 ms median
(6.48%)**. The corresponding measured throughput is 6,922.824 versus 7,398.286
tok/s for this pp512 lifecycle.

## Correctness and census gates

Both JSON results report PASS with 180 compiler mains and 180 Q8 producers:
72 gate/up, 36 K, and 72 Q/O; zero old fixups, copies, partial workspace, or
admitted FP16 overlays. The unroll-4 run passed all 20 deep replay cycles:
records, stage outputs, K/V, logits, and token were exact. The saved token/logit
NPZ comparison against the matched base is bit-exact (`max_abs=0.0`).

## Evidence

- Base: `docs/task_workflow/evidence/nv-prefill-gateup-unroll-20260829/combined-base-deep20-r9-matched.json`
- Candidate: `docs/task_workflow/evidence/nv-prefill-gateup-unroll-20260829/combined-unroll4-deep20-r9-v2.json`
- Candidate logits: `/tmp/combined-unroll4-v2-logits.npz`
- Matched-base logits: `/tmp/combined-base-matched-logits.npz`

The result is booked only against the matched baseline above. The remaining
Q4K prefill lanes (Q4-V live qualification, Q4-down producer/compiler chain,
and the flash graph live hook) remain separate gates.

## Why this is not compared to the older 69.378 ms authority

`candidate-safe-cut-v2-deep20-r9.json` (the older 69.378154 ms authority) has
the same environment names, cut-policy path, 180-row census, route identities,
pp512/deep20/R9 shape, and exact replay gates. However, its compiled program
inventory is not the current inventory: it contains the older `E_1024...`,
`E_1187...`, and `E_512...` variants, while the current no-unroll run contains
different `E_2048...` and `E_4096...` variants. Those hashes encode changed
compiler-generated kernels in this dirty working tree (including the later K
and lifecycle work); they are not an arm-semantics difference—the old and new
JSONs both use `arm: candidate`, and the current base/candidate share the
same current route identities and census.

Therefore 69.378 ms is a historical authority for an earlier compiled
program set, not a valid current-tree no-unroll control. The 73.958260 ms
current baseline should be treated as a protocol-matched current control, and
the 4.753040 ms unroll improvement is booked only against it. The artifact
comparison alone cannot isolate which intervening compiler/lifecycle change
accounts for the ~4.58 ms shift; doing so would require a separate bisect,
not relabeling this result as a regression.
