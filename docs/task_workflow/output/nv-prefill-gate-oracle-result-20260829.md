# NV pp512 gate/up critical-path oracle result

## Decision: PASS

The exact 72-call gate/up population is exposed on the current-best composed route's critical path. Investment in a llama-style packed Q4_K x Q8 gate/up kernel is qualified.

## R9 control/oracle/control

| Arm | Median ms | Minimum ms | Structural status | Deep replay |
|---|---:|---:|---|---|
| control 0 | 65.298799 | 65.281587 | PASS | exact |
| gate oracle | 44.830975 | 44.793935 | PASS | exact |
| control 2 | 65.301318 | 65.266894 | PASS | exact |

Median recovery is 20.467824 ms versus control 0 and 20.470343 ms versus control 2. Minimum recovery is 20.487652 ms and 20.472959 ms respectively.

## Structural isolation

- Gate/up compiler mains: 0.
- ABI-compatible gate oracle mains: 72.
- Unchanged K mains: 36.
- Unchanged Q/O mains: 72.
- Unchanged serialized Q4 V mains: 18.
- Unchanged graph-owned Q6 V mains and producers: 18 each.
- Total admitted mains and Q8 producers: 216 each.
- Canonical unique packed weight arguments: 216.
- Weight-copy kernels and old fixups: 0.
- Oracle graph-owned record/output allocations: 72 each.
- Twenty-cycle deep replay: exact.

The oracle writes deterministic zeros and intentionally does not preserve model semantics; token 112236 is not a correctness result. It preserves stable execution and graph structure to measure the recoverable critical-path upper bound.

## Interpretation

The 20.47 ms oracle recovery is the full removable gate/up-main service, not the expected gain from a real kernel. The matched tinygrad-to-llama service delta remains 190.112 us per projection, or 13.688064 ms across 72 projections. That target fits comfortably inside the measured critical-path upper bound and would project the current 65.30 ms route to roughly 51.61 ms if fully realized.

This result qualifies only the gate/up packed matmul body. FFN down, activation reuse, and role epilogues remain separately gated by the input scope.
