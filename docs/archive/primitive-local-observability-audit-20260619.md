# Primitive-local observability audit - 2026-06-19

Audit target: use `extra/qk_primitive_ledger.py` as the source-of-truth checker for the current primitive frontier,
including the latest TPE-7a rebindable-node artifact.

## Method

Commands:

```bash
PYTHONPATH=. .venv/bin/python extra/qk_primitive_ledger.py --print-summary
.venv/bin/python -m py_compile extra/qk_primitive_ledger.py extra/qk_tensile_rebindable_node.py
git diff --check
```

This is a replay audit. It reads committed artifacts and docs; it does not launch a new hardware search, change a
model route, or change defaults.

## Result

The ledger now contains 11 observations and passes validation. Runner smoke passes for three replay sessions:
TPE-5 shape matrix, TPE-6 runtime boundary, and TPE-7a rebindable node.

| primitive frontier | current audit state | next action |
|---|---|---|
| q8/MMVQ activation lifecycle | DEFERRED | wait for named codegen capability for fused norm -> per-32 q8 side-channel |
| pure tinygrad prefill WMMA | KILL | do not reopen bounded sweep without a new codegen primitive |
| extracted Tensile shape matrix | PASS | eligible only after graph route + artifact policy |
| TPE-6 FFN block transfer | REDIRECT | solve graph integration; naive per-op route loses to host sync |
| TPE-7a rebindable node | PASS | proceed to in-model captured `Ops.PROGRAM` gate, still research-flag only |
| spec decode shortcut | CLOSED | do not reopen as a single-kernel shortcut |
| reuse-free flash-prefill | REFUTED | do not reopen without locality/LDS/register reuse |

## What changed in this audit

The audit found one missing coverage row: `bench/qk-tensile-extraction/rebindable_node.json` existed but was not
ingested by the primitive ledger. The runtime-boundary adapter now records it as `tensile:tpe7a:rebindable_node`.

This matters because TPE-7a is a different claim from TPE-6:

- TPE-6 answered whether the extracted kernel's GPU math speed transfers into an FFN block.
- TPE-7a answers whether one precompiled kernel object can be rebound to current buffers, which is required for graph
  replay and different model layers.

The result is a graph-protocol prerequisite, not a throughput pass.

## Remaining unmapped or under-evidenced areas

- In-model capture is still unmapped. The ledger has no observation for a captured precompiled Tensile `Ops.PROGRAM`
  inside PREFILL_V2 TinyJit.
- End-to-end warm pp512/pp1024 after TPE-7 routing is still unmapped.
- External-artifact policy is still unresolved. The ledger can measure the primitive, but it cannot decide whether
  shipping a ROCm/Tensile artifact is acceptable.
- Level-4 PMU evidence is still absent in this shell (`rocprofv3` and `rocprof-compute` are not on PATH). Current
  bottleneck labels remain Level 1-3 claims unless backed by static metadata or traces.
- Decode q8/MMVQ remains codegen-deferred. The audit does not create a new small edit path around the pack lifecycle.

## Principle check

- The audit did not revive killed/refuted routes.
- The new TPE-7a row is correctness/protocol evidence only; it does not overclaim throughput.
- Graph-boundary claims remain separated from kernel math claims.
- The search memory records what not to retry (`naive_per_op_host_sync`, bounded WMMA sweep, spec single-kernel
  shortcut) and what can proceed only behind explicit gates.
