# In-Model Integration Penalty Audit Scope - 2026-06-20

Verdict: `PASS_INMODEL_INTEGRATION_PENALTY_AUDIT_SCOPED`

The prefill breakthrough changes the audit standard for both prefill and decode:

```text
isolated kernel win != production route win
```

## Source Rows

| row | value | source |
|---|---:|---|
| PREFILL_V2 in-model warm full-forward | `2797 tok/s`, `183ms / 512`, `~45 TFLOPS effective` | `prefill-amd-LEARNINGS-BANKED-and-prefill-benchmark-20260620.md` |
| llama pp512 reference | `3020 tok/s` | same source |
| dependency-free GEMM isolated GPU-time | `78.6 TFLOPS median` | `prefill-amd-gemm-gputime-thorough-20260620.md` |
| Tensile `.co` isolated GPU-time | `70.9 TFLOPS median` | same source |
| decode mixed q8 lifecycle | `123.64us` vs `115.24us` target | `decode-owned-q8-lifecycle-attribution-result-20260620.md` |

Topline ratios:

| ratio | value |
|---|---:|
| prefill in-model / isolated GEMM TFLOPS | `0.572` |
| isolated ours / isolated Tensile | `1.109` |
| production prefill / llama | `~0.93` |

## Audit Questions

1. How much of the `78.6 -> 45 TFLOPS` gap is matmul coverage versus attention, KV writes, norms/elementwise,
   scheduler/JIT, and fusion boundaries?
2. Which model roles dominate `PREFILL_V2` wall time after the dependency-free GEMM win is banked?
3. What is the Amdahl ceiling for more GEMM-only work?
4. Are isolated and in-model timings using comparable clock, launch path, and host/GPU separation?
5. Does decode q8 show the same component-to-lifecycle penalty pattern, especially the unstable producer row?

## Required Measurements

| id | measurement | gate |
|---|---|---|
| AUDIT-1 | source-of-truth row ledger | committed rows distinguish isolated GPU-time, batch-amortized time, and in-model tok/s |
| AUDIT-2 | prefill role wall-time split | split `PREFILL_V2` into FFN GEMMs, attention, KV/update, norms/elementwise, scheduler/JIT residual |
| AUDIT-3 | matmul coverage / Amdahl ledger | quantify maximum tok/s movement possible from further GEMM-only work |
| AUDIT-4 | timing-method parity | isolated kernels use raw `AMDProgram wait=True`; in-model rows record warm JIT wall and optional GPU sum |
| AUDIT-5 | decode lifecycle cross-check | q8 producer/consumer rows get the same launch-path, batch-isolate, lifecycle-attribution treatment |

## Kill Conditions

- Do not start another prefill GEMM microkernel unless AUDIT-3 shows material end-to-end headroom.
- Do not promote decode q8 from isolated component speed; require lifecycle/in-model proof.
- Do not compare rows across different launch paths without explicit host/GPU separation.

## Next Executable Work

Build the AUDIT-2/AUDIT-3 probe:

```text
PREFILL_V2 warm full-forward role split -> Amdahl ledger -> updated promotion decision
```

For decode, reuse the same standard on the q8 lifecycle:

```text
producer-only batch isolate -> producer with lifecycle buffers -> consumer -> lifecycle route
```
