# NVIDIA pp512 llama gap-closure ledger

Date: 2026-08-30

| quantity | measured value |
|---|---:|
| promoted tinygrad settled R9 median | 35.152125 ms |
| promoted tinygrad settled R9 minimum | 35.002284 ms |
| promoted tinygrad median throughput | 14,565 tok/s |
| established llama median | 35.019399 ms |
| established llama minimum | 34.680367 ms |
| tinygrad minus llama median | +0.132726 ms |

The settled R9 confirmation supersedes the faster R3 endpoint used in the
initial promotion ledger. Tinygrad is 0.132726 ms, or 0.379%, slower than the
established llama median on the same pp512 Qwen3-8B-Q4_K_M workload. This is
near parity, but the median gap is not literally closed. Tinygrad remains
0.471758 ms above llama's best settled sample.

The closing lever was substrate ownership, not a new arithmetic kernel. The O
opaque program retained two lazy result owners and HCQ physically launched its
q4 main twice per layer. Single-owner realization removes exactly 36 q4 mains,
reduces summed active time by 1.837952 ms, and recovers 2.177346 ms wall. Full
logits and rollback are bit-exact and every arm selects token 198.

The causal C/A/C result remains valid: single-owner realization recovered
2.177346 ms against its contemporaneous controls. The correction changes only
the settled endpoint claim, not the mechanism or promotion decision.

## Next gates

1. Remeasure the remaining 0.132726 ms before investing; it is small enough to
   be run-to-run noise or boundary/service residue.
2. Generalize the single-owner rule in the opaque PROGRAM substrate so future
   multi-output native bindings cannot acquire duplicate physical ownership.
3. Preserve the R9 endpoint and do not scale profiled regional shares onto it.

## Authority

- `docs/task_workflow/evidence/nv-o-single-owner-pp512-20260830/promotion-result.json`
- `docs/task_workflow/evidence/nv-o-single-owner-pp512-20260830/r9-confirmation.json`
