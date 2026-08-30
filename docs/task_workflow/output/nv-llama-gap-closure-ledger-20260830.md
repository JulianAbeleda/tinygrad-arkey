# NVIDIA pp512 llama gap-closure ledger

Date: 2026-08-30

| quantity | measured value |
|---|---:|
| promoted tinygrad median | 34.963556 ms |
| promoted tinygrad throughput | 14,644 tok/s |
| established llama median | 35.019399 ms |
| established llama minimum | 34.680367 ms |
| tinygrad minus llama median | -0.055843 ms |

The matched median gap is closed: tinygrad is 0.055843 ms faster than the
established llama median on the same pp512 Qwen3-8B-Q4_K_M workload. Tinygrad
remains 0.283189 ms above llama's best settled sample, so this is median parity,
not a claim of dominating every repetition.

The closing lever was substrate ownership, not a new arithmetic kernel. The O
opaque program retained two lazy result owners and HCQ physically launched its
q4 main twice per layer. Single-owner realization removes exactly 36 q4 mains,
reduces summed active time by 1.837952 ms, and recovers 2.177346 ms wall. Full
logits and rollback are bit-exact and every arm selects token 198.

## Next gates

1. Run settled R9 confirmation against the fixed llama R9 convention.
2. Generalize the single-owner rule in the opaque PROGRAM substrate so future
   multi-output native bindings cannot acquire duplicate physical ownership.
3. Rerun the regional trace only after R9; category ranking is now secondary to
   preserving the closed median gap.

## Authority

- `docs/task_workflow/evidence/nv-o-single-owner-pp512-20260830/promotion-result.json`
