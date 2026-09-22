# NVIDIA pp512 llama gap-closure ledger

Date: 2026-08-30

| quantity | measured value |
|---|---:|
| promoted tinygrad T/L/T mean R9 median | 35.186591 ms |
| promoted tinygrad settled R9 minimum | 35.002284 ms |
| promoted tinygrad median throughput | 14,551 tok/s |
| fresh intervening llama median | 35.334424 ms |
| fresh intervening llama minimum | 34.765611 ms |
| tinygrad minus llama median | -0.147834 ms |

The fresh cross-runtime bracket ran tinygrad R9, llama R9, then tinygrad R9 in
the same current machine state. Tinygrad medians were 35.152125 and 35.221056
ms; their mean is 35.186591 ms. The intervening llama settled median, excluding
its recorded first-use graph-setup sample, was 35.334424 ms. Tinygrad is
0.147834 ms, or 0.42%, faster at the matched median. The median gap is closed.
Tinygrad remains 0.236673 ms above llama's best settled sample, so this does not
claim dominance over every repetition.

The closing lever was substrate ownership, not a new arithmetic kernel. The O
opaque program retained two lazy result owners and HCQ physically launched its
q4 main twice per layer. Single-owner realization removes exactly 36 q4 mains,
reduces summed active time by 1.837952 ms, and recovers 2.177346 ms wall. Full
logits and rollback are bit-exact and every arm selects token 198.

The causal C/A/C result remains valid: single-owner realization recovered
2.177346 ms against its contemporaneous controls. The correction changes only
the settled endpoint claim, not the mechanism or promotion decision.

## Next gates

1. Generalize the single-owner rule in the opaque PROGRAM substrate so future
   multi-output native bindings cannot acquire duplicate physical ownership.
2. Preserve the fresh T/L/T R9 endpoint and do not scale profiled regional
   shares onto it.

## Authority

- `docs/task_workflow/evidence/nv-o-single-owner-pp512-20260830/promotion-result.json`
- `docs/task_workflow/evidence/nv-o-single-owner-pp512-20260830/r9-confirmation.json`
- `docs/task_workflow/evidence/nv-llama-fresh-r9-20260830/cross-runtime-bracket.json`
