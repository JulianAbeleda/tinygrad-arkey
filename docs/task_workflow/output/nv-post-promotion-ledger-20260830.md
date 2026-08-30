# NVIDIA pp512 post-promotion ledger

Date: 2026-08-30

| quantity | measured value |
|---|---:|
| promoted candidate mean wall | 38.4729335 ms |
| llama minimum wall | 34.680367 ms |
| llama median wall | 35.019399 ms |
| wall gap to llama median | 3.4535345 ms |
| vocabulary recovery | 2.5914185 ms |
| O recoverable ceiling | ~0.0662 ms |
| Flash category debt | 1.716985 ms |

The wall gap is a matched end-to-end difference; category debts are measured
regional exposures and are not additive recovery claims. Vocabulary recovery
is retained as measured. O fusion is rejected because its recoverable ceiling
is below the investment threshold. Support/RoPE/KV has no isolated primitive
with a defensible >=0.5 ms whole-model projection. Flash is the only remaining
measured category-level body lever above 0.5 ms, but the exact llama internal
ABI is unavailable.

## Decision

No further promotion is admitted from the current O or support evidence.
The next build is a clean-room Flash implementation with an independently
specified ABI, complete numerical oracle, and matched C/A/C plus R9 gates.

## Authorities

- `docs/task_workflow/output/nv-post-vocabulary-support-debt-audit-20260830.md`
- `docs/task_workflow/output/nv-prefill-current-hcq-ledger-20260829.md`
- `docs/task_workflow/input/nv-prefill-post-substrate-luna-low-scope-20260829.md`
