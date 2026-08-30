# NVIDIA pp512 post-Flash ledger

Date: 2026-08-30

| quantity | measured value |
|---|---:|
| contemporaneous pre-Flash control | 38.687355 ms |
| promoted whole-tile MMA Flash | 37.307464 ms |
| measured Flash recovery | 1.379891 ms |
| throughput gain | 3.70% |
| established llama median | 35.019399 ms |
| remaining gap to llama median | 2.288065 ms |

The primitive passes the isolated strict oracle and the whole-model C/A/C. The
candidate graph contains exactly 36 native Flash calls, all packed projection
counts remain unchanged, token 198 is preserved, full logits pass the declared
tolerance, candidate replay is exact, and rollback is exact. It is promoted
with `NV_LLAMA_FATTN_MMA_PP512=0` as the explicit opt-out.

The 1.379891 ms wall recovery is 80.4% of the previously measured 1.716985 ms
Flash category debt. Category accounting and end-to-end wall are not treated
as additive. The remaining 2.288065 ms is a wall difference against the prior
matched llama median, not a claim that one residual category owns that amount.

## Next measurement

Rerun the exact cross-runtime regional trace on this promoted graph. The old
regional ledger predates the Flash promotion and cannot identify the new
largest category without remeasurement.

## Authorities

- `docs/task_workflow/evidence/nv-llama-fattn-mma-pp512-model-20260830/promotion-result.json`
- `docs/task_workflow/output/nv-post-promotion-ledger-20260830.json`
