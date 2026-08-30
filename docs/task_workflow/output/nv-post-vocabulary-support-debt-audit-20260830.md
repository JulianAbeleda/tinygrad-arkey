# NVIDIA pp512 post-vocabulary support debt audit

Date: 2026-08-30

## Result

**STOP: no bounded candidate is admitted.** The current matched ledger does
not provide an exact primitive-shaped sub-debt with a defensible >=0.5 ms
whole-model projection, and the sequence Q/K candidate is independently
blocked by an NV MMU fault for multi-head geometry.

## Exhaustive retained accounting

The current HCQ authority reports 1,467 classified intervals and zero unknown
intervals. The post-unroll identity attribution reports 1,449 launches:

| service/region | launches | active time |
|---|---:|---:|
| down | 36 | 461.007 ms |
| residual/RoPE/KV transport | 1,233 | 79.869 ms |
| other support (Q8 compact producer) | 180 | 0.525 ms |
| total post-unroll attribution | 1,449 | 541.401 ms |

The matched unprofiled-region ledger separately reports residual/RoPE/KV
support at 4.224608 ms, with 721 elementwise `E_*` launches and 108 `r_*`
reshape/reduction-staging launches in the semantic map. These are complete
classifications, but the PROFILE=1 attribution is not unprofiled wall
authority and cannot be used as a recovery estimate.

## Largest actionable-shaped debt

The largest named support family is elementwise residual/RoPE/KV pack/unpack:
721 launches in the 4.224608 ms matched region. Its semantic map combines
required math with transport and does not isolate a single fusion boundary or
provide a correctness-preserving savings estimate. The 108 `r_*` launches are
transport/materialization and likewise lack an admitted one-program ABI.

The existing retained evidence therefore supports measurement and accounting,
not candidate investment. No fresh candidate was run, no model route was
changed, and no whole-model C/A/C or R9 claim is made.

## Authorities

- `docs/task_workflow/output/nv-prefill-current-hcq-ledger-20260829.md`
- `docs/task_workflow/output/nv-prefill-post-unroll-support-attribution-20260829.md`
- `docs/task_workflow/output/nv-support-semantic-map-20260829.md`
