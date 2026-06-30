# pipe_tm2_tn2 PROMOTED to default

**Verdict:** PIPE_PROMOTE_PASS_DEFAULT_FLIPPED (flip commit afe0cae86)

| ctx | old default | new default (pipe) | Δ | tier |
|---|---|---|---|---|
| 512 | 3598 | 4289 | +19.2% | A |
| 1024 | 3506 | 4095 | +16.8% | A |
| 2048 | 3253 | 3708 | +14.0% | A |
| 4096 | 2821 | 3137 | +11.2% | A |
| 8192 | 2234 | 2423 | +8.5% | A |

Rollback: `PREFILL_GEMM_PIPELINE=0` -> old lds2 default (re-confirmed 3598/3506/3253/2821 @512-4096). Output-equivalent (H2). Cherry-pick: `git cherry-pick afe0cae86` to master (isolated 1-file diff).
