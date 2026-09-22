# NVIDIA pp512 support attribution after unroll — 2026-08-29

The final six segments of the existing post-unroll `PROFILE=1` capture were
reclassified by exact program identity. All 1,449 launches are classified and
the category sum closes to the captured device interval union.

| ownership | launches | active us |
|---|---:|---:|
| down (exact two captured N=12288 program identities) | 36 | 461,006.816 |
| V (admitted/live-owned identity) | 0 | 0 |
| residual/RoPE/KV transport (remaining graph programs) | 1,233 | 79,869.184 |
| other support (q8 compact producer) | 180 | 525.088 |
| total | 1,449 | 541,401.088 |

The prior undifferentiated support bucket was 937 launches / 501,106.784 us;
the new down identity map explains 36 launches / 461,006.816 us of it. The
remaining 40294.304 us consists of launches previously assigned to other
regions by the broad parser, so it must not be treated as a residual-bucket
delta. V is explicitly zero because this route has no admitted V-owned
identity; it is not inferred from timing.

The capture is instrumentation-perturbed (`PROFILE=1`) and is not authority
for unprofiled wall performance or recovery estimates.

Evidence: `docs/task_workflow/evidence/nv-prefill-post-unroll-trace-20260829/support-attribution.json`.
