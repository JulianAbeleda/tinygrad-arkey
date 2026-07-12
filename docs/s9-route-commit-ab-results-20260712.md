# S9 route-commit A/B results

Exact authority protocol was run in isolated worktrees with
`DEV=AMD PREFILL_V2=1 PREFILL_GRAPH_GEMM=1`, `K=8`, warmups=4, rounds=3,
ctx512, and pinned clocks:

| Commit | ctx512 ms | tok/s |
|---|---:|---:|
| `7564c7fa8` | 124.9 | 4099 |
| `1be6fcc18` | 124.8 | 4104 |
| `4df06d69a` | 124.8 | 4103 |
| `84536d5c6` | 124.8 | 4101 |

All four report the same S9 hand route (`prefill_pipe_role_selective_generated`)
and match current performance. None is the first causal boundary from the
historical `b1259638d` result (116.8 ms). The regression is earlier in the
history and these route-plan commits are ruled out as causes.
