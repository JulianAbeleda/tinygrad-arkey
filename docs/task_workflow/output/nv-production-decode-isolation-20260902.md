# NVIDIA production decode isolation

## Decision

The promoted live-context Flash kernels pass the fixed-depth decode gate. The
remaining integrated failure is prefill-v2 memory residency, not decode
throughput or context scaling.

| depth | tinygrad tok/s | llama tok/s | tinygrad delta |
| ---: | ---: | ---: | ---: |
| 512 | 245.822 | 247.784 | -0.79% |
| 1024 | 246.270 | 244.343 | +0.79% |
| 2048 | 238.190 | 234.713 | +1.48% |
| 4096 | 225.992 | 226.022 | -0.01% |

All four points are inside the 5% admission bound. The 1024/2048/4096 points
used generic, JIT-disabled prompt construction with proactive decode prewarm
suspended. After the KV prefix existed, the live-context lease was restored
and the one reachable full-logits graph captured lazily in discarded warmup
tokens. Every timed token used the installed production decode route.

## Memory localization

Three fresh-process controls failed before a depth-1024 token could be timed:

| setup | failure |
| --- | --- |
| four depths in one production process | 30.13 GB used; 10.62 MB allocation failed after depth 512 |
| one request-scoped S10 production process | 30.20 GB used; 10.62 MB allocation failed in prefill replay |
| decode prewarm disabled, eager prefill-v2 | 30.27 GB used; 10.62 MB allocation failed in prompt computation |

Disabling prefill-v2 for untimed KV construction removed the failure. This
proves the S10/S18/S34 decode graphs are not the source of the memory wall.

## Scope boundary

This closes the steady-state decode ledger, not the full prefill/decode
lifecycle. Production prefill-v2 must reduce or release its workspace before a
normal long-context request can reach decode on this device. The benchmark
isolation flags are diagnostic and are not a promoted runtime fallback.

Evidence: `docs/task_workflow/evidence/nv-production-endpoint-sweep-20260902/`.
