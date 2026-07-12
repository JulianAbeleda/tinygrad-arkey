# S9 causal A/B setup result

An isolated worktree at `b1259638d` was created at `/tmp/tg-hist` and the
existing authority harness was invoked against the exact Qwen3-8B GGUF. A
ctx512 smoke (`K=1`, warmup=1, round=1, pinned-clock) completed, but measured
`4230.3 ms` (121 tok/s), not the historical 116 ms band.

The attempted environment used `PREFILL_V2=1`, `PREFILL_GRAPH_GEMM=1`,
`PREFILL_WMMA_PIPE_PRIMITIVE=1`, and `PREFILL_WMMA_LDS_PRIMITIVE=1`, but the
historical artifact does not record the original route/environment. Therefore
this is not a causal performance comparison: it demonstrates that the
historical worktree cannot be reproduced from the available metadata alone.
The next requirement is recovering the exact S9 command/environment (route,
backend atom flags, direct-packed settings, and model harness revision) before
running K=8/warmup=4/rounds=3 authority. No main worktree or runtime route was
modified.
