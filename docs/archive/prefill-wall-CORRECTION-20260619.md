# CORRECTION — "prefill host-dispatch-bound / wall broken" was WRONG; it's busy-wait + unreliable graph timestamps

Supersedes `prefill-wall-broken-host-dispatch-bound-20260619.md` (claims there RETRACTED). Deeper measurement
(cProfile + graph-count) corrected two over-conclusions.

## What was wrong
- **"Host-dispatch-bound, ~1.3ms/kernel dispatch":** WRONG. cProfile of the warm prefill replay shows the wall is
  dominated by `hcq.py:wait` (busy-polling the GPU-done signal: 483K `value`/`to_mv`/`view` reads) — `__call__` +
  `submit`/dispatch are CHEAP (0.002s for 25 calls). CPU=wall is because the host **busy-WAITS for the GPU**, which
  is consistent with **GPU-bound** (normal HCQ), NOT dispatch-bound.
- **"GPU compute = 115ms (ProfileGraphEvent span)":** UNRELIABLE. Re-running, the graph timestamps are stale/racy:
  one graph reported span = 68 SECONDS, total "busy" = 342us (both bogus). So the clean GPU-vs-wall split was NOT
  achieved — the ProfileGraphEvent timestamp path does not give trustworthy numbers here.

## What is reliable
- **cProfile:** prefill wall is busy-wait on the GPU signal (the host spins) -> consistent with GPU-bound; dispatch
  is cheap.
- **Structure:** the prefill forward emits **5 distinct jit graphs** (32/64/128/256/249 kernels) = 5 submits/sync
  points per forward. (The 32/64/128/256 doubling is suspicious — possibly batched/warmstart artifacts captured per
  forward; worth investigating, but timestamps unreliable.)
- **e2e (clock-controlled, SOLID):** all 4 matmul-kernel wins -> ~1.00x; concrete-KV start_pos -> 1.24x byte-identical.

## Honest boundary (unchanged from the completeness assessment)
The **exact GPU-time breakdown of prefill remains unmeasurable on this stack**: PMC perturbs (impossible
757>332ms), ProfileGraphEvent timestamps are corrupt (68s spans), JIT replay emits no clean per-kernel data, eager
is untuned. We know the LEVER (matmul=no, attention/concrete=yes via 1.24x) from e2e A/Bs, but not a clean
quantitative GPU split.

## Actionable (unchanged)
The validated prefill lever is **concrete-start_pos (1.24x)** and possibly **explicit TC attention on concrete KV**
+ **fewer jit graphs** (the 5-graph structure -> if collapsible to 1, fewer sync points). Matmul is not the lever.
The "reduce per-kernel dispatch" lever (#2) is REFUTED as stated (dispatch is cheap; the wall is GPU-wait).

## Integrity note
Two over-conclusions this session (host-dispatch-bound; 115ms GPU split), both corrected by deeper measurement.
Lesson: CPU=wall means busy-WAIT (GPU-bound), not dispatch-bound; and ProfileGraphEvent signal timestamps are not
trustworthy on this stack (verify against a second read before concluding).
