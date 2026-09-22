# NV reduce-output Phase 6 wall bracket record (gates PASS, bracket NO-GO)

Status: **NO-GO for promotion. Phase 6 route efficiency landed and every gate
PASSES (smoke, exact logits, census with 18 coalesced fused bodies), but the
reverse control/candidate/control wall bracket still does NOT promote: the
candidate is 18.5 us/token SLOWER than the control bracket median (5.420 vs
5.401 ms/token, -0.63 tok/s). The +495.330 us norms row stays unbooked and no
policy promotion happens. The residual slowdown is now intrinsic to the fused
route: the fused body plus its output materialization costs more than the
ordinary reduce + epilogue + cast trio it replaces, even with zero flag-leak
side effects and 22 fewer kernels.**

Scope: `docs/task_workflow/input/nv-reduce-output-phase6-route-efficiency-scope-20260810.md`.
Branch `nvidia-bringup-20260731` (HEAD `b93bc18c4`). Campaign harness
`extra/llm_research/decode/nv_reduce_output_primitive_ab.py`, run 2026-08-10 on
the RTX 5090 (sm_120, Qwen3-8B-Q4_K_M at fixed depth 512) under the shared GPU
bench lock with a fresh process per arm; record artifact
`docs/task_workflow/output/nv-reduce-output-phase6-wall-bracket-20260810.json`.

## What Phase 6 changed

Workstream A (`tinygrad/schedule/rangeify.py`, committed `554cbd3f7`-era
follow-up): the C6 CALL-input lowering now coalesces ONE fused body and ONE
weight materialization per unique `REDUCE_OUTPUT` marker across all of its
consumers, instead of one body per consuming call argument (54 -> 18 bodies).

Workstream B (`tinygrad/callify.py`, committed `b93bc18c4`): the owned-output
redirect and direct invocation-input view are scoped to functions on the
marker's OUTPUT boundary. `transform_to_call` now computes
`(route_ids, out_route_ids)`; `out_route_ids` contains only terminal functions
(body result carries the marker) and producers feeding a terminal marker.
Marker-bearing producers whose output is ordinary (the production per-block
`_run`, whose result is the residual stream) keep the closed-graph spelling, so
the residual `E_32_32_4` kernel identities stop shifting. New regression test
`test_non_terminal_marker_producer_keeps_closed_graph_spelling` mirrors the
chained `_run` shape and fails under the previous predicate.

## Evidence (2026-08-10, fresh processes under the GPU bench lock)

1. Phase 0 (NV render smoke) PASS: the candidate graph survives on sm_120 and
   the observed decode window contains 18 fused
   `reduce_output_rmsnorm_1_4096` bodies.
2. Phase 1 (exact full-logit gate) PASS: control and candidate logits SHA
   identical `70838f5237ce2cf215e937caed807c4827daa1336e69c3d6c396b1aad4434819`;
   token streams identical, shape `[32, 1, 151936]`, all rows finite, sampled
   token == argmax on every row.
3. Phase 2 (census gate) PASS: fused bodies 18 (control 0); rmsnorm_reduce
   56 -> 38 (drop 18, consistent with body count); rmsnorm_epilogue 55 -> 37
   (removed 18); q/k norm reduce roles unchanged at 36; kernels 936 -> 914
   (honest net -22). `callify_redirect_side_effects` is reduced to the fused
   body's own footprint: per fused norm the ordinary
   `[E_32_32_4_fab82d40 cast + r_16_256 reduce + E_32_32_4_f14a5cc0 epilogue]`
   trio becomes `[E_32_32_4_8eeb0be1 materialization + fused body]`
   (3 -> 2 kernels, identity swap on the materialization, -4 residual casts
   removed). The previous flag-leak reshuffle (-36/+36/+54/-71 across
   non-norms families) is gone.
4. Phase 3 (reverse wall bracket) NOT_PROMOTED, identical token streams
   (stream hash `f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905`
   across all three arms), no rejected high-contention samples:

| arm | median ms/token | tok/s |
| --- | ---: | ---: |
| control A | 5.3973 | 185.28 |
| candidate | 5.4196 | 184.52 |
| control B | 5.4049 | 185.01 |

   Candidate minus control A: -22.3 us; minus control B: -14.7 us; minus the
   bracket median: -18.5 us. Promotion requires >= +50 us vs both controls.

## Why it still does not pay

Per fused norm the candidate route runs `[E_32_32_4_8eeb0be1 (2.56 us) + fused
body (5.98 us)]` = 8.54 us against the control trio `[cast (1.66 us) + reduce
(3.92 us) + epilogue (2.34 us)]` = 7.92 us: the fused body saves 0.28 us over
reduce + epilogue, but the output materialization around the fused body's fresh
buffer is 0.9 us more expensive than the ordinary cast it swaps with, and 18
fewer kernel launches do not make up the difference (census-window `kernel_us`
5903.45 vs 5888.19). The cooperative body itself would need to be cheaper than
the two ordinary kernels it replaces by more than the materialization penalty,
or land its output directly in the consumers' buffers without the extra cast.

## Rules honored

No policy promotion: `decode-reduce-output-rmsnorm-route-policy.json` stays
`promoted_targets: []`; no model wiring change; no default flip; exact-output
contract unchanged (logits SHA identical). Scratch artifacts in /tmp only;
committed artifacts are small JSON + markdown.
