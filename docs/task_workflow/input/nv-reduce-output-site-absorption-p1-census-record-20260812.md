# NV reduce-output site absorption P1 census record (CPU arm)

Status: **P1 substrate SHIPPED; CPU census gate FAILS for the q/k site as
scoped; FFN-down site body-free on CPU. q/k site admission moves to the P2
GPU A/B arm (NV sm_120), where the fp32 q/k route already emits 36+36 bodies
at HEAD. No production default changed; no promotion.**

Scope: `docs/task_workflow/input/nv-reduce-output-site-absorption-scope-20260812.md`.
Census evidence: `docs/task_workflow/evidence/nv-reduce-output-site-absorption-census-20260812.json`.
Branch `nvidia-bringup-20260731`. Three-arm DEV=CPU decode census,
`rollout_greedy_jit_flash`, fixed depth 256, phase decode, callify flags ON in
the baseline-context and promoted arms (the production decode capture
context), markers + callify owned-precompiled-output redirect only in the
promoted arm.

## Arm totals

| arm | programs | `reduce_output_rmsnorm` bodies |
| --- | ---: | ---: |
| baseline (no markers, no callify) | 261 | 0 |
| baseline-context (callify flags, no markers) | 261 | 0 |
| promoted (markers + callify) | 734 | 36 (all `1_4096`) |

Callify flags alone are count-neutral (261 = 261). Markers are what change
the graph, and they change it differently per site.

## Per-site admission on DEV=CPU

| site | markers | carrier | admitted | bodies |
| --- | ---: | --- | --- | ---: |
| FFN-down (`1_4096`) | 2 | CONTIGUOUS base, `precompiled_output_identity=true` | yes | 36 |
| q/k (`32_128`) | 72 | `PERMUTE(CAST(...))`, dtype half, `buffer_identity=false`, `precompiled_output_identity=false` | no (`marker_not_eligible`) | 0 |
| q/k (`8_128`) | 72 | same PERMUTE/CAST carrier | no (`marker_not_eligible`) | 0 |

Selector trace: `accepted 3, entry 13, marker_not_eligible 10`.

## Finding

The q/k markers are not CPU-admissible because their production carrier is
`PERMUTE(CAST(qk_half))`: the typed-view ownership gate requires a
precompiled-output identity proof on the marker input, and the CPU substrate
does not carry one (NV does, via the precompiled-output substrate that the
fp32 q/k route already uses). So on DEV=CPU the q/k site cannot even enter
the body-free contract; it is not a CPU question.

The FFN-down site admits body-free: 36 bodies, `weight_store_after={}`,
`weight_materialization_delta=0`. The promoted arm does fragment the CPU
graph (+473 programs) because callify splits the 36-slot production chains
when the q/k site cannot absorb its markers; that fails the scoped
`program_count_identical_or_reduced` gate as written, but the failure is the
q/k site's non-admission, not an FFN-down body problem.

## P1 substrate shipped (hermetic, CPU-only)

- `test/unit/test_generic_reduce_output.py` +108: per-site admission tests
  and multi-row digest pins for the generic reduce-output primitive.
- `test/unit/test_reduce_output_rmsnorm.py` +9: CPU skips for NV-only tests.
- `test/unit/test_reduce_output_rmsnorm_route.py`: policy test pins NV sm_120
  promotion.
- `tinygrad/codegen/gpudims.py` (CPU thread model only): serialize
  WARP/LOCAL/GROUP_REDUCE ranges when `ctx.has_threads`; `r.arg[-1]` axis fix
  for split ranges; `gshape = global_shape or (1,)`. CUDARenderer
  `has_threads=False`, so the NV path is untouched.
- `tinygrad/llm/device_facts.py` + `tinygrad/runtime/ops_cpu.py`: CPU allocator
  memory probe (`allocation_granularity`, `memory_stats` via /proc/meminfo).
- `scratchpad/nv_reduce_output_rmsnorm_census.py`: three-arm census tool
  (baseline / baseline-context / promoted).

Hermetic suite: `pytest test/unit/test_generic_reduce_output.py
test/unit/test_reduce_output_rmsnorm.py test/unit/test_reduce_output_rmsnorm_route.py`
-> **57 passed, 2 skipped** (deliberate CPU guards on NV-only tests).

## Decision and recommendation

P1's CPU gate cannot certify the q/k site by construction. Do not relitigate
the PERMUTE/CAST identity proof on CPU; the site was already producing
36+36 fused bodies on NV at HEAD under the fp32 q/k route, so the body-free
question for q/k is an NV-arm measurement. The FFN-down site is certified
body-free on CPU and can go straight to the P2 GPU A/B. Next step: P2
real-token reverse wall bracket on the RTX 5090 under the shared bench lock,
each site against the +50 us bar (package both sites if neither clears
alone), exact full-logit SHA-256 `9e6664fd...`, promotion record only on
pass.

## Ledger translation

Unchanged until P2 books: reduce_output row 89 kernels / 392.0 us/token.
Ceilings at the 0.6 census-to-wall map: FFN-down alone ~195.6 tok/s (+3.4),
q/k alone ~197.7 (+5.5), both ~201.3 (+9.1), both at 1:1 pure removal
~207.9 (+15.7). P2 decides which of these book.
