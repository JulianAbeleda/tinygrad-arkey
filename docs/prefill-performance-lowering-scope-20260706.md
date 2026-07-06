# Prefill Performance Lowering Scope - 2026-07-06

## Purpose

This scope answers the current prefill question:

> We lowered prefill enough for pure-machine-search defaults, so why did the old 8B baseline disappear, and what still
> needs lowering for 8B and 14B performance?

Short answer: the repo has lowered the **default provenance** far enough to pass strict purity, but it has not lowered
the **old fast execution substrate** far enough to recover the previous graph-GEMM performance without the raw WMMA
route.

Two distinct targets are now being conflated:

1. **8B resident-fp16 graph-GEMM recovery**
   - Goal: recover the historical 4.4k-5.1k tok/s 8B pp512 class without `Ops.INS`.
   - Missing piece: generated/scheduler-owned WMMA+LDS staging that replaces `extra/qk/prefill/wmma.py`.
2. **14B/32B memory-safe packed prefill recovery**
   - Goal: move beyond the roughly 365 tok/s direct-packed/VALU ceiling toward llama-class quantized prefill MMQ.
   - Missing piece: generated packed Q4_K/Q6_K or Q4_K/Q8_1 tiled MMQ substrate that avoids fp16 dequant as the main
     work.

These are related because both are prefill matmul problems, but they are not the same lowering problem.

## Current Evidence

### Current strict-pure default

Recent strict-pure warm pp512 measurements:

| Model | Route state | Warm pp512 |
| --- | --- | ---: |
| Qwen3-8B-Q4_K_M | `PREFILL_GRAPH_GEMM=0`, `prefill_v2_scheduler_matmul_default` | about 2436 tok/s |
| Qwen3-14B-Q4_K_M | `PREFILL_GRAPH_GEMM=0`, direct-packed Q4/Q6 where needed | about 358 tok/s |

This path passes `PURE_MACHINE_SEARCH_ONLY=1` because it avoids the raw graph-GEMM route.

### 2026-07-06 progress gates

Two new gates narrow the remaining compiler work:

| Target | Gate | Current result | What it proves | What it does not prove |
| --- | --- | --- | --- | --- |
| 8B/14B generated graph-GEMM baseline | `extra.qk.prefill_v2_schedule_table_gate` | `PREFILL_V2_SCHEDULE_TABLE_APPLIES_PASS` without AMD timing | Representative 8B (`4096x4096`) and 14B (`5120x5120`) PREFILL_V2 graph-GEMM shapes are present in the frozen warmstart table, select LOCAL schedules, and are checked through the same schedule-search worker path used to build the table. | Without `--run-amd`, it proves table coverage and selected opts only; it does not prove current TFLOPS or route-bound LDS operand staging. |
| 8B graph-GEMM recovery substrate | `extra.qk.prefill_graph_gemm_fp16_stage_gate --run-amd` | `PREFILL_GRAPH_GEMM_FP16_SINGLE_OPERAND_STAGE_PROBE_PASS` | A generated fp16 shaped-WMMA kernel can keep one operand in `AddrSpace.LOCAL` with `BufferizeOpts(..., removable=False)`, emit shared local storage plus a barrier, match the direct WMMA output, and avoid raw-marker strings. | It is a tiny fp16 substrate probe, not the fp16 prefill TC route, not a medium GEMM timing gate, not route-bound `Ops.INS` proof, and not 8B performance recovery. |
| 8B graph-GEMM recovery substrate | `extra.qk.prefill_graph_gemm_fp16_stage_gate --run-amd --both-operands` | `PREFILL_GRAPH_GEMM_FP16_BOTH_OPERANDS_STAGE_PROBE_PASS` | A generated fp16 shaped-WMMA kernel can keep both A and B operands in `AddrSpace.LOCAL`, emit two local buffers plus barriers, match direct WMMA output, and avoid raw-marker strings. | It is still a tiny custom-kernel probe, not route-bound prefill execution, not cooperative partitioning, and not a performance gate. |
| 8B route-bound default | `extra.qk.prefill_graph_gemm_route_bound_stage_gate --run-amd --local-stage a` | `PREFILL_GRAPH_GEMM_ROUTE_BOUND_LOCAL_STAGE_PASS` | The actual strict-pure `prefill_v2_scheduler_matmul_default` route can execute the 512³ diagnostic with generated A-operand LOCAL staging, fp16 WMMA, shared local storage, barrier, and no raw `Ops.INS` markers. | It is still a tiny route-bound diagnostic, not a medium warmstart schedule or 8B pp512 performance recovery. |
| 8B graph-GEMM recovery substrate | `extra.qk.prefill_graph_gemm_tile_loop_stage_gate --run-amd` | `PREFILL_GRAPH_GEMM_TILE_LOOP_LOCAL_STAGE_PASS` | A generated fp16 shaped-WMMA kernel can stage a WMMA operand in `AddrSpace.LOCAL` inside an enclosing tile loop while keeping the `STAGE` index tile-shaped (`lane` only), emitting bounded LDS plus a barrier and matching the direct kernel. | It is still a custom-kernel substrate probe, not route-bound scheduler integration and not a medium GEMM timing gate. |
| 8B medium warmstart staging | `extra.qk.prefill_graph_gemm_medium_stage_gate --run-amd --pin-clock` | `PREFILL_GRAPH_GEMM_MEDIUM_LOCAL_STAGE_BLOCKED` | The representative `4096x4096` warmstart `LOCAL:0:4` schedule is correct at about 35 TFLOPS, while forced final-WMMA A-staging is wrong, post-LOCAL final-WMMA staging compile-fails, and scalar-before-contract post staging also compile-fails. | It does not solve performance; it rules out final/scalar WMMA operand wrapping as the Route-B implementation path and points to tile-shaped cooperative staging before `STAGE`. |
| 14B packed/MMQ recovery | `extra.qk.q4k_wmma_full_role_contract_gate` | `Q4K_WMMA_FULL_ROLE_CONTRACT_PASS` | The Q4_K/Q8_1 14B role geometry is centralized, bounded, uses the selected shaped-WMMA surface, and keeps tile-local RAW lifetime bounded to 256 elements. | It is structural only. Full-role execution is still blocked by the missing scheduler-owned tile loop. |
| 14B packed/MMQ recovery | `extra.qk.q4k_wmma_tiled_lifecycle_gate` + `extra.qk.q4k_wmma_tiled_role_shape_exec_gate` | lifecycle pass; role-shape execution blocked | The deleted tiled gate sources have been restored against current APIs; the small generated lifecycle emits iu8 WMMA and stays bounded, while the role-shape gate records the real full-role blocker without falling back. | It still does not execute all 14B role shapes or beat the direct-packed ceiling. |

Run:

```sh
PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_single_operand_stage_gate --run-amd --compact
PYTHONPATH=. python3 -m extra.qk.prefill_v2_schedule_table_gate --compact
PYTHONPATH=. python3 -m extra.qk.prefill_v2_schedule_table_gate --run-amd --pin-clock --compact
PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_fp16_stage_gate --run-amd --compact
PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_fp16_stage_gate --run-amd --both-operands --compact
PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_route_bound_stage_gate --run-amd --local-stage a --compact
PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_tile_loop_stage_gate --run-amd --compact
PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_medium_stage_gate --run-amd --pin-clock --compact
PYTHONPATH=. python3 -m extra.qk.q4k_wmma_tiled_lifecycle_gate
PYTHONPATH=. python3 - <<'PY'
import json
from extra.qk.q4k_wmma_tiled_role_shape_exec_gate import build
print(json.dumps(build(), indent=2))
PY
PYTHONPATH=. python3 -m extra.qk.q4k_wmma_full_role_contract_gate --compact
```

The artifacts are:

- `bench/prefill-graph-gemm-single-operand-stage/latest.json`
- `bench/prefill-v2-schedule-table/latest.json`
- `bench/prefill-graph-gemm-fp16-single-operand-stage/latest.json`
- `bench/prefill-graph-gemm-fp16-both-operands-stage/latest.json`
- `bench/prefill-graph-gemm-route-bound-stage/latest.json`
- `bench/prefill-graph-gemm-tile-loop-stage/latest.json`
- `bench/prefill-graph-gemm-medium-stage/latest.json`
- `bench/q4k-wmma-tiled-lifecycle/latest.json`
- `bench/q4k-wmma-tiled-role-shape-exec/latest.json`
- `bench/q4k-wmma-full-role-contract/latest.json`

## Tracking Scaffold

Machine-readable rows for this scope are in:

- `extra/qk/prefill_performance_lowering_registry.py`
- `extra/qk/prefill_performance_lowering_report.py`

Run:

```sh
PYTHONPATH=. python3 -m extra.qk.prefill_performance_lowering_report
PYTHONPATH=. python3 -m extra.qk.prefill_performance_lowering_report --compact
PYTHONPATH=. python3 -m extra.qk.prefill_performance_lowering_report --target target_2
```

to print JSON without requiring GPU or model artifacts.

### Historical fast 8B baseline

`docs/8b-prefill-generated-route-closed-20260705.md` records the old 8B graph-GEMM authority:

| Context | Historical whole-prefill speed |
| --- | ---: |
| pp512 | 5117 tok/s |
| pp1024 | 4918 tok/s |
| pp2048 | 4439 tok/s |
| pp4096 | 3684 tok/s |

That route was `PREFILL_GRAPH_GEMM=1`. It is now opt-in research because the schedule selection is spec-driven, but the
executing WMMA substrate still lowers through raw instruction lists in `extra/qk/prefill/wmma.py` wrapped as `Ops.INS`.

### 14B packed baseline and target

`docs/prefill-14b-llama-parity-trace-20260704.md` and the int8/MMQ handoff record:

| System/route | 14B pp512 |
| --- | ---: |
| old direct-output packed prefill | about 144 tok/s |
| improved direct-packed/generated tile class | about 173-365 tok/s depending point in history |
| current strict-pure warm run | about 358 tok/s |
| llama.cpp reference | about 1625-1849 tok/s |

The 14B gap is not launch count or raw source bytes. It is the packed quantized matmul substrate. Current tinygrad
spends most time in Q4_K/Q6_K packed GEMM rows; llama's quantized prefill MMQ family does much more useful work per byte.

## What "Lowered Enough" Means Here

The repo's lowering levels are defined in `docs/handwritten-kernel-exhaustive-lowering-scope-20260706.md`.

| Meaning | Status | Why it matters |
| --- | --- | --- |
| Strict default purity lowered enough | Done | `PREFILL_GRAPH_GEMM=0` selects ordinary scheduler-owned/default generated routes. |
| Old fast 8B graph-GEMM lowered enough | Not done | `PREFILL_GRAPH_GEMM=1` still executes through raw `Ops.INS` WMMA substrate. |
| 14B memory-safe route lowered enough for provenance | Mostly done | Direct-packed Q4/Q6 routes are descriptor/spec-owned. |
| 14B memory-safe route lowered enough for performance | Not done | Direct-packed route remains in the dequant/VALU-bound performance class. |

So the previous claim that lowering was complete was true for the **strict default audit queue**, not for **performance
parity with the old graph-GEMM or llama-class packed MMQ**.

## Target 1: 8B Resident-FP16 Graph-GEMM Recovery

### Objective

Make the old fast `PREFILL_GRAPH_GEMM=1` behavior unnecessary by teaching tinygrad/codegen to generate the relevant
fp16 WMMA+LDS staged GEMM path directly.

Success means:

- `PREFILL_GRAPH_GEMM=0` or a pure generated route reaches the old 8B graph-GEMM class.
- No selected default path executes `extra/qk/prefill/wmma.py`.
- `PURE_MACHINE_SEARCH_ONLY=1` remains passing.
- pp512 speed is compared against both:
  - current strict-pure 8B warm median, about 2436 tok/s,
  - historical graph-GEMM 8B pp512, about 5117 tok/s.

### Existing reusable work

Use these; do not duplicate them:

- `docs/codegen-wmma-lds-staging-design-20260705.md`
  - Track 1 schedule search.
  - Track 2A/2B LDS input-staging design.
  - Known blockers around fragment layout, `bufferize(..., removable=False)`, and cooperative partition.
- `extra/qk/prefill_v2_schedule_search.py`
  - Offline schedule table generator.
- `extra/qk/prefill_v2_schedule_table.json`
  - Frozen warm-start table for real 8B/14B shapes.
- `extra/qk/prefill_v2_schedule_table_gate.py`
  - Canonical representative-shape verifier for the frozen warm-start table; use this before treating 8B/14B
    generated graph-GEMM baseline numbers as current.
- `tinygrad/llm/model.py::_build_prefill_v2_warmstart`
  - Reads the warm-start TC schedule table.
- `tinygrad/codegen/opt/postrange.py`
  - TensorCore opt construction and warm-start matching.
- `tinygrad/schedule/rangeify.py`
  - `bufferize_to_store` auto-barrier behavior and `pm_remove_bufferize` removal behavior.
- `extra/qk/prefill_graph_gemm_route.py` and `extra/qk/prefill_schedule_spec.py`
  - Shape/role policy reference only. The raw executing substrate must not be reused as the final implementation.

### What is already done

- Track 1 schedule search is done and wired.
- Warm-start TC opts recover part of the gap versus static codegen.
- The representative warm-start table gate now proves the 8B/14B graph-GEMM baseline shapes are present and use LOCAL
  schedules; run it with `--run-amd` to refresh current TFLOPS.
- Current strict default can run 8B pp512 at about 2436 tok/s warm.
- The raw graph-GEMM route remains available as an opt-in research baseline.
- Tiny generated iu8, fp16 single-operand, and fp16 both-operand shaped-WMMA LOCAL-staging probes now pass on AMD. This
  proves the current `Ops.STAGE` / `BufferizeOpts(None, AddrSpace.LOCAL, removable=False)` / `pm_add_buffers_local`
  substrate can preserve staged WMMA operand layouts in custom generated kernels. It does not prove fp16 route-bound
  graph-GEMM recovery.
- The actual strict-pure default route is now pinned by a route-bound gate: with `PREFILL_TC_LOCAL_STAGE=a`, the 512³
  diagnostic emits fp16 WMMA with generated shared local storage and barriers, without raw `Ops.INS`.
- A tile-loop staging gate now proves the key distinction needed after the medium-gate failure: keep the tile loop as
  an enclosing loop and stage only the per-lane fragment shape. That emits bounded tile-local LDS instead of capturing
  the whole GEMM/reduce shape in the `STAGE` buffer.
- The medium warmstart gate now pins the next failure: final-WMMA operand staging does not compose with real
  `OptOps.LOCAL` schedules. Pre-WMMA forced staging aliases local output lanes and is numerically wrong; post-LOCAL
  final-WMMA staging creates an oversized/unfriendly generated kernel that COMGR rejects; scalar-before-contract post
  staging still allocates a 64K-half LDS buffer because the source value shape carries loop/reduce dimensions. The next
  implementation must stage a tile-shaped cooperative value before final fragment expansion, not wrap the final WMMA
  operand value.
- A naive postrange experiment that wrapped the TC operand `CONTRACT` in `bufferize(... AddrSpace.LOCAL ...)` is not the
  solution: B-only emitted shared local/barrier but was numerically wrong (`rel_rmse` about 1.22 on the 512x512x512
  route-bound probe), while A/both fell off the WMMA route. The route-bound gate now checks Numpy-reference error so this
  class of false positive cannot pass.
- The Q4_K/Q8_1 tiled authority gate sources are restored and current-compatible. The small lifecycle gate passes on AMD,
  while the role-shape gate writes the explicit `scheduler_owned_tile_loop_missing` blocker instead of classifying stale
  artifacts as execution.

### What is missing

#### 1A. Generated LDS input staging for WMMA operands

The current scheduler-owned fp16 path is still below the old raw graph-GEMM route because it lacks the same quality of
LDS staging/cooperative input movement.

Required work:

- Bind the proven fp16 generated LOCAL staging mechanism to the actual fp16 prefill TC route.
- Preserve exact WMMA fragment layout at real prefill fragment shapes. The tiny shaped-WMMA probe proves the primitive
  can work; the route-bound fp16 graph-GEMM layout is still unproven.
- Keep staged bufferizes alive with `removable=False` or equivalent proof that `pm_remove_bufferize` cannot erase them.
- Verify actual AMD kernels contain expected `ds_store`, `ds_load`, and `s_barrier`.

Done criteria:

- Small and medium fp16 GEMM shapes are numerically correct on real AMD.
- LDS traffic is present in emitted AMD code.
- No raw `Ops.INS`, source-string, or hand-assembly substrate is selected.
- A route-bound gate proves the 8B prefill path, not only a custom microprobe, uses the generated staging.

#### 1B. Cooperative partition of staged tiles

Milestone staging that redundantly stores the full tile is correctness-only. The performance win requires cooperative
partitioning across local threads.

Required work:

- Split global-to-LDS stores across workgroup lanes/warps.
- Reconstruct per-lane WMMA fragment reads from the staged layout.
- Avoid LDS bank conflicts; the design doc notes padding is not automatic.
- Keep vectorized LDS loads where possible; avoid scalarizing the staged operand into excessive `ds_load`s.

Done criteria:

- 8B pp512 improves materially over the current strict-pure about 2436 tok/s.
- Per-shape fp16 microbench closes the gap toward the historical raw route.
- The raw graph-GEMM route becomes a baseline only, not the speed path.

#### 1C. Double buffering / software pipeline only if needed

This is the expensive part and should be conditional.

Required work if 1B leaves more than about 20% residual versus the raw route:

- Add a size-2 staged buffer axis or equivalent generated representation.
- Express read-before-overwrite ordering explicitly.
- Avoid relying on emit order for WAR/RAW safety.

Done criteria:

- Race-stress correctness passes repeatedly.
- pp512 moves enough to justify the added scheduler complexity.

### Primary blockers

- WMMA fragment layout through LDS is the correctness blocker.
- Cooperative store/read partition is the performance blocker.
- Barrier/read-overwrite dependency is the double-buffer blocker.
- AMD wedge safety limits online search; schedule search must remain deliberate/offline.

### Non-goals

- Do not promote `PREFILL_GRAPH_GEMM=1` as strict pure while it uses `Ops.INS`.
- Do not add another raw instruction-list emitter.
- Do not chase int8 WMMA as a raw throughput multiplier for 8B fp16; RDNA3 iu8 WMMA was measured throughput-neutral or
  worse versus fp16.

## Target 2: 14B/32B Memory-Safe Packed Prefill Recovery

### Objective

Move 14B prefill out of the current direct-packed/dequant-VALU ceiling and into a generated quantized MMQ class.

Success means:

- 14B pp512 beats the current strict-pure/direct-packed class by a large margin.
- The route keeps Q4_K/Q6_K packed weights resident; it does not require a full fp16 resident copy.
- The selected implementation remains pure under `docs/pure-machine-search.md`.
- The final target is measured against llama.cpp's 14B pp512 reference, about 1625-1849 tok/s.

### Existing reusable work

Use these; do not duplicate them:

- `docs/prefill-14b-llama-parity-trace-20260704.md`
  - Attribution: packed GEMM schedule/substrate gap.
- `docs/prefill-packed-generated-tile-scope-20260704.md`
  - Direct-packed history and refuted tile probes.
- `docs/route-b-iu8-wmma-mmq-design-20260705.md`
  - Q4_K/Q8_1 algebra and Route B design.
- `docs/q4k-wmma-full-role-lowering-solution-scope-20260705.md`
  - Full-role tiled Q4_K/Q8_1 WMMA scope.
- `extra/qk/prefill_int8_wmma_spec.py`
  - Q4_K/Q8_1 Tensor oracle, one-tile tiled proof, descriptors.
- `tinygrad/llm/prefill_routes.py`
  - Existing `PREFILL_Q4K_Q8=wmma` / `wmma_tiled` branches and direct-packed defaults.
- `tinygrad/llm/generated_candidates.py`
  - Generated candidate rows for scheduler default and Q4_K int8 WMMA research substrates.
- Existing Q4_K/Q6_K direct-packed route specs and emitters.

### What is already done

- Direct-packed Q4_K/Q6_K default routes are descriptor/spec-owned for provenance.
- Q4_K int8 WMMA Tensor oracle exists.
- Q4_K int8 WMMA tiled one-tile proof exists.
- Full-model graph explosion is guarded explicitly.
- Scalar sdot4/MMQ research route was deleted after poor measurements/dead-end classification.
- The Q4_K/Q8_1 full-role lowering contract now exists in `extra/qk/q4k_wmma_tile_lowering.py`.
- `extra.qk.q4k_wmma_full_role_contract_gate` passes using the existing shaped-WMMA surface, lifecycle, and no-hand-kernel
  artifacts.

### What is missing

#### 2A. Full-role bounded tiled MMQ lifecycle

The current Q4_K/Q8_1 WMMA route cannot run full 14B shapes because the oracle materializes or implies a huge
`[groups, M, N]` RAW tensor.

Required work:

- Implement the scheduler-owned execution loop for the generated tile lifecycle:
  - output tile loop,
  - Q4_K group loop,
  - tile-local int32 RAW,
  - tile-local QSUM,
  - fp32 scale/min correction,
  - direct final output write.
- Keep `live_raw_elems <= m_tile * n_tile * group_tile`.
- Avoid Python-level Tensor tiling loops that create many graph fragments or large intermediate tensors.

Done criteria:

- Synthetic role-shape execution gate runs real 14B dimensions without loading the model.
- Kernel count and compile time are bounded.
- No graph explosion guard is tripped.

#### 2B. WMMA surface decision

The repo has multiple possible surfaces:

- ordinary `Tensor.matmul(..., dtype=int)` that tensorizes to iu8 WMMA,
- `Ops.SHAPED_WMMA` lowering,
- generated custom UOp descriptors.

Required work:

- Decide which surface can own the full tile lifecycle without materializing global RAW.
- Prove the chosen surface emits `wmma_i32_16x16x16_iu8` on AMD.
- Keep route-local code free of HIP strings, inline asm, and direct builtins.

Done criteria:

- A gate reports selected surface, numeric correctness, no route-local builtin/asm, and visible WMMA binding.

#### 2C. Q6_K parity or fallback policy

14B Qwen3 Q4_K_M includes mixed Q4_K/Q6_K rows. A Q4_K-only MMQ route may leave Q6_K as direct-packed.

Required work:

- Measure how much wall time remains in Q6_K after Q4_K route improvement.
- Either:
  - keep Q6_K direct-packed if its residual wall share is acceptable, or
  - add an analogous generated Q6_K packed/MMQ route.

Done criteria:

- Whole-model 14B pp512 attribution shows the remaining bottleneck.
- The route policy is explicit and does not silently fall back under the Q4_K route name.

#### 2D. Promotion and authority gates

Required work:

- Correctness gate against q8-dequant or full dequant reference.
- Route-bound/no-hidden-fallback gate.
- AMD source/ISA gate proving expected WMMA or packed-MMQ binding.
- Whole-prefill authority on 14B pp512.
- Strict purity audit.

Done criteria:

- Beats current direct-packed about 358-365 tok/s by enough to justify promotion.
- Promotion threshold should be tiered:
  - minimum useful: clear >1.25x over current strict-pure 14B,
  - strong: >2x,
  - parity target: approaches llama.cpp 1625-1849 tok/s.

### Primary blockers

- Full-role tiled lifecycle does not exist.
- The one-tile proof does not prove full-role execution.
- `Tensor.matmul`-style graph construction explodes for full model shapes.
- Q4_K route alone may not remove all packed prefill wall time because Q6_K remains.
- RDNA3 iu8 WMMA is not faster than fp16 WMMA by itself; the win must come from avoiding fp16 dequant and reducing
  bandwidth/work, not from a magical int8 tensor-core rate.

### Non-goals

- Do not revive deleted scalar `sdot4/mmq/mmq_direct` env modes as the promotion route.
- Do not materialize `[groups, M, N]` RAW.
- Do not use the old raw fp16 WMMA substrate as a Q4_K implementation path.
- Do not claim success from one-tile microgates without full-role shape execution.

## Scheduler, Codegen, Or Vocab?

| Need | Main owner | Why |
| --- | --- | --- |
| Recover old 8B graph-GEMM speed | scheduler/codegen | Existing vocab can emit WMMA; missing performance behavior is LDS staging/cooperative scheduling. |
| Replace raw `extra/qk/prefill/wmma.py` | codegen/backend lowering | The executing substrate must move from raw instruction lists to generated lowering. |
| 14B Q4_K/Q8_1 full-role tiled MMQ | codegen + generated route descriptors | The algebra exists; the missing part is bounded tile lifecycle and route-bound generated execution. |
| New hardware intrinsic names | vocab/backend only if blocked | Add only if current WMMA/dot surfaces cannot express the required operation. |
| Better route selection | policy/search after candidates exist | Search cannot find a primitive that the generator cannot express. |

The current diagnosis is therefore:

- **8B:** mostly scheduler/codegen.
- **14B:** codegen/lowering substrate first, then scheduler/search over tile shapes and policies.
- **Vocab:** secondary, only if the chosen generated surface cannot express needed WMMA/dot/cross-lane operations.

## Phased Plan

### Phase 0 - Re-establish baselines

Run and store current authority numbers:

- 8B strict-pure `PREFILL_GRAPH_GEMM=0`.
- 8B research baseline `PREFILL_GRAPH_GEMM=1` with rollback allowed.
- 14B strict-pure/direct-packed.
- Optional llama.cpp authority if available.

Exit criteria:

- One artifact table names route flags, route report, pp512 speed, warm/cold policy, and purity status.

### Phase 1 - 8B generated fp16 WMMA/LDS substrate

Start from `docs/codegen-wmma-lds-staging-design-20260705.md`.

Build order:

1. Correct single-operand LDS staging with preserved WMMA fragment layout.
2. Both-operand staging.
3. Cooperative partition.
4. Double-buffer only if needed.

Exit criteria:

- 8B strict-pure pp512 materially closes the gap to historical graph-GEMM.
- Raw graph-GEMM remains opt-in baseline only.

### Phase 2 - 14B Q4_K/Q8_1 bounded tiled MMQ

Start from `docs/q4k-wmma-full-role-lowering-solution-scope-20260705.md`.

Build order:

1. Tile lowering contract.
2. WMMA surface decision gate.
3. Small multi-tile lifecycle.
4. Synthetic 14B role-shape execution.
5. Model-bound 14B authority.

Exit criteria:

- No graph explosion.
- Whole-prefill 14B beats the current direct-packed class.

### Phase 3 - Residual attribution and Q6_K decision

After Q4_K route improves, profile the full 14B run.

Exit criteria:

- Either Q6_K is small enough to leave as direct-packed, or Q6_K gets its own generated MMQ scope.

### Phase 4 - Promotion

Promotion requires:

- strict purity audit pass,
- route-bound execution,
- correctness,
- no hidden fallback,
- timing artifact,
- rollback/reference quarantine,
- docs and manifest update.

## Final Done Definition

This work is done only when both statements are true:

1. **8B:** The strict-pure default no longer gives up the old graph-GEMM class merely because the raw route is disabled.
2. **14B:** The memory-safe prefill route is no longer stuck at the direct-packed/dequant-VALU ceiling.

Passing the pure audit alone is not the done definition for performance lowering. It only proves the selected route is
not using forbidden handwritten/default surfaces.
