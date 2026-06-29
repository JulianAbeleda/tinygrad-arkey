# AMD ISA Backend Phase H-O Claude Scope

## Purpose

This is the handoff scope for taking the native AMD ISA backend from native decode block-tile correctness toward a pure
Q4 8B decode-attention promotion.

Do not redo Q4_K GEMV. Q4_K GEMV is already pure for tracked decode roles through BubbleBeam G3. The remaining pure
claim is blocked by decode attention.

## Current State

- Native AMD ISA backend is opt-in through `DEV=AMD:ISA`.
- Inc 0/1/2/3, Phase B/C/F/G passed.
- Phase H passed: the model route can inject the native ISA decode tile, `token_match=true`, deterministic, no hidden
  HIP/LLVM/owned fallback.
- Phase I passed: route-bound W==D baseline captured.
- Phase J passed: consumer-only waitcnt is correct.
- Phase K built a sound scheduler, but scheduling alone did not move W==D materially before grid parallelism.
- Grid parallelism passed after `RANGE(GLOBAL)` was mapped to launch grid axes: native attention moved from ~0.4% to
  ~45% of owned.
- Phase L/M tested scheduler/latency and occupancy/LDS pressure. Both were not the dominant remaining lever.
- Phase N0/N1/N2/N3F progressed through evidence-backed performance closure:
  - N0 pinned the first static gap to VALU excess.
  - N1A lowered `exp2` to hardware `v_exp_f32`, moving W==D to about `61.09/57.92 tok/s`.
  - N1B scalar address math was tested and refuted/dead on the live path.
  - N2 per-PC ATT/SQTT attribution hit an infrastructure wall; N2B PMC category attribution bypassed it.
  - N2B/N3F.0 pinned dynamic work volume to the fixed-S whole-cache sweep.
  - N3F dynamic-S launched only valid splits and moved ctx512 to `66.91 tok/s`, with ctx4096 flat as predicted.
- The next correct step is not more native-tile guessing. N3F showed the attention tile is now a limited fraction of
  full decode wall time. Continue with whole-step attribution before spending more work on the native tile residual.

## Global Rules

- Proceed phase by phase.
- Stop at the first hard blocker.
- Do not claim pure until Phase O passes.
- Keep `DEV=AMD:ISA` default-off.
- No hidden HIP/LLVM fallback in the native candidate route.
- No owned attention tile in the native candidate route.
- Existing default AMD route must remain unchanged.
- Do not edit `tinygrad/runtime/autogen/**`.
- Keep existing Inc/Phase gates passing.
- Artifacts must include exact command, selected renderer, route attribution, correctness, and blocker if any.

## Core Files

- `tinygrad/renderer/isa/amd.py`
- `tinygrad/renderer/amd/elf.py`
- `tinygrad/runtime/ops_amd.py`
- `tinygrad/llm/model.py`
- `extra/qk_decode_route_attribution_wd.py`
- `extra/qk_decode_attention_block_tile_microgate.py`
- `extra/amd_isa_phase_*_gate.py`
- `bench/amd-isa-backend-phase-*/latest.json`
- `bench/qk-owned-oracle-parity/latest.json`
- `bench/qk-pure-search-loop/*`
- `docs/pure-machine-search-roadmap.md`

## Phase H: Model Route Binding And Token Correctness

Goal: run the native AMD ISA generated decode-attention tile through the real model forward/generate path,
default-off, and prove route attribution plus token/output equivalence.

Known first blocker:

- `CAST ulong -> float` is not lowered by `AMDISARenderer` in full-model `m.forward`.
- Inspect the exact dtype pair and semantics before adding coverage.

Work:

1. Reproduce the Phase H blocker by running the xlane-score/model-forward gate under `DEV=AMD:ISA`.
2. Confirm the unsupported `CAST ulong -> float` and dump the exact UOp context.
3. Implement the missing cast correctly.
4. Add a small standalone cast microgate.
5. Re-run model forward.
6. If more unsupported model-route ops appear, classify and lower them with the same discipline.
7. Bind the native tile route in-model using the route contract:
   - `DECODE_ATTN_GENERATED_WHOLECACHE=1`
   - `DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1`
   - `DECODE_ATTN_BLOCK_TILE=1`
   - `DECODE_ATTN_AMDGCN_TILE=0`
   - native AMD ISA tile selected for the candidate route
8. Add or extend route attribution to record:
   - `native_block_tile_fired`
   - `hip_llvm_block_tile_absent`
   - `owned_tile_absent`
   - `token_match` or output equivalence
   - selected renderer is `AMDISARenderer`
   - no hidden fallback
9. Run a short model correctness gate. Do not make a performance claim.

Artifact:

- `bench/amd-isa-backend-phase-h/latest.json`

Acceptance:

- Full model route reaches the native tile.
- No HIP/LLVM/owned fallback in the candidate route.
- Missing cast/model-route op coverage is resolved or precisely blocked.
- Token/output correctness passes for the short gate.
- Existing Inc/B/C/F/G gates still pass.

Verdicts:

- `AMD_ISA_PHASE_H_PASS_MODEL_ROUTE_BOUND`
- `AMD_ISA_PHASE_H_BLOCKED_CAST_LOWERING`
- `AMD_ISA_PHASE_H_BLOCKED_MODEL_OP_COVERAGE`
- `AMD_ISA_PHASE_H_BLOCKED_MODEL_ROUTE_BINDING`
- `AMD_ISA_PHASE_H_BLOCKED_ROUTE_ATTRIBUTION`
- `AMD_ISA_PHASE_H_BLOCKED_TOKEN_MATCH`
- `AMD_ISA_PHASE_H_BLOCKED_HIDDEN_FALLBACK`

## Phase I: Route-Bound Native W==D Baseline

Goal: measure the native tile in the real W==D harness before performance optimization. This phase is allowed to be
slow; it establishes ground truth.

Work:

1. Run route-bound W==D with the native tile.
2. Measure at minimum `ctx512` and `ctx4096`.
3. Record native tok/s, owned tok/s, percent of owned, token match, route attribution, instruction markers, waitcnt
   count, and VGPR/LDS/scratch/resource summary if available.
4. Compare against the current generated HIP/LLVM block tile baseline if easy.

Artifact:

- `bench/amd-isa-backend-phase-i/latest.json`

Acceptance:

- W==D completes.
- `route_bound=true`.
- `token_match=true`.
- No hidden fallback.
- Numbers recorded.

Verdicts:

- `AMD_ISA_PHASE_I_PASS_NATIVE_WD_BASELINE`
- `AMD_ISA_PHASE_I_BLOCKED_WD_ROUTE`
- `AMD_ISA_PHASE_I_BLOCKED_TOKEN_MATCH`
- `AMD_ISA_PHASE_I_BLOCKED_RUNTIME_STABILITY`
- `AMD_ISA_PHASE_I_BLOCKED_COUNTER_ATTRIBUTION`

## Phase J: Correct Consumer-Only Waitcnt

Goal: replace conservative drain-every-memory waits with a correct consumer-only waitcnt model.

Work:

1. Track outstanding memory classes:
   - global memory / `vmcnt`
   - LDS / `lgkmcnt`
   - scalar memory / `lgkmcnt`
2. Insert waits only before actual consumers.
3. Preserve correctness on primitive microgates, block tile, model route, and W==D token match.
4. Compare waitcnt count against conservative native baseline, LLVM generated route, and owned hand ASM if available.

Artifact:

- `bench/amd-isa-backend-phase-j/latest.json`

Acceptance:

- Correctness unchanged.
- Waitcnt count drops materially.
- No nondeterminism over repeated runs.

Verdicts:

- `AMD_ISA_PHASE_J_PASS_CONSUMER_WAITCNT`
- `AMD_ISA_PHASE_J_BLOCKED_HAZARD_ANALYSIS`
- `AMD_ISA_PHASE_J_BLOCKED_MEMORY_CLASS_TRACKING`
- `AMD_ISA_PHASE_J_BLOCKED_NONDETERMINISM`
- `AMD_ISA_PHASE_J_NO_COUNTER_MOVEMENT`

## Phase K: Inst-Stream Scheduler

Goal: implement native Inst-stream scheduling to hide latency.

Work:

1. Build or use latency metadata for emitted RDNA3 instructions.
2. Build a dependency DAG over register defs/uses, memory dependencies, barriers, waitcnt constraints, and
   `EXEC`/`VCC`/`SCC` hazards.
3. Add legality-preserving list scheduling.
4. Run on Phase F primitive gates, Phase G tile, and Phase H model route.
5. Compare waitcnt, exposed latency, and tok/s.

Artifact:

- `bench/amd-isa-backend-phase-k/latest.json`

Acceptance:

- Correctness preserved.
- Hotloop schedule counters move toward owned.
- W==D improves over Phase I baseline.

Verdicts:

- `AMD_ISA_PHASE_K_PASS_SCHEDULER_IMPROVES_NATIVE_TILE`
- `AMD_ISA_PHASE_K_BLOCKED_DEPENDENCY_DAG`
- `AMD_ISA_PHASE_K_BLOCKED_LATENCY_MODEL`
- `AMD_ISA_PHASE_K_BLOCKED_ILLEGAL_REORDER`
- `AMD_ISA_PHASE_K_NO_PERFORMANCE_MOVEMENT`

## Phase L: Cross-Iteration / Modulo Scheduling

Goal: attack the known long-context cliff from exposed cross-iteration online-softmax/reduce latency.

Work:

1. Identify the online-softmax/PV recurrence in the native Inst stream.
2. Implement software-pipeline or modulo-schedule candidates.
3. Preserve correctness/token match.
4. Measure `ctx512` and `ctx4096`.
5. Compare against owned: waitcnt should move toward owned's lower envelope and `ctx4096` percent of owned should
   improve materially from the current generated baseline.

Artifact:

- `bench/amd-isa-backend-phase-l/latest.json`

Acceptance:

- `token_match=true`.
- `route_bound=true`.
- `ctx4096` improves materially.
- Schedule counters move toward owned.

Verdicts:

- `AMD_ISA_PHASE_L_PASS_LONG_CONTEXT_SLOPE_IMPROVES`
- `AMD_ISA_PHASE_L_BLOCKED_RECURRENCE_IDENTIFICATION`
- `AMD_ISA_PHASE_L_BLOCKED_SOFTWARE_PIPELINE`
- `AMD_ISA_PHASE_L_BLOCKED_CORRECTNESS`
- `AMD_ISA_PHASE_L_NO_LONG_CONTEXT_MOVEMENT`

## Phase M: Regalloc / Occupancy Quality

Goal: move native tile resource usage toward owned tile quality.

Work:

1. Measure VGPR, SGPR, LDS, scratch/spills, and occupancy.
2. Compare native tile against owned.
3. Add occupancy-aware allocation or work-removal only if needed.
4. Avoid spills on the hot path.
5. Preserve scheduler gains.

Artifact:

- `bench/amd-isa-backend-phase-m/latest.json`

Acceptance:

- No hot-path spills.
- VGPR/resource usage moves toward owned if limiting.
- W==D does not regress.
- Occupancy improves if it was limiting.

Verdicts:

- `AMD_ISA_PHASE_M_PASS_RESOURCE_QUALITY_IMPROVES`
- `AMD_ISA_PHASE_M_BLOCKED_REGALLOC_PRESSURE`
- `AMD_ISA_PHASE_M_BLOCKED_SPILLS`
- `AMD_ISA_PHASE_M_NO_RESOURCE_MOVEMENT`

## Phase N: Performance Closure And Search Binding

Goal: close the remaining decode gap with measured attribution, then expose only proven useful levers to BubbleBeam.

Do not treat Phase N as "try optimizer knobs." Every N subphase must be selected by an artifact-backed bottleneck row.
When a lever is refuted, record it and move to the next measured target.

### Completed N Results To Preserve

- `bench/amd-isa-backend-phase-n0/latest.json`
  - Verdict: `AMD_ISA_PHASE_N0_PASS_THROUGHPUT_DIFF_PINNED`.
  - Static diff pinned native VALU excess.
- `bench/amd-isa-backend-phase-n1a/latest.json`
  - Verdict: `AMD_ISA_PHASE_N1A_PASS_HARDWARE_EXP_LOWERING`.
  - `exp2 -> v_exp_f32`, VALU dropped, W==D improved to roughly `61.09/57.92 tok/s`.
- `bench/amd-isa-backend-phase-n1b/latest.json`
  - Verdict: `AMD_ISA_PHASE_N1B_BLOCKED_SGPR_REGALLOC`.
  - Uniform scalar address math was opt-in/default-off and dead on the real tile path. Do not re-chase it without new
    attribution.
- `bench/amd-isa-backend-phase-n2/latest.json`
  - Verdict: `AMD_ISA_PHASE_N2_PASS_DEGRADED_TIMING_ATTRIBUTION`.
  - Per-PC ATT/SQTT mapping is blocked in this HCQ setup.
- `bench/amd-isa-backend-phase-n2b/latest.json`
  - Verdict: `AMD_ISA_PHASE_N2B_PASS_CATEGORY_ATTRIBUTION_PINNED`.
  - PMC category attribution pinned dynamic loop/instruction volume, not VMEM, LDS wait, or cross-lane wait.
- `bench/amd-isa-backend-phase-n3/n3f_latest.json`
  - Verdict: `AMD_ISA_PHASE_N3F_PASS_DYNAMIC_S`.
  - Dynamic-S valid split launch improved ctx512 W==D `61.09 -> 66.91 tok/s`, ctx4096 flat/no regression.
  - This proved the tile is no longer the obvious whole-step bottleneck.

### Phase N4: Whole-Step Attribution

Goal: identify which full-decode kernels dominate after dynamic-S. This phase decides whether to keep optimizing the
native attention tile, fix gmax/combine, revisit GEMV/FFN, or bind search candidates.

Work:

1. Capture full decode step attribution for owned/default and native-dynamic-S routes at `ctx512` and `ctx4096`.
2. Use `PROFILE=1 PMC=1` where possible. If PMC capture is too expensive for every kernel, collect accurate per-kernel
   GPU time first, then PMC the top kernels.
3. Record every kernel in the decode step:
   - `native_block_tile`
   - `owned_flash_tile_gqa_whole`
   - `flash_state_gmax_*`
   - `flash_state_combine_*` / fused combine
   - Q4_K GEMV / FFN kernels
   - projection kernels
   - cache/copy/small kernels
   - host/runtime overhead if visible
4. Emit a sorted whole-step table:
   - kernel name
   - route owner: native / HIP / owned / generated GEMV
   - calls
   - total GPU ms
   - percent of decode step
   - token-match route evidence
   - PMC category summary if available
5. Compare native route vs owned/default route and identify the largest remaining wall-clock delta.

Artifact:

- `bench/amd-isa-backend-phase-n4/latest.json`
- `bench/amd-isa-backend-phase-n4/summary.md`

Acceptance:

- `token_match=true`.
- `route_bound=true`.
- no hidden fallback.
- deterministic repeated runs.
- whole-step kernel attribution sums to the measured W==D envelope.
- top remaining bottleneck kernel/class is named.

Verdicts:

- `AMD_ISA_PHASE_N4_PASS_WHOLE_STEP_ATTRIBUTION_PINNED`
- `AMD_ISA_PHASE_N4_BLOCKED_ROUTE_ATTRIBUTION`
- `AMD_ISA_PHASE_N4_BLOCKED_TOKEN_MATCH`
- `AMD_ISA_PHASE_N4_BLOCKED_COUNTER_CAPTURE`
- `AMD_ISA_PHASE_N4_INCONCLUSIVE_TIMING_BIMODAL`

### Phase N5: Branch By Whole-Step Bottleneck

Goal: implement exactly one high-value fix selected by Phase N4. Do not optimize a component that is not a top wall-clock
contributor.

Decision tree:

- If `native_block_tile` is still the top delta:
  - N5A: native tile residual dynamic inefficiency.
  - Re-run N2B/N3F PMC at the chosen context and target the remaining per-token work.
- If `flash_state_gmax_*` or combine dominates:
  - N5B: gmax/combine route optimization.
  - Candidate levers: fuse gmax/combine, reduce partials traffic, specialize split count, native-compile combine, or
    reuse owned combine shape.
- If Q4_K GEMV / FFN kernels dominate:
  - N5C: GEMV/FFN route attribution and search binding.
  - Do not redo the solved G3 route blindly. Confirm whether the current route is actually the pure generated G3 path,
    whether shape/role drift occurred, and whether BubbleBeam is selecting it.
- If small kernels/host/runtime dominate:
  - N5D: lifecycle/fusion/small-op tax reduction.
- If no single component dominates:
  - N5E: multi-component search package with strict per-candidate target rows.

Artifacts:

- `bench/amd-isa-backend-phase-n5/latest.json`
- branch-specific subdirectory if needed, for example:
  - `bench/amd-isa-backend-phase-n5/native_tile_residual/latest.json`
  - `bench/amd-isa-backend-phase-n5/combine/latest.json`
  - `bench/amd-isa-backend-phase-n5/gemv_ffn/latest.json`

Acceptance:

- Branch chosen from N4 evidence.
- One minimal implementation lever lands or a precise blocker is recorded.
- W==D measured at `ctx512` and `ctx4096`.
- `token_match=true`, deterministic, no fallback.
- The targeted whole-step row moves, or the branch records `NO_MOVEMENT` and is refuted.

Verdicts:

- `AMD_ISA_PHASE_N5A_PASS_NATIVE_TILE_RESIDUAL_MOVED`
- `AMD_ISA_PHASE_N5B_PASS_GMAX_COMBINE_MOVED`
- `AMD_ISA_PHASE_N5C_PASS_GEMV_FFN_ROUTE_MOVED`
- `AMD_ISA_PHASE_N5D_PASS_LIFECYCLE_TAX_MOVED`
- `AMD_ISA_PHASE_N5E_PASS_MULTI_COMPONENT_SEARCH_READY`
- `AMD_ISA_PHASE_N5_BLOCKED_NO_DOMINANT_TARGET`
- `AMD_ISA_PHASE_N5_BLOCKED_CORRECTNESS`
- `AMD_ISA_PHASE_N5_NO_WD_MOVEMENT`

### Phase N6: Search-Space Binding

Goal: expose only measured, non-refuted decisions to BubbleBeam/native machine search.

Allowed candidates are the levers proven useful by N4/N5. Do not include refuted axes:

- Do not include N1B scalar address math unless new attribution proves a live target.
- Do not include occupancy/LDS cuts as a primary lever; Phase M refuted them for W==D.
- Do not include scheduler-only candidates as a primary lever unless a new profile row shows scheduler pressure.

Candidate examples if supported by N4/N5:

- native dynamic-S on/off and valid split policy
- hardware exp lowering policy
- gmax/combine fusion or native-compile variant
- split count / partials stride / combine count policy
- native tile residual rewrite variant
- GEMV/FFN route selector only if N4 shows it is still a wall-clock delta

Work:

1. Add a declared candidate manifest with each axis tied to an artifact-backed target row.
2. Add candidate generator integration.
3. Add a durable ledger.
4. Add evaluator output:
   - token match
   - route attribution
   - W==D
   - per-kernel time
   - PMC category if available
   - VGPR/LDS/scratch/waitcnt for native kernels
   - target row before/after
5. Search must stop/classify if the target row does not move.

Artifact:

- `bench/amd-isa-backend-phase-n6/latest.json`

Acceptance:

- BubbleBeam can generate/select at least one native attention/full-decode candidate.
- Candidate provenance is search-owned, not manual flags.
- Full evaluator output is recorded.
- Refuted candidates are recorded in the ledger.

Verdicts:

- `AMD_ISA_PHASE_N6_PASS_BUBBLEBEAM_NATIVE_BINDING`
- `AMD_ISA_PHASE_N6_BLOCKED_CANDIDATE_SPACE`
- `AMD_ISA_PHASE_N6_BLOCKED_EVALUATOR`
- `AMD_ISA_PHASE_N6_SEARCH_SPACE_BUG_NO_COUNTER_MOVEMENT`

### Phase N7: Pre-Promotion Package

Goal: assemble the exact route that Phase O will judge.

Work:

1. Freeze the selected candidate route.
2. Re-run W==D at minimum `ctx512`, `ctx1024`, `ctx2048`, and `ctx4096`.
3. Record:
   - owned/default tok/s
   - native/search-selected tok/s
   - percent of owned
   - per-kernel route table
   - token match
   - determinism
   - fallback absence
   - search provenance
4. Update the pure-search ledger with all refuted N levers and selected route.

Artifact:

- `bench/amd-isa-backend-phase-n7/latest.json`

Acceptance:

- One candidate route is ready for Phase O.
- No hidden manual-only flag bundle is required.
- Search provenance is clear.

Verdicts:

- `AMD_ISA_PHASE_N7_PASS_PRE_PROMOTION_PACKAGE_READY`
- `AMD_ISA_PHASE_N7_BLOCKED_SEARCH_PROVENANCE`
- `AMD_ISA_PHASE_N7_BLOCKED_WD_THRESHOLD`
- `AMD_ISA_PHASE_N7_BLOCKED_ROUTE_STABILITY`

## Phase O: Promotion / Pure Claim Gate

Goal: decide whether native generated decode attention replaces the owned tile for Q4 8B decode.

Promotion requirements:

- `route_bound=true`
- `token_match=true`
- no hidden fallback
- generated/native route selected by BubbleBeam
- W==D within promotion threshold of owned
- `ctx512` and `ctx4096` acceptable
- benchmark manifest updated
- owned hand ASM becomes fallback/reference only
- docs updated

Artifact:

- `bench/amd-isa-backend-phase-o/latest.json`

Final verdicts:

- `AMD_NATIVE_Q4_8B_DECODE_ATTENTION_PROMOTABLE`
- `AMD_NATIVE_Q4_8B_DECODE_ATTENTION_CORRECT_BUT_NOT_FAST`
- `AMD_NATIVE_Q4_8B_DECODE_ATTENTION_BLOCKED_BY_SCHEDULER_LIMIT`
- `AMD_NATIVE_Q4_8B_DECODE_ATTENTION_BLOCKED_BY_REGALLOC_LIMIT`
- `AMD_NATIVE_Q4_8B_DECODE_ATTENTION_BLOCKED_BY_CODEGEN_COVERAGE`

## Execution Discipline

1. Start by running existing gates.
2. Start Phase H by fixing `CAST ulong -> float`.
3. Proceed phase by phase.
4. Write an artifact after every phase.
5. Stop at the first hard blocker.
6. Never count owned/HIP/LLVM fallback as native success.
7. Never claim pure before Phase O.
8. Final report must list:
   - last passing phase
   - first failing phase, if any
   - exact blocker
   - commands run
   - files changed
   - artifacts written
