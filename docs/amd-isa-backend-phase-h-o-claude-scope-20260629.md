# AMD ISA Backend Phase H-O Claude Scope

## Purpose

This is the handoff scope for taking the native AMD ISA backend from native decode block-tile correctness toward a pure
Q4 8B decode-attention promotion.

Do not redo Q4_K GEMV. Q4_K GEMV is already pure for tracked decode roles through BubbleBeam G3. The remaining pure
claim is blocked by decode attention.

## Current State

- Native AMD ISA backend is opt-in through `DEV=AMD:ISA`.
- Inc 0/1/2/3 passed.
- Phase B/C passed.
- Phase F passed all decode-attention primitive microgates: LDS staging, barrier ordering, `ds_bpermute`, and
  `v_dot2_f32_f16`.
- Phase G has resolved full block-tile op coverage and progressed through numeric-correctness work in the native tile.
- Current handoff note says the first Phase H blocker is known: full-model route through `DEV=AMD:ISA` needs more op
  coverage than the block-tile microgate. `CAST ulong -> float` surfaced when the xlane-score gate ran `m.forward`.

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

## Phase N: BubbleBeam Native Candidate Binding

Goal: expose native backend choices as machine-search candidates.

Candidate axes:

- waitcnt policy
- scheduler policy
- modulo schedule on/off/config
- regalloc pressure policy
- vector/scalar memory choices
- unroll/pipeline depth
- split/topology only if still relevant

Work:

1. Add a declared native attention candidate space.
2. Add candidate generator integration.
3. Add a durable ledger.
4. Require every candidate to target a named counter delta.
5. Evaluator must record token match, route attribution, W==D, waitcnt, VGPR/LDS/scratch, and status.

Artifact:

- `bench/amd-isa-backend-phase-n/latest.json`

Acceptance:

- BubbleBeam can generate/select native attention candidates.
- At least one candidate run records full evaluator output.
- Search stops/classifies if counters do not move.

Verdicts:

- `AMD_ISA_PHASE_N_PASS_BUBBLEBEAM_NATIVE_BINDING`
- `AMD_ISA_PHASE_N_BLOCKED_CANDIDATE_SPACE`
- `AMD_ISA_PHASE_N_BLOCKED_EVALUATOR`
- `AMD_ISA_PHASE_N_SEARCH_SPACE_BUG_NO_COUNTER_MOVEMENT`

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

