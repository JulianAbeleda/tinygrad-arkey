# Machine Search Representation Expansion — Decode + Prefill Scope / Claude Prompt

Date: 2026-06-23

## Mission

Make the oracle primitives **expressible to machine search**.

Current state:

- The evaluator/oracle layer works.
- Bounded high-level searches work.
- Decode Mode A/B and prefill tile-config search have executed.
- The current search surfaces are exhausted.
- The remaining differences are below the current representation:
  - decode: ABI/materialization was solved by a human-discovered whole-buffer
    boundary; native codegen still cannot express `v_dot2` or cross-lane
    reduction; long-ctx residual is strided whole-cache read coalescing;
  - prefill: Tensile residual is K-loop software pipelining + register-pool
    lifetime scheduling, not BK/PAD/DBUF/waves.

The goal is not another sweep. The goal is to design and implement the next
search representation layer so machine search can operate on the primitives
that currently only the oracle/human path provides.

In one sentence:

```text
Turn "oracle-only primitive" into "bounded searchable representation + gates".
```

## Key Distinction

Machine search did not fail as an evaluator. It failed to beat the oracle because
the oracle contains primitives outside the search representation.

Current machine search can express:

- env/policy choices;
- split count;
- combine variant;
- tile constants;
- high-level GEMM tile config;
- route flags;
- ISA/resource gates;
- whole-path authority.

It cannot yet express:

- new ABI/materialization boundaries;
- precompiled-call buffer identity transformations;
- register lifetime / dynamic VGPR pooling;
- instruction interleaving and software-pipelined K-loops;
- renderer lowering alternatives for `v_dot2`;
- renderer lowering alternatives for cross-lane reductions;
- direct schedule-template mutations over AMDGCN/HIP code objects.

This scope is about adding those missing representations.

## Required Reading

Read these first:

1. `docs/decode-oracle-explanation-and-schedule-diff-result-20260623.md`
2. `docs/prefill-schedule-diff-oracle-and-search-reduction-result-20260623.md`
3. `docs/native-codegen-microprimitive-search-result-20260623.md`
4. `docs/decode-mode-b-search-result-20260623.md`
5. `docs/prefill-search-result-20260623.md`
6. `docs/machine-code-translation-roadmap-result-20260623.md`
7. `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`
8. `docs/oracle-guided-gpu-primitive-explorer-runner-design-20260623.md`
9. `docs/project-search-ledger-contract-20260623.md`
10. `docs/primitive-space-learning-loop-lora-first-result-20260623.md`
11. `docs/primitive-space-learning-loop-lora-first-scope-20260623.md`
12. `docs/amd-gpu-holistic-primitive-model-20260623.md`
13. `bench/qk-decode-eval/HARNESS_GUIDE.md`
14. `structure/Development/performance-primitive-research-principles.md`
15. `structure/Development/session-handoff.md`

Inspect:

- `extra/qk_project_search_ledger.py`
- `extra/qk_search_spec.py`
- `extra/qk_decode_search_runner.py`
- `extra/qk_decode_mode_b_execute.py`
- `extra/qk_prefill_search_execute.py`
- `extra/qk_native_codegen_microsearch.py`
- `extra/qk_isa_primitive_audit.py`
- `extra/qk_amdgpu_isa_primitive_audit.py`
- `extra/qk_owned_flash_decode_graph_node.py`
- `extra/qk_owned_flash_decode.hip`
- `extra/qk_prefill_graph_gemm_route.py`
- `bench/qk-project-search-ledger/ledger.jsonl`
- `bench/qk-decode-oracle-explanation/`
- `bench/qk-prefill-schedule-diff-oracle/`
- `bench/qk-native-codegen-microsearch/`

## Non-Goals

- Do not flip defaults.
- Do not claim a speedup without W==D / synced whole-prefill authority.
- Do not run broad random kernel generation.
- Do not reopen already-closed Mode A/B or tile-config grids.
- Do not implement a full renderer rewrite in this scope.
- Do not train LoRA/RLVR in this scope.
- Do not use no-sync/profile/raw dispatch as promotion authority.

## Output

Produce a complete representation-expansion plan and, where cheap, prototype
schema/tools. No performance implementation is required unless the task
explicitly calls for it later.

Primary result doc:

```text
docs/machine-search-representation-expansion-decode-prefill-result-20260623.md
```

Primary artifact dir:

```text
bench/qk-machine-search-representation-expansion/
```

## Phase 0 — Authority And Gap Inventory

Summarize the current oracle-only primitives and their evidence.

Required table:

| lane | oracle-only primitive | current evidence | current search cannot express | expected upside |
|---|---|---|---|---|
| decode | buffer-identity ABI | +13-19% W==D | ABI/materialization transform search | solved/default-on |
| decode | `v_dot2` lowering | ISA present in owned tile, absent in native | renderer dot primitive | learning/generalization |
| decode | cross-lane reduce | ISA present in owned tile, absent in native | renderer shuffle primitive | learning/generalization |
| decode | strided whole-cache coalescing | ctx slope residual <2% | cache-read schedule/coalescing search | low |
| prefill | K-loop software pipelining | static diff + Tensile gap | instruction interleave / prefetch schedule | ~4-5% whole-prefill |
| prefill | register-pool lifetime | PLRAB blocked by VGPR wall | dynamic VGPR lifetime / register pool | enables pipeline |

Deliverable:

```text
bench/qk-machine-search-representation-expansion/gap_inventory.json
```

Verdict:

- `REPRESENTATION_GAP_INVENTORY_READY`

## Phase 1 — Search Representation Taxonomy

Define representation levels.

Required levels:

1. `env_policy`
   - flags, thresholds, route choices.
2. `tile_config`
   - S, TK, BK, PAD, waves, DBUF.
3. `kernel_template`
   - named generated HIP/AMDGCN template parameters.
4. `abi_layout_transform`
   - buffer identity, base+offset, slice avoidance, precompiled-call ABI.
5. `isa_microprimitive`
   - `v_dot2`, cross-lane, LDS staging, vector loads.
6. `schedule_template`
   - K-loop load/compute interleave, wait/barrier placement, prefetch distance.
7. `register_lifetime`
   - dynamic VGPR pool, liveness window, accumulator/prefetch allocation.
8. `renderer_lowering`
   - tinygrad UOp -> AMD renderer transformation.
9. `cross_shape_policy`
   - shape/model/GPU-specific selection over proven primitives.
10. `learned_primitive_spec`
   - LoRA/SFT-generated SearchRow proposal, never authority.

For each level define:

- what it can express;
- candidate examples;
- required gates;
- authority;
- current tools;
- missing tools;
- risk.

Deliverable:

```text
bench/qk-machine-search-representation-expansion/search_representation_taxonomy.json
```

Verdict:

- `SEARCH_REPRESENTATION_TAXONOMY_READY`

## Phase 2 — Decode Representation Expansion Plan

Classify decode surfaces into:

### A. Already Solved / Search Closed

- `env_policy`: S/combine/min_ctx searched, oracle best.
- `tile_config`: TK/VEC/UNROLL searched, oracle best.
- `abi_layout_transform`: buffer-identity ABI solved/default-on.

### B. Low-Priority Searchable With New Representation

- `strided_whole_cache_coalescing`
  - Need representation:
    - whole-cache K/V load grouping;
    - per-head contiguous staging strategy;
    - maybe local staging layout variants;
    - read-vector mapping for strided K/V.
  - Required gates:
    - route fires;
    - no `E_49152`;
    - ISA no spill;
    - byte-identical;
    - W==D at ctx2048/4096;
  - Stop rule:
    - require >1% @ctx4096 and no ctx512 regression; otherwise close.

### C. Learning / Native-Codegen

- `v_dot2_lowering`
  - Need representation:
    - UOp dot pattern -> AMD `v_dot2` intrinsic/lowering.
  - Authority:
    - local correctness + ISA target, not W==D.

- `cross_lane_reduce_lowering`
  - Need representation:
    - warp-axis reduce -> `ds_bpermute`/`ds_swizzle`/permlane.
  - Authority:
    - local correctness + ISA target, not W==D.

Deliverable:

```text
bench/qk-machine-search-representation-expansion/decode_representation_plan.json
```

Verdicts:

- `DECODE_REPRESENTATION_PLAN_READY`
- `DECODE_SPEED_SEARCH_REMAINS_CLOSED`
- `DECODE_CODEGEN_SEARCH_REPRESENTATION_NEEDED`

## Phase 3 — Prefill Representation Expansion Plan

Classify prefill surfaces into:

### A. Already Searched / Closed

- BK/PAD/DBUF/waves;
- occupancy for well-occupied roles;
- LEANADDR / VALU address overhead as speed lever;
- generic tile-config search.

### B. Missing Representation: K-Loop Software Pipeline

Define a schedule-template representation:

```json
{
  "primitive": "kloop_software_pipeline",
  "stages": [
    {"name": "prefetch_next_A", "distance": 1},
    {"name": "prefetch_next_B", "distance": 1},
    {"name": "ds_store_next", "interleave": "inside_wmma_group"},
    {"name": "ds_load_current", "grouping": "before_wmma_microgroup"},
    {"name": "wmma_current", "group_size": 4},
    {"name": "waitcnt", "placement": "before_consumer_only"},
    {"name": "barrier", "placement": "stage_boundary"}
  ]
}
```

Representation must express:

- prefetch distance;
- A/B prefetch separately;
- global-load placement relative to WMMA span;
- LDS store/load placement;
- waitcnt placement;
- barrier placement;
- WMMA group size;
- liveness of prefetched operands;
- VGPR budget.

### C. Missing Representation: Register Lifetime / Pool

Define a register-lifetime representation:

- accumulator VGPR region;
- current tile operand region;
- next tile prefetch region;
- scalar address state;
- lifetimes by K-loop stage;
- max live VGPR constraint;
- spill reject condition;
- dynamic reuse / pool assignment.

Search candidates must be rejected before W==D if:

- VGPR exceeds envelope;
- scratch/spill appears;
- ISA shows no interleaving;
- correctness fails.

Deliverable:

```text
bench/qk-machine-search-representation-expansion/prefill_representation_plan.json
```

Verdicts:

- `PREFILL_REPRESENTATION_PLAN_READY`
- `PREFILL_TILE_CONFIG_SEARCH_CLOSED`
- `PREFILL_SCHEDULE_TEMPLATE_REPRESENTATION_NEEDED`
- `PREFILL_REGISTER_LIFETIME_REPRESENTATION_NEEDED`

## Phase 4 — Generic Search Spec Extension

Extend the conceptual `SearchRow` schema to support representation level and
required gate plugins.

Do not necessarily edit production code unless safe; a schema artifact is
enough for this phase.

Add fields:

- `representation_level`;
- `oracle_primitive_id`;
- `candidate_template_id`;
- `schedule_template`;
- `register_budget`;
- `abi_transform`;
- `isa_targets`;
- `reject_before_compile`;
- `reject_after_isa`;
- `authority_kind`;
- `learning_only`.

Deliverable:

```text
bench/qk-machine-search-representation-expansion/search_spec_extension.json
```

Verdict:

- `SEARCH_SPEC_EXTENSION_SCOPED`

## Phase 5 — Gate Plugin Plan

Define new gate plugins needed for deeper search.

Required gates:

### `abi_identity_gate`

Checks:

- no slice/reshape across precompiled boundary;
- whole buffer identity if required;
- expected base+offset ABI.

### `schedule_interleave_gate`

Checks static disassembly:

- global loads inside WMMA span;
- LDS stores/loads inside steady region;
- waitcnt placement;
- barrier count/placement;
- WMMA grouping.

### `register_lifetime_gate`

Checks:

- VGPR/SGPR;
- scratch/spill;
- estimated live ranges if available;
- reject if VGPR wall hit.

### `renderer_lowering_gate`

Checks:

- target ISA primitive appears (`v_dot2`, `ds_bpermute`);
- correctness preserved;
- no spill.

### `whole_path_authority_gate`

Existing:

- W==D for decode;
- synced whole-prefill for prefill.

Deliverable:

```text
bench/qk-machine-search-representation-expansion/gate_plugin_plan.json
```

Verdict:

- `DEEP_SEARCH_GATE_PLUGIN_PLAN_READY`

## Phase 6 — Minimal Prototype Decision

Pick one smallest prototype per lane.

### Decode prototype options

Recommended:

1. `decode_codegen_cross_lane_microsearch`
   - learning-only;
   - local correctness + ISA target;
   - does not touch default decode.

or:

2. `decode_strided_read_coalescing_probe`
   - low-priority speed;
   - only worth it if owner wants long-ctx >4k push.

### Prefill prototype options

Recommended:

1. `prefill_schedule_interleave_detector`
   - static-only gate/tool;
   - classify kernels as phased vs pipelined.

Then, if authorized:

2. `prefill_kloop_schedule_template_microkernel`
   - tiny shape / local correctness only;
   - proves representation can emit interleaving before trying in-model.

Do not jump directly to a full prefill hand-asm kernel.

Deliverable:

```text
bench/qk-machine-search-representation-expansion/prototype_recommendation.json
```

Verdicts:

- `NEXT_PROTOTYPE_DECODE_CODEGEN_MICROSEARCH`
- `NEXT_PROTOTYPE_PREFILL_SCHEDULE_INTERLEAVE_DETECTOR`
- `FULL_SPEED_SEARCH_NOT_READY_UNTIL_REPRESENTATION_EXISTS`

## Phase 7 — LoRA / Primitive-Space Proposer Integration

Update the learning-loop plan to include these deeper representation levels.

The LoRA/SFT model should propose:

- representation level;
- primitive id;
- bounded knobs;
- required gates;
- stop rules;
- whether the result is speed-search or learning-only.

It must not:

- produce unbounded assembly;
- declare a speedup;
- bypass deterministic gates.

Deliverable:

```text
bench/qk-machine-search-representation-expansion/learning_loop_integration.json
```

Verdict:

- `PRIMITIVE_SPACE_PROPOSER_REPRESENTATION_LEVELS_READY`

## Phase 8 — Result Doc

Write:

```text
docs/machine-search-representation-expansion-decode-prefill-result-20260623.md
```

Required answers:

1. Why did current machine search lose to the oracle?
2. Which oracle primitives are outside current representation?
3. What representation levels are needed?
4. Which decode primitives are now expressible, closed, or missing?
5. Which prefill primitives are now expressible, closed, or missing?
6. What new gates are required?
7. What can be machine-searched next?
8. What remains deterministic hand-implementation / renderer work?
9. What should the LoRA primitive-space proposer emit?
10. What is the next executable prototype?

## Expected Final Verdicts

Expected:

```text
MACHINE_SEARCH_REPRESENTATION_GAP_EXPLAINED
DECODE_SPEED_SEARCH_CLOSED_BUT_CODEGEN_REP_NEEDED
PREFILL_TILE_CONFIG_CLOSED_SCHEDULE_REP_NEEDED
DEEP_SEARCH_GATES_SCOPED
PURE_MACHINE_SEARCH_NOT_READY_UNTIL_REPRESENTATION_EXPANDS
```

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch
`qk-prefill-flag-leak-resolution`.

Task: execute the machine-search representation expansion scope for both decode
and prefill.

The goal is to explain why current machine search loses to the oracle and to
design the missing representation layer that would let search operate on the
actual remaining primitives.

Read first:

- `docs/machine-search-representation-expansion-decode-prefill-scope-20260623.md`
- `docs/decode-oracle-explanation-and-schedule-diff-result-20260623.md`
- `docs/prefill-schedule-diff-oracle-and-search-reduction-result-20260623.md`
- `docs/native-codegen-microprimitive-search-result-20260623.md`
- `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`
- `docs/primitive-space-learning-loop-lora-first-result-20260623.md`
- `bench/qk-decode-eval/HARNESS_GUIDE.md`
- `structure/Development/performance-primitive-research-principles.md`
- `structure/Development/session-handoff.md`

Execute phases:

1. Build gap inventory.
2. Define search representation taxonomy.
3. Write decode representation expansion plan.
4. Write prefill representation expansion plan.
5. Scope SearchRow/spec extension.
6. Scope deep gate plugins.
7. Choose minimal prototypes.
8. Integrate with primitive-space LoRA/SFT plan.
9. Write result doc.

Boundaries:

- no default flips;
- no broad search;
- no kernel implementation;
- no adapter training;
- no RLVR;
- no speed claims without authority;
- preserve history with superseding notes only.

Final response must include:

- verdict labels;
- why current search loses to oracle;
- new representation levels;
- decode plan;
- prefill plan;
- gate plugin plan;
- recommended first prototype;
- artifacts written;
- files changed;
- git status.
