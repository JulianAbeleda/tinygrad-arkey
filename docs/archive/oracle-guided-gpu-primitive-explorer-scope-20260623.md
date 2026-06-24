# Oracle-Guided GPU Primitive Explorer — Exhaustive Scope / Claude Prompt (2026-06-23)

## Mission

Scope the full "Explore the GPU" system: an oracle-guided, lifecycle-aware primitive search workbench for tinygrad
inference performance.

This is broader than ordinary kernel autotuning. It should use known-good oracles, ISA/resource evidence, graph/runtime
checks, correctness, and whole-path transfer before accepting any candidate.

Core idea:

```text
oracle primitive
-> extract machine/lifecycle facts
-> generate bounded candidates
-> cheap structural/ISA prune
-> correctness
-> route/materialization/lifecycle checks
-> authoritative whole-path benchmark
-> rank, remember, and update learned rules
```

The system should support:

1. decode search;
2. prefill search only when attribution reopens a searchable lane;
3. native-codegen microprimitive search;
4. cross-shape/model/GPU generalization search;
5. project-level search ledger and learned-rule memory.

## Novelty / Positioning

Existing systems already do autotuning:

- TVM AutoTVM / Ansor;
- Triton autotune;
- OpenXLA persisted autotuning;
- Kernel Tuner;
- profiling-guided Triton optimization;
- vendor profilers and SASS/AMDGCN inspection.

The differentiator here is the **lifecycle oracle**:

```text
not just kernel speed
but graph route + ABI/materialization + ISA/resource + token correctness + W==D/whole-prefill transfer
```

So this project should not claim "invented autotuning." It should claim:

```text
oracle-guided lifecycle search for LLM inference primitives
```

## Required Reading

Read these first:

1. `docs/project-wide-machine-search-roadmap-result-20260623.md`
2. `docs/project-wide-machine-search-roadmap-scope-20260623.md`
3. `docs/decode-machine-search-readiness-package-result-20260623.md`
4. `docs/decode-machine-search-execution-result-20260623.md`
5. `docs/prefill-post-decode-parity-frontier-result-20260623.md`
6. `docs/native-codegen-microprimitive-search-result-20260623.md`
7. `docs/machine-code-translation-roadmap-result-20260623.md`
8. `docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md`
9. `docs/amd-gpu-holistic-primitive-model-20260623.md`
10. `bench/qk-decode-eval/HARNESS_GUIDE.md`
11. `structure/Development/performance-primitive-research-principles.md`
12. `structure/Development/session-handoff.md`

Inspect current tools:

- `extra/qk_decode_search_runner.py`
- `extra/qk_decode_search_gate.py`
- `extra/qk_decode_search_execute.py`
- `extra/qk_decode_mode_b_execute.py` if present
- `extra/qk_native_codegen_microsearch.py`
- `extra/qk_project_search_ledger.py`
- `extra/qk_search_spec.py`
- `extra/qk_isa_primitive_audit.py`
- `extra/qk_amdgpu_isa_primitive_audit.py`
- `extra/qk_prefill_whole_synced.py`
- `extra/qk_prefill_per_role_time_tax.py`
- `bench/qk-decode-search-readiness/`
- `bench/qk-decode-machine-search/`
- `bench/qk-prefill-post-decode-parity-frontier/`
- `bench/qk-machine-code-translation/`

## Global Principles

### 1. The oracle is lifecycle-complete

An oracle is not just a fast kernel. It includes:

- source/code object;
- expected ISA/resource facts;
- graph route signature;
- ABI/materialization expectations;
- correctness tokens/reference;
- authority benchmark baseline;
- fallback behavior;
- supported shapes;
- known failure modes.

### 2. Search spaces are bounded

No random free-form kernel generation as the default. Every search space must list:

- knobs;
- allowed ranges;
- expected code path;
- reject rules;
- authority benchmark;
- stop rules.

### 3. Promotion authority is lane-specific

| lane | authority |
|---|---|
| decode | clean synced W==D |
| prefill | clean synced whole-prefill |
| native-codegen microprimitive | local correctness + ISA/resource target, no W==D claim |
| cross-shape | target-specific decode/prefill authority |
| small-op fusion | W==D or whole-prefill after one manual fusion gate |

### 4. Cheap prune before expensive benchmark

Default gate order:

```text
schema/structural
-> route/lifecycle
-> materialization/ABI
-> ISA/resource
-> correctness
-> local diagnostic if useful
-> authority benchmark
```

### 5. Remember failures

Every rejected candidate must record:

- first failed gate;
- artifact links;
- learned reject rule if any;
- whether the failure should prune future search.

## System Architecture

### Component A — Oracle Registry

Create a registry of lifecycle-complete oracles.

Possible artifact:

```text
bench/qk-oracle-gpu-primitive-explorer/oracles.json
```

Initial oracles:

| oracle id | lane | purpose |
|---|---|---|
| `decode_whole_cache_owned_tile_8b_gfx1100` | decode | current default at/above llama |
| `q4k_gemv_warp_8b_gfx1100` | decode/codegen | FFN GEMV schedule oracle |
| `prefill_graph_gemm_8b_gfx1100` | prefill | current prefill default after kv_proj fix |
| `owned_attention_isa_template` | codegen | v_dot2/LDS/cross-lane/no-spill reference |
| `tensile_prefill_reference` | prefill/codegen | external GEMM reference if available |

Required fields:

- oracle id;
- source artifacts;
- code object artifacts;
- supported shapes;
- authority benchmark;
- expected correctness;
- expected route signature;
- expected ISA facts;
- expected materialization/ABI facts;
- current status;
- owner/default policy.

Verdicts:

- `ORACLE_REGISTRY_READY`
- `ORACLE_REGISTRY_PARTIAL`

### Component B — Project Search Spec

Define a shared search-spec format.

Possible artifact:

```text
bench/qk-oracle-gpu-primitive-explorer/search_spec_schema.json
```

Required fields:

- search id;
- lane;
- oracle id;
- candidate generator;
- knobs/ranges;
- structural gates;
- route/lifecycle gates;
- materialization/ABI gates;
- ISA/resource gates;
- correctness gates;
- authority benchmark;
- budget;
- stop rules;
- result schema.

Verdict:

- `SEARCH_SPEC_SCHEMA_READY`

### Component C — Candidate Generators

Each generator must be lane-specific.

#### Decode policy generator

Knobs:

- `S`;
- min ctx;
- route policy;
- combine variant where still valid.

Status:

- ready/near-ready.

#### Decode generated tile generator

Knobs:

- `TK`;
- workgroup size;
- vector width;
- unroll;
- split count;
- offset strategy.

Status:

- gated by Mode B scope.

#### Native-codegen microprimitive generator

Knobs:

- expression shape;
- schedule choices;
- local memory usage;
- reduction method;
- vectorization;
- unroll.

Targets:

- `v_dot2`;
- LDS;
- `ds_bpermute`;
- vector loads;
- no spill.

#### Prefill role-policy generator

Only if attribution says search is justified.

Knobs:

- BN/waves_n by role;
- tile sizes;
- BK;
- LDS layout;
- prefetch/unroll.

#### Cross-shape route-policy generator

Only after target selection/oracles.

Knobs:

- model shape guards;
- route thresholds;
- split count;
- role-specific policy.

### Component D — Gate Plugins

Required gate plugins:

| gate | tool |
|---|---|
| harness contract | `extra/qk_harness_contract.py` |
| decode W==D | `extra/qk_decode_search_gate.py` |
| decode route fire | `extra/qk_decode_route_fire_check.py` |
| decode materialization | `extra/qk_decode_materialization_check.py` |
| ISA audit | `extra/qk_isa_primitive_audit.py` |
| prefill synced authority | `extra/qk_prefill_whole_synced.py` |
| prefill role attribution | `extra/qk_prefill_per_role_time_tax.py` |
| ledger write | `extra/qk_project_search_ledger.py` |

Missing/optional plugins:

- generic oracle loader;
- generic search-spec loader;
- cross-shape oracle builder;
- native-codegen microprimitive scorer;
- unified leaderboard.

### Component E — Project Search Ledger

Use or extend:

- `docs/project-search-ledger-contract-20260623.md`
- `extra/qk_project_search_ledger.py`

Required ledger result:

```text
bench/qk-project-search-ledger/results.jsonl
```

Each entry must include:

- search id;
- candidate id;
- lane;
- oracle;
- knobs;
- gates run;
- first failure;
- authority result;
- artifact links;
- learned rule.

## Phase Plan

### Phase 0 — Inventory Current State

Record:

- HEAD/git status;
- current dirty files;
- existing search tools;
- existing search artifacts;
- current oracles;
- missing pieces.

Artifact:

- `bench/qk-oracle-gpu-primitive-explorer/inventory.json`

Verdict:

- `EXPLORER_INVENTORY_READY`

### Phase 1 — Oracle Registry

Build the initial oracle registry from existing artifacts.

Artifact:

- `bench/qk-oracle-gpu-primitive-explorer/oracles.json`

Stop if decode oracle cannot be loaded/reconciled.

Verdict:

- `ORACLE_REGISTRY_READY`

### Phase 2 — Search Spec Schema

Write the shared search-spec schema and one example per lane:

- decode policy search;
- native-codegen microprimitive search;
- prefill role-policy search placeholder;
- cross-shape placeholder.

Artifacts:

- `bench/qk-oracle-gpu-primitive-explorer/search_spec_schema.json`
- `bench/qk-oracle-gpu-primitive-explorer/spec_decode_policy_example.json`
- `bench/qk-oracle-gpu-primitive-explorer/spec_native_codegen_micro_example.json`
- `bench/qk-oracle-gpu-primitive-explorer/spec_prefill_placeholder.json`
- `bench/qk-oracle-gpu-primitive-explorer/spec_cross_shape_placeholder.json`

Verdict:

- `SEARCH_SPEC_SCHEMA_READY`

### Phase 3 — Unified Runner Design

Do not necessarily implement full runner in this scope unless explicitly requested.

Write:

- `docs/oracle-guided-gpu-primitive-explorer-runner-design-20260623.md`

Required sections:

1. CLI;
2. oracle loading;
3. candidate generation;
4. gate ordering;
5. lane-specific authority;
6. artifact output;
7. ledger write;
8. stop rules;
9. safety boundaries.

Possible CLI:

```bash
PYTHONPATH=. .venv/bin/python extra/qk_oracle_gpu_primitive_explorer.py \
  --spec bench/qk-oracle-gpu-primitive-explorer/spec_decode_policy_example.json \
  --out bench/qk-oracle-gpu-primitive-explorer/runs/decode_policy_001
```

Verdict:

- `EXPLORER_RUNNER_DESIGN_READY`

### Phase 4 — Decode Search Integration

Use existing decode search package as first backend.

Requirements:

- link existing `qk_decode_search_runner.py`;
- do not duplicate it unless necessary;
- ensure results can be written to project ledger;
- ensure HARNESS_GUIDE contract compliance.

Artifact:

- `bench/qk-oracle-gpu-primitive-explorer/decode_backend_integration.json`

Verdicts:

- `DECODE_SEARCH_BACKEND_INTEGRATED`
- `DECODE_SEARCH_BACKEND_NEEDS_ADAPTER`

### Phase 5 — Native-Codegen Microprimitive Integration

Use existing:

- `docs/native-codegen-microprimitive-search-result-20260623.md`
- `extra/qk_native_codegen_microsearch.py`

Define:

- target ISA facts;
- candidate generator;
- correctness scorer;
- ISA scorer;
- ledger mapping.

Artifact:

- `bench/qk-oracle-gpu-primitive-explorer/native_codegen_backend_integration.json`

Verdicts:

- `NATIVE_CODEGEN_SEARCH_BACKEND_INTEGRATED`
- `NATIVE_CODEGEN_SEARCH_BACKEND_SCOPE_ONLY`

### Phase 6 — Prefill Search Gate

Do not run prefill search yet unless current prefill docs say it is ready.

Define a gate:

```text
prefill search allowed only if role-specific residual is material and kernel/searchable
```

Artifact:

- `bench/qk-oracle-gpu-primitive-explorer/prefill_search_gate.json`

Verdicts:

- `PREFILL_SEARCH_GATED_OFF_AT_REST`
- `PREFILL_SEARCH_READY_ROLE_SPECIFIC`
- `PREFILL_SEARCH_NEEDS_ATTRIBUTION`

### Phase 7 — Cross-Shape Search Gate

Define target-selection requirements:

- model available;
- baseline oracle available;
- correctness harness;
- route eligibility;
- expected cost.

Artifact:

- `bench/qk-oracle-gpu-primitive-explorer/cross_shape_search_gate.json`

Verdicts:

- `CROSS_SHAPE_SEARCH_NEEDS_TARGETS`
- `CROSS_SHAPE_SEARCH_READY_AFTER_TARGET_SELECTION`

### Phase 8 — Result + Roadmap

Write:

- `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`

Required sections:

1. Verdict.
2. Relation to existing autotuning systems.
3. Oracle registry.
4. Search spec schema.
5. Backend integration status.
6. Decode search status.
7. Prefill search gate.
8. Native-codegen microprimitive search status.
9. Cross-shape gate.
10. What can be searched now.
11. What remains blocked.
12. Next implementation step.
13. Files changed.
14. Git status.

Update if useful:

- `docs/README.md`
- `structure/Development/session-handoff.md`
- `structure/Development/performance-primitive-research-principles.md`

## Final Verdict Labels

- `ORACLE_GUIDED_GPU_PRIMITIVE_EXPLORER_SCOPED`
- `ORACLE_REGISTRY_READY`
- `SEARCH_SPEC_SCHEMA_READY`
- `DECODE_SEARCH_BACKEND_READY`
- `NATIVE_CODEGEN_MICROSEARCH_BACKEND_READY`
- `PREFILL_SEARCH_GATED_OFF`
- `CROSS_SHAPE_SEARCH_NEEDS_TARGETS`
- `EXPLORER_RUNNER_DESIGN_READY`

## Boundaries

- Do not run broad autonomous search.
- Do not start prefill search unless gate says ready.
- Do not start cross-shape search without target selection.
- Do not flip defaults.
- Do not change model behavior.
- Do not touch decode/prefill kernels in this scoping task.
- Do not compare with no-sync/raw timings as authority.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

The owner wants to scope an "Explore the GPU" / oracle-guided primitive search system: as much machine search as
possible, without losing lifecycle correctness.

Read and execute:

```text
docs/oracle-guided-gpu-primitive-explorer-scope-20260623.md
docs/project-wide-machine-search-roadmap-result-20260623.md
docs/decode-machine-search-readiness-package-result-20260623.md
docs/prefill-post-decode-parity-frontier-result-20260623.md
docs/native-codegen-microprimitive-search-result-20260623.md
bench/qk-decode-eval/HARNESS_GUIDE.md
```

Do not start broad search. This is a scoping/infrastructure consolidation task.

Required work:

1. Inventory existing search tools/artifacts.
2. Build an initial oracle registry.
3. Define a shared search-spec schema.
4. Design the unified runner.
5. Map the existing decode search package into the explorer.
6. Map native-codegen microprimitive search into the explorer.
7. Define prefill search gate.
8. Define cross-shape search gate.
9. Write `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`.
10. Update README/handoff/principles if useful.

Final response must include:

- final verdict;
- what is novel/differentiated vs normal autotuning;
- oracle registry status;
- search-spec status;
- decode backend status;
- native-codegen backend status;
- prefill gate status;
- cross-shape gate status;
- what can be searched now;
- what remains blocked;
- files changed;
- git status.
