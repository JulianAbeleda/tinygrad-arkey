# Runtime-KV + ISA Audit + Native Codegen Three-Lane Scope (2026-06-23)

## Mission

Scope the three lanes the owner wants to pursue from the post-exhaustion roadmap:

2. **Runtime-KV core persistence** — biggest remaining speed prize, core-runtime-blocked.
3. **ISA audit infrastructure** — ready guardrail, should become reusable infrastructure.
6. **Native tinygrad codegen learning** — long-term path to folding proven escape-hatch primitives back into tinygrad.

These lanes are related but must not be conflated:

| lane | nature | expected near-term W==D | ownership |
|---|---|---:|---|
| Runtime-KV core persistence | tinygrad runtime/graph lifecycle capability | high if solved (`~+11%`, parity-class) | core runtime |
| ISA audit infrastructure | tooling / evidence layer | none directly | performance infrastructure |
| Native codegen learning | compiler/codegen capability | none unless tied to a fresh residual gap | compiler/backend |

The immediate objective is **not** to implement all three. The objective is to produce executable scopes and dependency
ordering so future work does not mix runtime semantics, ISA validation, and native-codegen ambition into one unbounded task.

## Current Checkpoint

Known verdicts:

- `POST_DEFAULT_AUDIT_COMPLETE`
- `RUNTIME_KV_CORE_RUNTIME_BLOCKED_SMALL_OPS_NEXT`
- `RUNTIME_KV_NOT_ISA_BLOCKED`
- `ISA_AUDIT_GENERAL_PRINCIPLE_CONFIRMED`
- `AMD_ISA_AUDIT_READY`
- `MACHINE_SEARCH_NOT_READY`
- `ATTENTION_CLOSED_MAINTENANCE_ONLY`
- `GEMV_CLOSED_MAINTENANCE_ONLY`

Known facts:

- tinygrad is `~88-89%` of llama on Qwen3-8B-Q4_K_M decode.
- owned AMDGCN attention is default-on for the validated shape.
- Q4K GEMV warp is at/near llama parity.
- attention and FFN GEMV are closed for 8B.
- MAXC shrink proves the KV materialization tax transfers:
  - `+11.8%` at MAXC 1536;
  - `+12.9%` at MAXC 1280;
  - reaches llama-parity-class tok/s.
- `E_49152` full-cache materialization is on the W==D critical path.
- opaque append passes standalone but fails model-local persistence.
- the blocker is `RUNTIME_GRAPH_LIFECYCLE_GAP`, not ISA/codegen.
- AMD ISA audit tooling works and confirms the owned tile (`v_dot2`, LDS, cross-lane, 56 VGPR, 0 spill).

## Required Reading

Read these first:

1. `docs/post-exhaustion-remaining-lanes-roadmap-result-20260623.md`
2. `docs/post-exhaustion-remaining-lanes-roadmap-scope-20260623.md`
3. `docs/post-default-runtime-kv-diagnostic-result-20260623.md`
4. `docs/runtime-kv-core-runtime-blocker-result-20260623.md`
5. `docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md`
6. `docs/cross-vendor-isa-primitive-audit-and-search-scope-20260623.md`
7. `docs/amd-gpu-holistic-primitive-model-20260623.md`
8. `docs/owned-amdgcn-tile-short-ctx-result-20260623.md`
9. `docs/decode-ffn-gemv-warp-result-20260622.md`
10. `structure/Development/performance-primitive-research-principles.md`
11. `structure/Development/session-handoff.md`

Inspect:

- `tinygrad/llm/model.py`
- `tinygrad/engine/jit.py`
- `tinygrad/engine/realize.py`
- `tinygrad/runtime/graph/hcq.py`
- `tinygrad/runtime/support/hcq.py`
- `tinygrad/runtime/ops_amd.py`
- `extra/qk_amdgpu_isa_primitive_audit.py`
- `extra/qk_owned_flash_decode_graph_node.py`
- `extra/qk_owned_flash_decode.hip`
- prior Runtime-KV probes under `extra/qk_*kv*.py`
- `bench/qk-post-default-runtime-kv-course/`
- `bench/qk-isa-primitive-audit/`

## Global Boundaries

- Do not reopen attention/GEMV optimization.
- Do not start machine search.
- Do not do 14B/32B.
- Do not flip defaults.
- Do not implement production Runtime-KV unless explicitly authorized after a separate design review.
- Do not make source changes in this scope unless the owner explicitly asks to execute a chosen implementation.
- Do not use position-written proxies for correctness; token correctness is authority.
- Do not call an ISA audit result a W==D result.
- Do not call native codegen learning a performance win unless W==D proves it.

## Dependency Ordering

Recommended order:

```text
Lane 3 ISA audit infrastructure
  -> Lane 2 Runtime-KV core persistence design
  -> Lane 6 Native codegen learning charter
```

Reason:

- ISA audit is cheap and improves evidence quality for both other lanes.
- Runtime-KV is the only high-W==D remaining prize.
- Native codegen learning is valuable but should not distract from the core-runtime blocker unless it is tied to a concrete
  residual gap or portability goal.

## Lane 3 — ISA Audit Infrastructure

### Goal

Promote the AMD ISA audit from a one-off tool into a standing evidence contract for future candidates.

### Current Asset

```text
extra/qk_amdgpu_isa_primitive_audit.py
```

Current proven artifact:

```text
bench/qk-isa-primitive-audit/owned_decode_attention.json
```

### Scope

Build or scope a vendor-neutral wrapper with AMD backend active:

```text
extra/qk_isa_primitive_audit.py
```

Minimum acceptable implementation:

- accepts `--vendor amd`;
- accepts `--code-object`;
- accepts `--candidate`;
- optionally accepts `--wd-artifact`;
- calls the existing AMD parser/tool;
- emits normalized JSON;
- writes under `bench/qk-isa-primitive-audit/`;
- gracefully reports unsupported vendors as scoped/unavailable.

### Normalized Output Contract

Required fields:

```json
{
  "candidate": "...",
  "vendor": "amd",
  "arch": "...",
  "code_object": "...",
  "symbols": [],
  "resources": {
    "vgpr": null,
    "sgpr": null,
    "lds_bytes": null,
    "scratch_bytes": null,
    "spills": null
  },
  "instruction_flags": {
    "has_vector_dot": false,
    "has_lds": false,
    "has_cross_lane": false,
    "has_vector_global_load": false,
    "has_spill": false
  },
  "graph_lifecycle": {
    "route_fires": null,
    "fallback": null,
    "runtime_vars": []
  },
  "wd": {
    "artifact": null,
    "tokens_match": null,
    "delta_pct": null
  },
  "verdict": "..."
}
```

### Required Validation

Run the wrapper on the owned attention code object if discoverable, or on the same code object path used by the prior audit.

Expected:

- owned tile:
  - `has_vector_dot=true`;
  - `has_lds=true`;
  - `has_cross_lane=true`;
  - `has_spill=false`;
  - VGPR around prior 56 value if available.

### Verdicts

- `ISA_WRAPPER_AMD_ONLY_READY`
- `ISA_WRAPPER_PARTIAL`
- `ISA_WRAPPER_BLOCKED_TOOLING`

### Stop Rules

- Do not implement NVIDIA/Intel backends in this task.
- Do not block if code-object discovery is hard; allow explicit input path.
- Do not overbuild CI/search integration yet.

## Lane 2 — Runtime-KV Core Persistence

### Goal

Scope the core tinygrad runtime capability needed to remove full-MAXC materialization:

```text
persistent mutable decode state without full-cache .after() materialization across replay
```

This is the only remaining lane with parity-class W==D potential.

### What Is Already Proven

| fact | evidence |
|---|---|
| materialization is costly | `E_49152` ~1.5 ms/token |
| materialization transfers | MAXC shrink +11.8/+12.9% |
| opaque append can work locally | standalone microbench passes |
| owned tile is not the blocker | dtype fixed, ISA-confirmed, real-cache-correct |
| GraphRunner scalar patching is not the blocker | args verified previously |
| model route still fails | persistence/lifecycle bakes from decode step 1 |

### Scope Type

This lane needs a **design scope**, not an implementation patch.

Write:

```text
docs/runtime-kv-core-persistence-capability-scope-20260623.md
```

### Required Design Questions

| question | required answer |
|---|---|
| What object owns mutable KV state? | Tensor buffer, runtime object, or explicit state object |
| How does state persist across TinyJit replay? | replay contract |
| How are append writes ordered before attention reads? | dependency primitive |
| How is full-cache `.after()` avoided? | no materialization dependency |
| How is aliasing represented? | state token, bounded alias rule, or two-graph split |
| How is `start_pos` handled? | runtime var, not baked symbolic index |
| How does fallback work? | default-safe |
| What is the smallest proof? | one-layer or mini-model before full decode |

### Candidate Designs To Evaluate

#### Design A — Runtime-Managed KV Object

Mutable cache state is represented outside pure Tensor functional graph and passed to decode as a runtime-managed object.

Pros:

- semantically honest;
- avoids pretending mutable cache is pure Tensor dataflow.

Risks:

- new runtime/API surface;
- may not compose with existing TinyJit assumptions.

#### Design B — State Token Dependency Primitive

Append returns a lightweight state token; attention read depends on token but does not materialize full cache.

Pros:

- targeted to dependency ordering;
- avoids full alias analysis if bounded.

Risks:

- core scheduler/runtime semantics;
- must prove no reordering/corruption.

#### Design C — Bounded KV Alias Rule

Special-case append/read ranges for KV cache and allow symbolic prefix read without full buffer materialization.

Pros:

- possibly narrow for decode.

Risks:

- symbolic alias analysis can become broad/unbounded;
- prior work hit similar walls.

#### Design D — Two-Graph Decode Split

Separate append graph and attention/read graph with explicit runtime state boundary.

Pros:

- avoids same pure graph read-after-write conflict;
- closer to runtime-managed cache systems.

Risks:

- extra launch/graph boundary;
- API and lifecycle complexity.

### Required Minimal Proof Ladder

1. **Toy buffer proof**
   - append scalar/vector at runtime index;
   - read prefix;
   - replay changing `start_pos`;
   - no full materialization.

2. **One-layer KV proof**
   - append K/V for one layer;
   - owned tile reads;
   - token/logit proxy or numeric reference;
   - multi-step persistence.

3. **Full-model shadow proof**
   - route disabled by default;
   - compare tokens to baseline;
   - verify `E_49152` absent/reduced;
   - ctx1024 first.

4. **W==D proof**
   - ctx512/1024/2048/4096;
   - no regression;
   - expected >=5%, likely parity-class if full copy removed.

### Required Result If Scoped

The design scope must end with one of:

- `RUNTIME_KV_CORE_CAPABILITY_SCOPE_READY_DESIGN_A`
- `RUNTIME_KV_CORE_CAPABILITY_SCOPE_READY_DESIGN_B`
- `RUNTIME_KV_CORE_CAPABILITY_SCOPE_READY_DESIGN_C`
- `RUNTIME_KV_CORE_CAPABILITY_SCOPE_READY_DESIGN_D`
- `RUNTIME_KV_CORE_CAPABILITY_TOO_BROAD_DEFER`

### Stop Rules

- If the design requires general symbolic alias analysis, stop and mark too broad.
- If the design requires rewriting all TinyJit buffer semantics, stop and mark too broad.
- If no minimal toy proof can be described, do not implement.
- If it cannot preserve token correctness, stop.

## Lane 6 — Native tinygrad Codegen Learning

### Goal

Scope how tinygrad-native codegen could learn from the proven escape-hatch/native schedules:

- owned AMDGCN attention tile;
- Q4K GEMV warp schedule.

This is a learning/codegen capability lane, not an immediate speed lane.

### Why It Exists

The project now has two strong primitive exemplars:

| exemplar | implementation style | lesson |
|---|---|---|
| owned attention tile | hand-written HIP/AMDGPU code object | tinygrad native attention codegen does not yet emit llama-class LDS/vector-dot/split-KV tile |
| Q4K GEMV warp | tinygrad-native UOp schedule | work decomposition can be represented natively and transfer |

Native codegen learning asks:

```text
which parts of the escape-hatch attention primitive should become expressible in tinygrad's scheduler/renderer?
```

### Scope Type

Write a charter, not implementation:

```text
docs/native-codegen-learning-from-owned-primitives-scope-20260623.md
```

### Required Sections

1. Current closed primitives:
   - attention;
   - GEMV.
2. What tinygrad-native already learned:
   - Q4K GEMV warp schedule.
3. What remains escape-hatch-only:
   - owned attention tile;
   - split-KV policy;
   - LDS staging;
   - `v_dot2`;
   - cross-lane reduction;
   - graph-node binary injection.
4. Candidate compiler capabilities:
   - first-class wave reductions;
   - explicit LDS tiling templates;
   - vector-dot lowering controls;
   - split-KV decode attention schedule;
   - code-object/ISA feedback loop;
   - dtype/cache ABI propagation.
5. Non-goals:
   - no immediate W==D target unless fresh residual gap exists;
   - no replacing working owned route prematurely;
   - no broad renderer rewrite.
6. First bounded learning experiment:
   - reproduce one owned-tile ISA property in a tinygrad-native kernel;
   - local correctness;
   - ISA audit;
   - no requirement to beat default owned route.

### Candidate First Experiments

| experiment | goal | risk |
|---|---|---|
| tinygrad-native `v_dot2` microkernel | prove controlled dot lowering | narrow, useful |
| tinygrad-native LDS + cross-lane reduction microkernel | prove workgroup/wave primitive | medium |
| tinygrad-native split-KV toy attention | prove dataflow shape | higher |
| renderer feedback from ISA audit | automate missing-primitive detection | infrastructure-heavy |

Recommended first experiment:

```text
tinygrad-native LDS + cross-lane reduction microkernel with ISA audit
```

Reason:

- directly targets the gap between owned attention and native codegen;
- bounded;
- does not risk default decode;
- uses the new ISA audit guard.

### Verdicts

- `NATIVE_CODEGEN_LEARNING_CHARTER_READY`
- `NATIVE_CODEGEN_FIRST_EXPERIMENT_SCOPED`
- `NATIVE_CODEGEN_DEFER_NO_WD_NEED`

### Stop Rules

- Do not replace owned attention.
- Do not pursue full native flash attention until micro-primitives are proven.
- Do not claim performance win without W==D.
- Do not let this block Runtime-KV if runtime work is authorized.

## Combined Roadmap For 2/3/6

Recommended execution order:

1. Lane 3: vendor-neutral ISA wrapper with AMD backend only.
2. Lane 2: Runtime-KV core persistence capability scope.
3. Lane 6: native codegen learning charter.

Why:

- ISA wrapper improves evidence for all future work.
- Runtime-KV is the only parity-class speed prize.
- Native codegen learning is valuable but should be bounded and not confused with current W==D optimization.

## Required Final Result Doc

If Claude executes this scope, write:

```text
docs/runtime-kv-isa-native-codegen-three-lane-result-20260623.md
```

Required sections:

1. Verdict.
2. Authority / current repo state.
3. Lane 3 ISA infrastructure result.
4. Lane 2 Runtime-KV core persistence scope result.
5. Lane 6 native codegen learning scope result.
6. Recommended execution order.
7. What remains explicitly out of scope.
8. Files changed.
9. Git status.

Allowed final verdicts:

- `THREE_LANE_SCOPES_READY`
- `ISA_READY_RUNTIME_KV_SCOPE_READY_NATIVE_SCOPE_READY`
- `RUNTIME_KV_TOO_BROAD_NATIVE_DEFER`
- `THREE_LANE_SCOPE_INCOMPLETE`

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

The owner wants to pursue lanes 2/3/6 from the post-exhaustion roadmap:

2. Runtime-KV core persistence.
3. ISA audit infrastructure.
6. Native tinygrad codegen learning.

Read and execute:

```text
docs/runtime-kv-isa-native-codegen-three-lane-scope-20260623.md
```

Also read:

```text
docs/post-exhaustion-remaining-lanes-roadmap-result-20260623.md
docs/post-default-runtime-kv-diagnostic-result-20260623.md
docs/runtime-kv-core-runtime-blocker-result-20260623.md
docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md
docs/amd-gpu-holistic-primitive-model-20260623.md
```

Do not implement production Runtime-KV. Do not reopen attention/GEMV. Do not start machine search. Do not flip defaults.

Execute as a scoping/infrastructure task:

1. For Lane 3, either build a minimal vendor-neutral ISA wrapper with AMD backend only, or write why the existing AMD tool is sufficient for now.
2. For Lane 2, write a concrete Runtime-KV core persistence capability scope with design alternatives, minimal proof ladder, gates, and stop rules.
3. For Lane 6, write a native-codegen learning charter from owned attention + Q4K GEMV warp, including the first bounded micro-primitive experiment.
4. Write `docs/runtime-kv-isa-native-codegen-three-lane-result-20260623.md`.
5. Update README/session handoff if appropriate.

Final response must include:

- final verdict;
- Lane 3 ISA result;
- Lane 2 Runtime-KV scope result;
- Lane 6 native-codegen scope result;
- recommended execution order;
- files changed;
- git status.
