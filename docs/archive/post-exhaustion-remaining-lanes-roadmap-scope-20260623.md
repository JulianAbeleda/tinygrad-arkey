# Post-Exhaustion Remaining Lanes Roadmap — Exhaustive Scope / Claude Prompt (2026-06-23)

## Mission

Consolidate every remaining lane after the 8B decode exhaustion checkpoint, and define exactly what should happen next.

This is **not** a request to implement all lanes immediately. It is a sequencing and scope document so future Claude/Codex
runs do not reopen closed work, conflate kernel work with core-runtime work, or start machine search before a bounded lane
exists.

Current checkpoint:

- `POST_DEFAULT_AUDIT_COMPLETE`
- `RUNTIME_KV_CORE_RUNTIME_BLOCKED_SMALL_OPS_NEXT`
- `ISA_AUDIT_GENERAL_PRINCIPLE_CONFIRMED`
- `AMD_ISA_AUDIT_READY`
- `MACHINE_SEARCH_NOT_READY`

Current performance state:

- owned AMDGCN attention is default-on for the validated 8B shape;
- Q4K GEMV warp is default-eligible and used in near-llama measurements;
- tinygrad is about `~88-89%` of llama on Qwen3-8B-Q4_K_M decode;
- attention and FFN GEMV are at/near llama parity;
- the remaining llama delta is explained:
  - KV materialization / `E_49152` copy: implement-worthy but core-runtime-blocked;
  - small-ops / activation-like residuals: bounded fallback but likely overlapped;
  - ISA audit: ready as guard infrastructure;
  - machine search: not ready.

## Required Reading

Read these in order:

1. `docs/post-owned-attention-default-audit-result-20260623.md`
2. `docs/post-default-runtime-kv-diagnostic-result-20260623.md`
3. `docs/runtime-kv-core-runtime-blocker-result-20260623.md`
4. `docs/small-ops-activation-fusion-scope-20260623.md`
5. `docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md`
6. `docs/cross-vendor-isa-primitive-audit-and-search-scope-20260623.md`
7. `docs/amd-gpu-holistic-primitive-model-20260623.md`
8. `docs/post-owned-attention-promotion-synthesis-20260623.md`
9. `structure/Development/performance-primitive-research-principles.md`
10. `structure/Development/session-handoff.md`

Inspect:

- `extra/qk_amdgpu_isa_primitive_audit.py`
- `extra/qk_decode_time_tax_audit.py`
- `extra/qk_tinygrad_vs_llama_time_tax.py`
- `extra/qk_decode_runtime_overhead.py`
- `tinygrad/llm/model.py`
- `bench/qk-post-owned-attention-default-audit/`
- `bench/qk-post-default-runtime-kv-course/`
- `bench/qk-isa-primitive-audit/`

## Global Rules

- Do not reopen attention unless a future audit proves a new residual attention gap after current default-on owned route.
- Do not reopen FFN GEMV unless a future audit proves a regression or cross-model gap.
- Do not start machine search until a lane exposes a bounded searchable knob.
- Do not implement Runtime-KV without explicit owner authorization for core-runtime work.
- Do not use position-written proxy as correctness; token correctness is authority.
- Do not trust bucket names without rendered-source / AST / trace evidence.
- Do not accept local-kernel wins without W==D transfer.
- Do not accept ISA/source claims without code-object/disassembly/resource evidence or an explicit tooling-limit note.

## Lane Summary

| priority | lane | status | next action |
|---:|---|---|---|
| 1 | Small-ops / activation fusion gate | bounded fallback, transfer unknown | run one narrow W==D-gated fusion experiment |
| 2 | Runtime-KV core persistence | biggest prize, core-runtime-blocked | scope only if owner authorizes core tinygrad runtime work |
| 3 | ISA audit infrastructure | ready, AMD implementation exists | make it mandatory guard; optionally add vendor-neutral wrapper |
| 4 | Machine search | not ready | defer until small-op fusion or Runtime-KV exposes knobs |
| 5 | Generalization / default hardening | strategically useful | after 8B lanes are closed, scope 14B/32B/cross-shape |
| 6 | Native codegen learning | long-term | use owned tile/GEMV as reference, not immediate W==D lane |
| 7 | Attention/GEMV maintenance | closed | regression guard only |

## Lane 1 — Small-Ops / Activation Fusion Gate

### Status

This is the immediate bounded fallback because Runtime-KV is core-runtime-blocked.

Known state:

- residual GPU-busy gaps include small ops / activation-like kernels;
- prior bucket labels were frequently wrong;
- some of this work is likely overlapped;
- therefore the first fusion must prove W==D transfer before expansion.

Existing scope:

- `docs/small-ops-activation-fusion-scope-20260623.md`

### Mission

Run exactly one bounded fusion gate:

```text
one confirmed kernel group -> one fusion/removal attempt -> token correctness -> ISA/graph evidence -> >=1-2% W==D
```

### Required First Gate

1. Pick one confirmed kernel group from corrected buckets.
2. Prove it is not mislabeled KV/cache materialization.
3. Show rendered source / AST / trace evidence.
4. Implement or route one minimal fusion/removal only.
5. Verify:
   - token correctness;
   - old kernel group removed or reduced;
   - no new spills/resource blow-up if code object exists;
   - W==D >= `1-2%` at ctx1024 or ctx4096;
   - no ctx512 regression.

### Stop Rules

- If one fusion does not reach >=1-2% W==D, close small-ops as overlapped/low-return.
- If the target is actually KV materialization, stop and redirect to Runtime-KV.
- If fusion requires broad codegen redesign, stop and scope it separately.
- If correctness fails, revert and classify.

### Required Artifacts

Suggested directory:

```text
bench/qk-small-ops-fusion-gate/
```

Artifacts:

- `authority.json`
- `candidate_group.json`
- `local_fusion_ab.json`
- `isa_or_graph_evidence.json`
- `wd.json`
- `decision.json`

### Verdicts

- `SMALL_OPS_FUSION_GATE_PASS`
- `SMALL_OPS_FUSION_GATE_NO_WD_TRANSFER`
- `SMALL_OPS_TARGET_MISBUCKETED_KV`
- `SMALL_OPS_FUSION_CORE_CODEGEN_BLOCKED`
- `SMALL_OPS_FUSION_CORRECTNESS_FAIL`

### If Passes

Only then scope:

- a second fusion;
- small machine search over fusion boundaries;
- or a native fusion/codegen lane.

### If Fails

Declare:

- `SMALL_OPS_OVERLAPPED_OR_LOW_RETURN`

and do not machine-search small ops.

## Lane 2 — Runtime-KV Core Persistence

### Status

This is the biggest remaining prize, but not a model/kern optimization.

Known state:

- MAXC shrink transfers strongly:
  - `+11.8%` at MAXC 1536;
  - `+12.9%` at MAXC 1280;
- `E_49152` is on the W==D critical path;
- opaque append passes standalone;
- model-local opaque append remains blocked;
- blocker is TinyJit / `@function` cross-replay persistence without full-cache materialization.

Existing result:

- `docs/runtime-kv-core-runtime-blocker-result-20260623.md`

### Mission

Only if owner explicitly authorizes core runtime work, scope a tinygrad runtime capability:

```text
persistent mutable decode state without full-MAXC .after() materialization
```

This is not:

- attention work;
- GEMV work;
- ISA work;
- a one-off append kernel.

### Required Core Capability Questions

| question | requirement |
|---|---|
| How does a mutable buffer persist across TinyJit replay? | explicit state contract |
| How are writes ordered before reads without materializing full buffer? | dependency primitive |
| How is aliasing represented? | bounded cache-slice alias rule or state token |
| How is symbolic `start_pos` represented? | runtime var, not baked index |
| How is correctness proven? | token-correct multi-step decode |
| How is fallback handled? | default-safe, no silent corruption |

### Possible Design Families

| design | description | risk |
|---|---|---|
| runtime-managed KV object | explicit mutable state outside pure Tensor graph | broad runtime/API work |
| state-token dependency primitive | ordering token for append/read without full materialization | core graph semantics |
| bounded KV alias rule | special-case cache slice append/read aliasing | symbolic alias complexity |
| two-graph decode split | append graph + read/attention graph with explicit runtime state | lifecycle/API complexity |

### Required Scope If Authorized

Write:

- `docs/runtime-kv-core-persistence-capability-scope-YYYYMMDD.md`

Must include:

1. semantics;
2. API surface;
3. scheduling/alias model;
4. graph replay behavior;
5. correctness gates;
6. W==D gates;
7. fallback/default policy;
8. minimal implementation slice;
9. stop rules.

### Verdicts

- `RUNTIME_KV_CORE_CAPABILITY_SCOPE_READY`
- `RUNTIME_KV_CORE_CAPABILITY_TOO_BROAD`
- `RUNTIME_KV_DEFERRED_OWNER_DECISION`

## Lane 3 — ISA Audit Infrastructure

### Status

AMD ISA audit is ready.

Known state:

- `extra/qk_amdgpu_isa_primitive_audit.py` exists;
- owned attention tile confirmed:
  - `v_dot2`;
  - LDS;
  - cross-lane;
  - 56 VGPR;
  - 0 scratch/spill;
- cross-vendor principle is confirmed;
- NVIDIA/Intel backends are scoped only.

Existing docs:

- `docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md`
- `docs/cross-vendor-isa-primitive-audit-and-search-scope-20260623.md`

### Mission

Use ISA audit as a mandatory guard for future candidate lanes.

Optional infra follow-on:

```text
extra/qk_isa_primitive_audit.py
```

as a vendor-neutral wrapper with AMD backend only for now.

### Required Behavior

For future candidates, require:

- code object path/hash;
- symbols;
- architecture;
- resource usage;
- instruction flags:
  - vector dot / tensor op;
  - shared memory/LDS;
  - cross-lane;
  - vector loads;
  - spills/scratch;
- graph lifecycle link;
- W==D artifact link.

### Verdicts

- `ISA_AUDIT_GUARD_ACTIVE`
- `ISA_WRAPPER_AMD_ONLY_READY`
- `ISA_BACKEND_TOOLING_LIMITED`

## Lane 4 — Machine Search

### Status

Not ready.

Reason:

- attention closed;
- GEMV closed;
- Runtime-KV blocked by core graph lifecycle, not tunable kernel knobs;
- small-ops fusion has not yet proven a transferable gate.

### Entry Conditions

Machine search becomes allowed only if one of these happens:

1. small-ops fusion gate passes;
2. Runtime-KV core capability lands and exposes tunable knobs;
3. a future audit identifies a residual kernel with:
   - local correctness harness;
   - ISA/codegen gap;
   - W==D plausibility.

### Required Search Loop

```text
candidate generation
-> static source/shape validation
-> build/lower
-> ISA audit
-> local correctness
-> local timing
-> graph route fire check
-> W==D token correctness
-> W==D timing
-> artifact archive
```

### Reject Rules

- no token correctness;
- route does not fire;
- required ISA primitive missing;
- spills/scratch introduced;
- local-only win with no W==D transfer;
- stale bucket label;
- broad random search with no bounded lane.

### Verdicts

- `MACHINE_SEARCH_NOT_READY`
- `MACHINE_SEARCH_READY_SMALL_OPS`
- `MACHINE_SEARCH_READY_RUNTIME_KV`
- `MACHINE_SEARCH_READY_RESIDUAL_KERNEL`

## Lane 5 — Generalization / Default Hardening

### Status

Strategic alternative after 8B exhaustion.

Potential scope:

- validate owned attention route on 14B/32B shapes;
- validate Q4K warp route on other dimensions;
- default-on hardening across more model configs;
- fallback/guard expansion;
- packaging/documentation.

Entry condition:

- owner chooses productization/generalization over more 8B speed.

Boundaries:

- no 14B/32B until explicitly requested;
- no assumed shape-general default;
- all new shapes require token correctness + W==D.

Verdicts:

- `GENERALIZATION_SCOPE_READY`
- `GENERALIZATION_DEFER_8B_FIRST`

## Lane 6 — Native tinygrad Codegen Learning

### Status

Long-term.

The escape-hatch wins reveal what tinygrad-native codegen could eventually learn:

- owned attention tile:
  - split-KV work decomposition;
  - LDS staging;
  - `v_dot2`;
  - cross-lane reductions;
  - native fp16 cache contract;
- Q4K GEMV warp:
  - wave/row decomposition;
  - K-block parallelism;
  - warp reduction.

Mission if funded:

```text
turn proven escape-hatch/native schedules into reusable tinygrad renderer/codegen capabilities
```

Boundary:

- not a short-term W==D lane unless tied to a specific residual gap.

Verdicts:

- `NATIVE_CODEGEN_LEARNING_SCOPE_READY`
- `NATIVE_CODEGEN_DEFER_NO_WD_NEED`

## Lane 7 — Closed Maintenance Lanes

### Attention

Status:

- closed / near parity / default-on / ISA-confirmed.

Allowed work:

- regression tests;
- fallback correctness;
- documentation;
- cross-shape validation only if generalization is authorized.

Disallowed:

- more tile variants;
- combine-only work;
- attention machine search.

Verdict:

- `ATTENTION_CLOSED_MAINTENANCE_ONLY`

### FFN GEMV

Status:

- closed / llama parity.

Allowed work:

- regression tests;
- default decision/hardening;
- cross-shape validation.

Disallowed:

- more schedule variants without fresh residual gap.

Verdict:

- `GEMV_CLOSED_MAINTENANCE_ONLY`

## Final Decision Matrix

| If... | Then... |
|---|---|
| small-ops fusion gate passes >=1-2% W==D | scope second fusion or small machine search |
| small-ops fusion gate fails | close small ops, no machine search |
| owner authorizes core runtime | scope Runtime-KV core persistence capability |
| owner does not authorize core runtime | keep Runtime-KV deferred |
| no bounded 8B lane remains | move to generalization/default hardening |
| future audit finds new residual kernel | require ISA audit + W==D gate before search |

## Required Roadmap Result Doc

If asked to execute this roadmap synthesis, write:

- `docs/post-exhaustion-remaining-lanes-roadmap-result-20260623.md`

Required sections:

1. Current checkpoint.
2. Lane table.
3. Immediate next action.
4. Runtime-KV core work decision.
5. Small-ops fusion gate.
6. ISA audit guard policy.
7. Machine-search readiness.
8. Generalization decision.
9. Closed lanes.
10. Final recommendation.
11. Files changed.
12. Git status.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Read:

```text
docs/post-exhaustion-remaining-lanes-roadmap-scope-20260623.md
docs/post-default-runtime-kv-diagnostic-result-20260623.md
docs/runtime-kv-core-runtime-blocker-result-20260623.md
docs/small-ops-activation-fusion-scope-20260623.md
docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md
docs/amd-gpu-holistic-primitive-model-20260623.md
```

Synthesize the complete remaining-lanes roadmap after the 8B exhaustion checkpoint.

Do not implement anything unless explicitly asked. This is a roadmap/scope consolidation task.

Required output:

1. Confirm closed lanes:
   - attention;
   - FFN GEMV.
2. Confirm active bounded fallback:
   - small-ops / activation fusion gate.
3. Confirm bigger but blocked lane:
   - Runtime-KV core persistence.
4. Confirm infrastructure lane:
   - ISA audit guard.
5. Confirm deferred lane:
   - machine search until a bounded knob exists.
6. Confirm strategic alternative:
   - generalization/default hardening.
7. Write:
   - `docs/post-exhaustion-remaining-lanes-roadmap-result-20260623.md`
8. Update:
   - `docs/README.md`
   - `structure/Development/session-handoff.md`

Hard boundaries:

- no attention/GEMV reopen;
- no machine search;
- no Runtime-KV implementation;
- no default flips;
- no 14B/32B;
- no source changes unless only docs/tooling references are needed.

Final response must include:

- final recommended next action;
- all lane statuses;
- whether any lanes are closed;
- whether machine search is allowed;
- files changed;
- git status.
