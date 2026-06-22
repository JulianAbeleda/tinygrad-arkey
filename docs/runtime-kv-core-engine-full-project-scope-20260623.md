# Runtime-KV Core Engine Full Project Scope — Persistent In-Graph Decode State (2026-06-23)

## Mission

Scope the full Lane 2 project: remove the full-MAXC KV materialization tax by adding a core tinygrad engine capability for
persistent mutable decode state inside replayed TinyJit/HCQ execution.

This supersedes the earlier Design-A recommendation in
`docs/runtime-kv-core-persistence-capability-scope-20260623.md`.

The decisive update from the latest three-lane completion is:

```text
pre-graph append is fundamentally insufficient for transformer decode
```

because for each layer:

```text
layer input hidden
-> compute q/k/v for this layer
-> append this layer's k/v to that layer's cache
-> attention reads that same layer cache
-> output hidden feeds the next layer
```

K/V for layer N depend on the output of layer N-1, so all layer appends cannot be precomputed before the captured graph.
The required capability is therefore:

```text
in-graph cache store + later cache load, persistent across replay, without full-buffer materialization
```

This is a core engine/runtime project, not a model-route patch, not a hand-kernel project, and not an attention/GEMV
optimization.

## Current Evidence

Known verdicts:

- `RUNTIME_KV_CORE_CAPABILITY_BLOCKED`
- `RUNTIME_GRAPH_LIFECYCLE_GAP`
- `RUNTIME_KV_NOT_ISA_BLOCKED`
- `MAXC_SHRINK_TRANSFERS`
- `E49152_ON_CRITICAL_PATH`
- `OPAQUE_APPEND_RUNTIME_GRAPH_BLOCKED`

Measured value:

- full-MAXC materialization is about `~1.5 ms/token`;
- MAXC shrink transfers:
  - `+11.8%` at MAXC 1536;
  - `+12.9%` at MAXC 1280;
- eliminating it is parity-class, roughly `~86 tok/s -> ~96-97 tok/s` at ctx1024 on Qwen3-8B-Q4_K_M.

Refuted non-causes:

| suspected cause | status |
|---|---|
| owned tile dtype bug | fixed; not the current blocker |
| GraphRunner scalar arg patching | refuted; args advance |
| buffer identity / rebase | refuted |
| model multi-layer composition only | refuted; failure appears even at `NL=1` with real prefill |
| append kernel ISA | standalone passes; not ISA-blocked |
| pre-graph append | impossible for full transformer data dependency |

Current root cause:

```text
tinygrad's pure @function graph uses full-buffer materialization as the mechanism that both orders cache writes
and preserves mutated cache state across replay.
```

Removing that materialization removes cross-replay persistence or hits read-after-write hazards.

## Non-Goals

- No attention tile work.
- No GEMV work.
- No machine search.
- No 14B/32B.
- No default flip.
- No broad renderer/codegen work.
- No production route until toy + one-layer + full-model correctness gates pass.
- No position-written proxy as correctness. Token correctness is authority.

## Required Reading

Read these in order:

1. `docs/post-default-runtime-kv-diagnostic-result-20260623.md`
2. `docs/runtime-kv-core-runtime-blocker-result-20260623.md`
3. `docs/runtime-kv-core-persistence-capability-scope-20260623.md`
4. `docs/runtime-kv-isa-native-codegen-three-lane-result-20260623.md`
5. `docs/post-exhaustion-remaining-lanes-roadmap-result-20260623.md`
6. `docs/amd-gpu-holistic-primitive-model-20260623.md`
7. `docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md`
8. `structure/Development/performance-primitive-research-principles.md`
9. `structure/Development/session-handoff.md`

Inspect code:

- `tinygrad/engine/jit.py`
- `tinygrad/engine/realize.py`
- `tinygrad/runtime/graph/hcq.py`
- `tinygrad/runtime/support/hcq.py`
- `tinygrad/runtime/ops_amd.py`
- `tinygrad/ops.py`
- `tinygrad/tensor.py`
- `tinygrad/llm/model.py`
- prior Runtime-KV probes under `extra/qk_*kv*.py`
- `extra/qk_owned_flash_decode_graph_node.py`
- `extra/qk_isa_primitive_audit.py`

## Core Capability To Add

Name:

```text
Persistent in-graph mutable state for decode KV cache
```

Definition:

```text
A replayed graph can write a bounded slice of a persistent buffer and later read from that same persistent buffer,
with ordering guaranteed, without materializing the entire buffer as a pure Tensor value, and with the mutation
persisting across graph replays.
```

Minimum semantics:

1. A persistent mutable buffer exists across TinyJit replays.
2. An in-graph append/store writes `cache[layer, kv, head, start_pos, dim]`.
3. A later in-graph attention read sees that write.
4. The write persists into the next replay.
5. The dependency is explicit enough that HCQ/TinyJit does not reorder read before write.
6. The graph does not emit a full-MAXC materialization kernel like `E_49152`.
7. Unsupported cases fall back to the canonical materialized path.

## Candidate Engine Designs

### Design 1 — Mutable Buffer Op / State Object

Add a first-class runtime state object or mutable buffer op understood by TinyJit capture/replay.

Concept:

```text
KVState(buffer)
append = kv_state.store_slice(start_pos, k, v)
read = kv_state.read_prefix(start_pos, token=append.token)
```

Properties:

- state object owns the persistent buffer;
- append mutates state;
- read depends on append token;
- state persists across replays;
- no full Tensor value materialization.

Pros:

- semantically honest;
- aligns with decode cache reality;
- can be guarded to KV cache only.

Risks:

- new runtime-visible abstraction;
- needs scheduler/JIT/graph integration;
- must avoid infecting general Tensor purity.

### Design 2 — Effect Token / Ordering Token

Add an explicit side-effect token in the UOp/schedule/graph path.

Concept:

```text
token1 = cache_append(cache_buffer, start_pos, k, v, token0)
out = attention_read(cache_buffer, start_pos, token1)
```

The token orders operations and carries persistence metadata, but does not represent the full buffer.

Pros:

- narrow dependency primitive;
- avoids full alias analysis if restricted.

Risks:

- token must survive TinyJit replay;
- every graph backend must honor token ordering;
- persistence still needs explicit buffer ownership.

### Design 3 — HCQ Graph External Mutable Resource

Treat KV cache as an external mutable resource attached to the graph runner, with per-replay side-effect commands inside
the command stream.

Concept:

```text
GraphRunner owns/receives a stable cache buffer handle.
Captured graph includes commands that write/read that buffer.
Graph replay patches start_pos and preserves the buffer handle.
```

Pros:

- close to actual runtime behavior;
- may avoid large UOp semantic changes.

Risks:

- backend-specific if not abstracted;
- must integrate with TinyJit schedule enough to preserve dependencies;
- may be hard to keep portable.

### Design 4 — Bounded KV Alias Rule

Special-case the cache append/read pattern:

```text
write exactly [start_pos]
read prefix [0:start_pos+1]
```

and permit it without full-buffer materialization.

Pros:

- narrow to decode.

Risks:

- prior attempts hit symbolic alias/read-after-write walls;
- can become general alias analysis quickly.

Default stance:

```text
Design 4 is allowed only if bounded to the exact KV append/read pattern and rejected if it expands.
```

### Rejected Design — Pre-Graph Append

Do not pursue as the primary design.

Reason:

```text
Full transformer decode requires per-layer append/read interleaving; layer N K/V cannot be computed before layer N-1
attention has produced the next hidden state.
```

Pre-graph append can remain useful only for toy probes or non-transformer state patterns.

## Required Implementation Phases

### Phase 0 — Authority + Safety Lock

Record:

- HEAD;
- git status;
- GPU/arch;
- default route state;
- current W==D baseline;
- current `E_49152` materialization evidence;
- fallback path.

Artifact:

- `bench/qk-runtime-kv-core-engine/authority.json`

Stop if baseline decode is not byte-identical.

### Phase 1 — Engine Semantics Design Doc

Write a design doc before code:

- `docs/runtime-kv-core-engine-design-20260623.md`

Required sections:

1. chosen design;
2. rejected designs;
3. exact semantics;
4. UOp/scheduler/JIT changes;
5. HCQ graph changes;
6. fallback;
7. test ladder;
8. expected risks;
9. rollback plan.

Verdicts:

- `RUNTIME_KV_ENGINE_DESIGN_READY`
- `RUNTIME_KV_ENGINE_DESIGN_TOO_BROAD_STOP`

### Phase 2 — Toy Mutable Buffer Proof

Goal:

Prove persistent mutable state across replay without model complexity.

Toy program:

```text
persistent buffer B[MAXC, D]
for each replay start_pos:
  write vector x to B[start_pos]
  read/sum B[0:start_pos+1]
```

Requirements:

- same captured graph replays with changing `start_pos`;
- buffer mutation persists across replays;
- read sees current replay's write;
- no full-buffer materialization kernel;
- token/numeric correctness;
- CPU/numpy reference.

Artifacts:

- `extra/qk_runtime_kv_core_toy.py`
- `bench/qk-runtime-kv-core-engine/toy.json`

Verdicts:

- `TOY_MUTABLE_REPLAY_PASS`
- `TOY_MUTABLE_REPLAY_PERSISTENCE_FAIL`
- `TOY_MUTABLE_REPLAY_ORDERING_FAIL`
- `TOY_MUTABLE_REPLAY_MATERIALIZES_FAIL`

Stop unless `TOY_MUTABLE_REPLAY_PASS`.

### Phase 3 — One-Layer Transformer Proof

Goal:

Prove the actual per-layer ordering:

```text
q/k/v computed in graph -> k/v appended -> owned attention reads same cache -> output correct
```

Requirements:

- one block/layer only;
- real q/k/v producer path;
- owned attention read;
- fp16 cache;
- multi-step decode;
- token/logit or numeric reference vs materialized baseline;
- no `E_49152`;
- finite K/V;
- no gqa/owned fallback confusion.

Artifacts:

- `extra/qk_runtime_kv_core_one_layer.py`
- `bench/qk-runtime-kv-core-engine/one_layer.json`

Verdicts:

- `ONE_LAYER_RUNTIME_KV_PASS`
- `ONE_LAYER_RUNTIME_KV_CORRECTNESS_FAIL`
- `ONE_LAYER_RUNTIME_KV_PERSISTENCE_FAIL`
- `ONE_LAYER_RUNTIME_KV_MATERIALIZES_FAIL`

Stop unless `ONE_LAYER_RUNTIME_KV_PASS`.

### Phase 4 — Full-Model Shadow Route

Goal:

Wire the capability behind an explicit env flag, default-off, for Qwen3-8B validated shape.

Required flag:

```text
RUNTIME_KV_CORE=1
```

Requirements:

- default path unchanged when flag off;
- route guarded to validated shape/device;
- fallback on exception;
- token correctness on at least two prompts;
- ctx1024 first;
- multi-step decode;
- no `E_49152` or materialization reduced by expected amount;
- route fire evidence.

Artifacts:

- `bench/qk-runtime-kv-core-engine/full_model_shadow.json`

Verdicts:

- `FULL_MODEL_RUNTIME_KV_SHADOW_PASS`
- `FULL_MODEL_RUNTIME_KV_TOKEN_FAIL`
- `FULL_MODEL_RUNTIME_KV_MATERIALIZATION_REMAINS`
- `FULL_MODEL_RUNTIME_KV_FALLBACK_ONLY`

Stop unless `FULL_MODEL_RUNTIME_KV_SHADOW_PASS`.

### Phase 5 — W==D Measurement

Goal:

Measure transfer.

Required contexts:

- 512;
- 1024;
- 2048;
- 4096.

Required measurements:

- baseline default;
- `RUNTIME_KV_CORE=1`;
- owned attention route confirmed;
- Q4K warp state explicit;
- tok/s;
- ms/token;
- token correctness;
- spread;
- `E_49152` removed/reduced;
- no ctx512 regression.

Artifacts:

- `bench/qk-runtime-kv-core-engine/wd.json`

Promotion gates:

- `>= +5%` at ctx1024;
- no ctx512 regression;
- byte-identical tokens;
- no silent fallback.

Verdicts:

- `RUNTIME_KV_CORE_WD_PASS`
- `RUNTIME_KV_CORE_WD_NO_TRANSFER`
- `RUNTIME_KV_CORE_WD_REGRESSION`
- `RUNTIME_KV_CORE_WD_CORRECTNESS_FAIL`

### Phase 6 — Engine Hardening

Only if W==D passes.

Required:

- focused unit tests for toy mutable replay;
- one-layer regression;
- default-off full-model route test/harness;
- unsupported shape fallback;
- graph backend safety;
- docs for semantics.

Artifacts:

- `bench/qk-runtime-kv-core-engine/hardening.json`

Verdicts:

- `RUNTIME_KV_CORE_HARDENED_DEFAULT_OFF`
- `RUNTIME_KV_CORE_HARDENING_BLOCKED`

### Phase 7 — Default Decision

Do not flip default automatically.

If hardening passes, write:

- `docs/runtime-kv-core-engine-default-decision-20260623.md`

Decision states:

- `OWNER_DEFAULT_READY`
- `KEEP_DEFAULT_OFF_RESEARCH`
- `DO_NOT_PROMOTE`

## Required Result Doc

Write:

- `docs/runtime-kv-core-engine-result-20260623.md`

Sections:

1. Verdict.
2. Design chosen.
3. Engine semantics.
4. Toy proof.
5. One-layer proof.
6. Full-model shadow route.
7. W==D result.
8. Materialization removal evidence.
9. Correctness.
10. Hardening/default decision.
11. Files changed.
12. Git status.

## Required Artifact Directory

```text
bench/qk-runtime-kv-core-engine/
```

Required artifacts:

- `authority.json`
- `toy.json`
- `one_layer.json`
- `full_model_shadow.json`
- `wd.json`
- `hardening.json`

## Scope Boundaries

Allowed source areas:

- TinyJit capture/replay;
- schedule dependency semantics;
- HCQ graph dependency/execution ordering;
- model route only behind explicit env flag;
- tests/probes.

Disallowed unless separately authorized:

- attention/GEMV optimization;
- machine search;
- 14B/32B;
- default flip;
- broad unrelated renderer rewrite;
- broad general symbolic alias analysis.

## Stop Rules

Stop and classify if:

- toy proof cannot pass;
- solution requires unrestricted alias analysis;
- solution requires rewriting the entire Tensor purity model;
- token correctness fails;
- full materialization remains;
- W==D does not transfer;
- implementation cannot be guarded/fallback-safe.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

The owner authorizes pursuing Lane 2 fully: Runtime-KV core persistence.

Read and execute:

```text
docs/runtime-kv-core-engine-full-project-scope-20260623.md
docs/post-default-runtime-kv-diagnostic-result-20260623.md
docs/runtime-kv-core-runtime-blocker-result-20260623.md
docs/runtime-kv-isa-native-codegen-three-lane-result-20260623.md
docs/amd-gpu-holistic-primitive-model-20260623.md
```

Important update:

The older pre-graph append design is superseded. It is insufficient for transformer decode because layer K/V are produced
inside the layer sequence. The required capability is in-graph persistent mutable KV state:

```text
q/k/v computed in captured graph -> append K/V to persistent cache -> attention reads cache -> mutation persists across replay
```

Execute phases in order:

1. Authority + baseline lock.
2. Write `docs/runtime-kv-core-engine-design-20260623.md`.
3. Implement/probe toy mutable buffer replay.
4. Implement/probe one-layer transformer KV replay.
5. Implement default-off full-model shadow route only if prior gates pass.
6. Run W==D only if correctness and materialization-removal gates pass.
7. Harden only if W==D passes.
8. Write result doc and artifacts.

Hard boundaries:

- do not reopen attention/GEMV;
- do not machine-search;
- do not do 14B/32B;
- do not flip defaults;
- do not use position-written proxy as correctness;
- token correctness is authority;
- stop if this becomes broad symbolic alias analysis or whole Tensor purity rewrite.

Final response must include:

- final verdict;
- design chosen;
- toy proof result;
- one-layer result;
- full-model result if reached;
- W==D result if reached;
- whether `E_49152` was removed;
- files changed;
- git status.
