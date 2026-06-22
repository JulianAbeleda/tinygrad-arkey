# tinygrad Core Runtime-KV Capabilities — External Patterns + Full Capability Scope (2026-06-23)

## Mission

Scope the tinygrad-core capabilities needed to remove full-MAXC KV cache materialization in decode, using both the
project's measurements and external runtime patterns.

The required project is not a model-route optimization. It is a core runtime/graph capability:

```text
persistent mutable decode state inside replayed graph execution,
with explicit ordering,
without representing the whole KV cache as a newly materialized pure Tensor value.
```

This document is the capability map for implementing Lane 2 fully.

## External Pattern Check

The online evidence aligns with our diagnosis.

### CUDA Graph / graph replay pattern

CUDA Graph-style replay assumes stable execution structure and stable memory addresses. NVIDIA's CUDA Graph best-practice
docs state that CUDA Graphs traditionally require the same operation sequence and memory addresses across replay, and
PyTorch CUDA Graph material warns that captured graphs act on the same virtual addresses and must keep those addresses
valid across replays.

Relevance to tinygrad:

```text
TinyJit/HCQ replay likewise needs stable buffers/addresses for captured programs.
The KV cache should be a stable runtime-owned buffer, not a newly materialized Tensor value each replay.
```

### vLLM pattern

vLLM explicitly separates graph capture concerns from compilation and has graph paths for decode/prefill mixtures. Its
serving architecture treats KV cache memory as persistent runtime state, with CUDA Graph replay operating over stable
buffers and runtime metadata.

Relevance to tinygrad:

```text
Runtime-managed KV is the right conceptual model, but transformer decode requires append/read interleaving per layer.
Therefore tinygrad cannot simply do all appends before graph replay; it needs in-graph state mutation semantics.
```

### TensorRT-LLM pattern

TensorRT-LLM documents KV cache as a runtime system feature: it stores previous K/V pairs for reuse, supports paged KV
cache, reuse across requests, offloading, and variable attention windows. Its memory docs separately discuss runtime/decoder
buffers and KV cache tensors.

Relevance to tinygrad:

```text
KV cache is a runtime-managed inference resource, not ordinary temporary tensor dataflow.
```

### vAttention / virtual-memory pattern

vAttention argues that KV-cache management can be separated from attention kernels by using contiguous virtual memory and
on-demand physical allocation.

Relevance to tinygrad:

```text
The cache memory model and attention kernel can be decoupled. Our owned attention tile can remain unchanged if tinygrad
can provide a persistent, correctly ordered cache buffer.
```

## Sources

- vLLM CUDA Graphs design: `https://docs.vllm.ai/en/latest/design/cuda_graphs/`
- TensorRT-LLM KV Cache System: `https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html`
- TensorRT-LLM memory usage / runtime buffers: `https://nvidia.github.io/TensorRT-LLM/reference/memory.html`
- NVIDIA CUDA Graph best-practice dynamic patterns: `https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/handling-dynamic-patterns.html`
- NVIDIA CUDA Graph best-practice memory issues: `https://docs.nvidia.com/dl-cuda-graph/troubleshooting/memory-issues.html`
- PyTorch CUDA Graph trees discussion: `https://dev-discuss.pytorch.org/t/understanding-cudagraph-trees/1967`
- vAttention paper: `https://arxiv.org/html/2405.04437v2`

## Project Evidence

Measured facts:

| fact | result |
|---|---|
| full-MAXC materialization cost | about `~1.5 ms/token` |
| MAXC shrink transfer | `+11.8%` at MAXC 1536, `+12.9%` at MAXC 1280 |
| projected impact | parity-class, roughly `~86 -> ~96-97 tok/s` at ctx1024 |
| attention | closed / near llama parity |
| FFN GEMV | closed / llama parity |
| append kernel | standalone passes; not ISA-blocked |
| owned tile | fixed, default-on, real-cache-correct, ISA-confirmed |
| current blocker | `RUNTIME_GRAPH_LIFECYCLE_GAP` |

Core finding:

```text
cache_kv.after(store) currently provides both ordering and replay persistence by materializing the full cache.
```

We need to split those roles:

| role | current mechanism | desired mechanism |
|---|---|---|
| order append before attention read | full Tensor materialization via `.after()` | explicit effect/dependency token |
| persist mutated cache across replay | materialized Tensor value carried by `@function` | runtime-owned persistent buffer |
| attention reads updated cache | pure Tensor dependency | stable mutable resource read with ordering |

## Required tinygrad Core Capabilities

### Capability 1 — Runtime-Owned Persistent Buffer

tinygrad needs a way to mark a realized buffer as a persistent mutable runtime resource.

Requirements:

- stable device allocation across TinyJit capture/replay;
- not treated as a newly produced pure Tensor value every step;
- survives replay without `after(full_buffer_store)` materialization;
- address stability visible to HCQ/GraphRunner;
- explicit lifetime ownership;
- fallback-safe.

Possible API shapes:

```python
kv = RuntimeStateBuffer(shape, dtype, device)
```

or:

```python
kv = Tensor.empty(...).realize().as_runtime_state()
```

Do not commit to API until toy proof.

### Capability 2 — In-Graph Stateful Store

The graph must express:

```text
store K/V slice into persistent KV buffer at runtime start_pos
```

without creating a full Tensor result.

Requirements:

- runtime `start_pos` patching;
- bounded slice write;
- dtype/layout contract;
- backend sees it as side-effecting op;
- cannot be dead-code-eliminated;
- persists after replay.

Candidate representation:

```text
Ops.STATE_STORE or Ops.BUFFER_STORE with effect token
```

### Capability 3 — Ordered Stateful Read

Attention must read from the same persistent buffer after the store.

Requirements:

- read depends on current replay's store;
- read sees prior replays' stores;
- no full materialization;
- compatible with owned AMDGCN attention graph node;
- works with symbolic/runtime `start_pos`.

Candidate representation:

```text
read = state_read(kv, start_pos, token=store_token)
```

### Capability 4 — Effect / Dependency Token

tinygrad needs a non-data dependency primitive for side effects.

Requirements:

- orders stateful ops;
- captured by TinyJit;
- replay-safe;
- represented in schedule/ExecItem/HCQ graph;
- no fake full-buffer data edge;
- clear interaction with existing `.after()` semantics.

This is the likely central primitive.

### Capability 5 — Alias Scope: Bounded KV Pattern Only

Avoid general symbolic alias analysis.

Allowed alias model:

```text
write: cache[layer, kv, head, start_pos, dim]
read:  cache[layer, kv, head, 0:start_pos+1, dim]
```

This is a bounded decode-specific rule.

Stop if implementation expands into:

- arbitrary overlapping Tensor views;
- general symbolic range aliasing;
- global mutation semantics for all Tensor ops.

### Capability 6 — HCQ / GraphRunner Mutable Resource Support

HCQ graph execution must preserve:

- stable buffer handles;
- side-effecting store command;
- ordered read command;
- runtime var patching;
- replay persistence;
- no graph-internal buffer replacement.

Questions to answer:

| question | target answer |
|---|---|
| Where are runtime vars patched? | existing mechanism if possible |
| How does graph record mutable resource? | buffer handle + state metadata |
| How does graph avoid stale snapshot? | do not snapshot/cache full contents |
| How does replay guarantee order? | token/dependency edge or command ordering |

### Capability 7 — Model-Level Opt-In Route

Only after toy and one-layer proofs.

Requirements:

- explicit env flag, e.g. `RUNTIME_KV_CORE=1`;
- validated shape guard;
- default unchanged;
- fallback to canonical materialized path;
- route fire evidence;
- token-correct multi-step decode.

## Candidate Architecture

Recommended architecture to prototype:

```text
RuntimeStateBuffer
  + StateStore op
  + StateRead / opaque attention read dependency
  + EffectToken ordering
  + HCQ graph support for mutable resource handles
```

High-level flow per layer:

```text
hidden_l
  -> q,k,v projections
  -> state_store(kv_cache_l, start_pos, k, v) -> token_l
  -> owned_attention(q, kv_cache_l, start_pos, token_l)
  -> hidden_{l+1}
```

This preserves transformer data dependency and avoids the invalid pre-graph append assumption.

## Minimal Proof Ladder

### P0 — Design Lock

Write:

- `docs/runtime-kv-core-engine-design-20260623.md`

Must specify:

- chosen API/representation;
- UOp/schedule semantics;
- HCQ graph semantics;
- fallback;
- failure modes;
- files expected to change.

### P1 — Toy Mutable Replay

Program:

```text
state buffer B[MAXC,D]
captured graph:
  store x into B[start_pos]
  read/sum prefix B[0:start_pos+1]
replay start_pos = 0,1,2,...
```

Pass criteria:

- numeric correctness;
- persistence across replay;
- current write visible to current read;
- no full buffer materialization;
- no position proxy only.

### P2 — One-Layer Transformer Proof

Program:

```text
q,k,v generated in graph
state_store k/v
owned attention reads persistent cache
compare to materialized baseline
```

Pass criteria:

- multi-step correctness;
- finite K/V;
- owned route fires;
- no `E_49152`.

### P3 — Full-Model Shadow Route

Flag:

```text
RUNTIME_KV_CORE=1
```

Pass criteria:

- Qwen3-8B validated shape only;
- default unchanged when flag off;
- two prompts byte-identical;
- ctx1024 first;
- `E_49152` absent/reduced.

### P4 — W==D

Contexts:

- 512;
- 1024;
- 2048;
- 4096.

Promotion criteria:

- >= `+5%` at ctx1024;
- no ctx512 regression;
- byte-identical;
- no silent fallback;
- materialization removed/reduced.

Expected if successful:

- about `+11%`;
- llama-parity class.

### P5 — Hardening

Required tests:

- toy unit;
- one-layer regression;
- full-model default-off probe;
- fallback test;
- unsupported shape test;
- graph replay var patch test;
- no-materialization assertion.

## Stop Rules

Stop and classify if:

- requires general alias analysis;
- requires rewriting the whole Tensor purity model;
- cannot represent effect ordering in schedule/graph;
- toy proof fails;
- one-layer proof fails;
- token correctness fails;
- `E_49152` remains unchanged;
- W==D does not transfer;
- fallback cannot be made safe.

## Expected File Areas

Likely source areas:

- `tinygrad/ops.py`
- `tinygrad/tensor.py`
- `tinygrad/engine/jit.py`
- `tinygrad/engine/realize.py`
- `tinygrad/runtime/graph/hcq.py`
- `tinygrad/runtime/support/hcq.py`
- `tinygrad/runtime/ops_amd.py`
- `tinygrad/llm/model.py` behind env flag only
- `extra/qk_runtime_kv_core_*.py` probes/tests

Do not modify unrelated code.

## Required Artifacts

Directory:

```text
bench/qk-runtime-kv-core-engine/
```

Artifacts:

- `authority.json`
- `design.json`
- `toy.json`
- `one_layer.json`
- `full_model_shadow.json`
- `wd.json`
- `hardening.json`
- `decision.json`

Docs:

- `docs/runtime-kv-core-engine-design-20260623.md`
- `docs/runtime-kv-core-engine-result-20260623.md`

## Verdicts

Allowed final verdicts:

- `RUNTIME_KV_CORE_ENGINE_WD_PASS`
- `RUNTIME_KV_CORE_ENGINE_TOY_BLOCKED`
- `RUNTIME_KV_CORE_ENGINE_ONE_LAYER_BLOCKED`
- `RUNTIME_KV_CORE_ENGINE_FULL_MODEL_BLOCKED`
- `RUNTIME_KV_CORE_ENGINE_TOO_BROAD_ALIAS_ANALYSIS`
- `RUNTIME_KV_CORE_ENGINE_NO_WD_TRANSFER`
- `RUNTIME_KV_CORE_ENGINE_CORRECTNESS_FAIL`

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

The owner wants to pursue Runtime-KV core persistence fully.

Before coding, read:

```text
docs/tinygrad-core-runtime-kv-capabilities-scope-20260623.md
docs/runtime-kv-core-engine-full-project-scope-20260623.md
docs/post-default-runtime-kv-diagnostic-result-20260623.md
docs/runtime-kv-core-runtime-blocker-result-20260623.md
docs/runtime-kv-isa-native-codegen-three-lane-result-20260623.md
docs/amd-gpu-holistic-primitive-model-20260623.md
```

Important: the pre-graph append design is superseded. It cannot work for full transformer decode because per-layer K/V
are produced inside the layer sequence. The required capability is in-graph persistent mutable state:

```text
q/k/v computed in graph -> state_store K/V -> attention reads persistent state -> mutation persists across replay
```

Execute in phases and stop at the first hard blocker:

1. Authority lock and baseline.
2. Write `docs/runtime-kv-core-engine-design-20260623.md`.
3. Implement/probe toy mutable replay.
4. Implement/probe one-layer transformer replay.
5. Implement full-model shadow route only if previous gates pass.
6. Run W==D only if correctness and materialization-removal pass.
7. Harden only if W==D passes.
8. Write result doc and artifacts.

Hard boundaries:

- no attention/GEMV optimization;
- no machine search;
- no 14B/32B;
- no default flip;
- no broad symbolic alias analysis;
- no whole Tensor purity rewrite;
- token correctness is authority.

Final response must include:

- final verdict;
- design chosen;
- source files changed;
- toy proof result;
- one-layer result;
- full-model result if reached;
- W==D result if reached;
- whether `E_49152` was removed;
- files/artifacts/docs written;
- git status.
