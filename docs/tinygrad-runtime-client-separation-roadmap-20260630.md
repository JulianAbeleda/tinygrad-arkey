# Tinygrad Runtime / Client Separation Roadmap

Date: 2026-06-30

Status: future implementation scope. This is a product/runtime architecture roadmap, not a kernel optimization phase.

## Goal

Make tinygrad usable as a local inference runtime that can be called by either:

```text
OpenCode or another OpenAI-compatible client
```

or:

```text
the proprietary TUI/CLI/app layer
```

without mixing agent UX, repo context, tools, or session memory into the tinygrad runtime.

Target architecture:

```text
TUI / CLI / OpenCode / proprietary app
  owns sessions, repo context, tools, prompt packing, permissions, UX

        |
        | JSON HTTP API
        | /v1/* for OpenAI-compatible clients
        | /runtime/* for proprietary lifecycle controls
        v

tinygrad runtime server
  owns model load, tokenizer, KV cache, prefill/decode, sampling, compiled kernels, GPU memory

        |
        v

GGUF / tokenizer metadata / compiled kernel cache / runtime artifacts
```

## Non-Goals

Do not turn tinygrad into an agent framework.

Out of scope for the runtime:

- repo indexing
- retrieval
- file edits
- tool execution
- shell permissions
- multi-turn chat memory policy
- summarization
- OpenCode-style TUI behavior
- proprietary app UX

Those belong above the runtime boundary.

## Source Citations

Read these before implementing:

| source | role |
|---|---|
| `tinygrad/llm/cli.py` | current simple OpenAI-compatible server and tokenizer path |
| `tinygrad/llm/model.py` | model load, max_context, KV cache, prefix reuse, generate loop |
| `docs/pure-machine-search-artifact-cache-scope-20260630.md` | cache classification model for generated/runtime artifacts |
| `docs/pure-machine-search-true-generation-agnostic-scope-20260630.md` | profile/target/quant separation model |
| `README.md` | current performance actuals and user-facing defaults |
| `https://opencode.ai/docs/providers/` | OpenCode provider model; local runtimes via OpenAI-compatible baseURL |
| `https://opencode.ai/docs/references/` | OpenCode-style external repo/reference layer, deliberately outside runtime |

Important current tinygrad facts:

- `tinygrad/llm/cli.py` already exposes `/v1/models` and `/v1/chat/completions`.
- `Transformer.from_gguf` caps `max_context` by the GGUF context length.
- `Transformer.generate` pads the input token list to `max_context`, uses `get_start_pos` for prefix-cache reuse, and stops when total tokens reach `max_context`.
- The runtime has no semantic memory, repo index, or context packer. It only sees tokens.

## Boundary Contract

### Runtime Owns

```text
model registry:
  available models, loaded model, aliases, local paths

model lifecycle:
  load, unload, warmup, status

tokenization:
  tokenizer from GGUF metadata

inference:
  chat/completion request -> tokens -> prefill/decode -> sampled tokens

KV cache:
  allocated per loaded model/context length

compiled kernel lifecycle:
  JIT warmup, compiled artifact reuse where safe

runtime metrics:
  loaded model, max_context, prompt tokens, completion tokens, prefill tok/s, decode tok/s, VRAM/KV estimates
```

### Client Owns

```text
session model:
  conversations, branches, summaries, old-turn eviction

repo context:
  local references, git refs, file search, embeddings/BM25, citations

prompt construction:
  system prompt, developer rules, XML-style context envelope, truncation policy

tool orchestration:
  shell, file edits, MCP, web, approvals

UX:
  TUI/CLI, OpenCode, proprietary app, logs, history browser
```

## API Format Decision

Use JSON for the API boundary.

Use XML-style tags only inside prompt text when the client packs context.

Correct split:

```text
HTTP payload:
  JSON

message.content:
  free text, optionally XML-tagged by the client
```

Example:

```json
{
  "model": "qwen3-8b-q4k",
  "messages": [
    {
      "role": "user",
      "content": "<repo path=\"/home/ubuntu/tinygrad-arkey\">\n<file path=\"tinygrad/llm/model.py\" start=\"1551\">\n...\n</file>\n</repo>\n<task>Explain context handling.</task>"
    }
  ],
  "temperature": 0,
  "max_tokens": 512
}
```

Do not make `/v1/*` XML. OpenCode, AI SDK clients, llama.cpp-compatible clients, and most local-provider tooling expect JSON.

## Endpoint Plan

### OpenAI-Compatible Surface

Keep this clean and portable:

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
```

Optional later:

```text
POST /v1/embeddings
```

Only add embeddings if tinygrad actually hosts an embedding model or a compatible local embedding runtime. Do not fake it.

### Tinygrad Runtime Control Surface

Keep this outside `/v1/*`:

```text
GET  /runtime/status
GET  /runtime/models
POST /runtime/load
POST /runtime/unload
POST /runtime/warmup
POST /runtime/cancel
GET  /runtime/metrics
GET  /runtime/cache
POST /runtime/cache/clear
```

These endpoints are for the proprietary app or local operator tooling. OpenCode does not need them.

## Runtime Data Model

### Model Registry Row

```json
{
  "id": "qwen3-8b-q4k",
  "path": "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf",
  "architecture": "qwen3",
  "quant": "Q4_K_M",
  "max_context_supported": 40960,
  "default_context": 4096,
  "backend": "AMD",
  "target": "gfx1100",
  "status": "available | loading | loaded | error",
  "tags": ["chat", "qwen", "q4_k"]
}
```

### Loaded Runtime Status

```json
{
  "loaded": true,
  "model": "qwen3-8b-q4k",
  "path": "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf",
  "max_context": 4096,
  "kv_cache_tokens": 4096,
  "cached_prefix_tokens": 0,
  "prefill_v2": true,
  "prefill_concrete_kv": true,
  "backend": "AMD",
  "target": "gfx1100",
  "last_prefill_tok_s": null,
  "last_decode_tok_s": null,
  "last_error": null
}
```

### Load Request

```json
{
  "model": "qwen3-8b-q4k",
  "path": "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf",
  "max_context": 4096,
  "backend": "AMD",
  "warmup": true,
  "profile": "interactive"
}
```

Rules:

- `model` may refer to a registry id.
- `path` may override registry path for proprietary app flows.
- `max_context` must be explicit or resolved from model default.
- runtime should reject `max_context > model_context_length`.
- runtime should expose whether load paid warmup/JIT cost.

## Context Semantics

The runtime does not own long-term memory.

Runtime behavior:

```text
request messages -> tokenizer -> token list
token list + cached prefix -> prefill suffix only
decode appends tokens until max_tokens, stop token, or max_context
```

Client behavior:

```text
session history + repo snippets + tool results + user turn
  -> context packer
  -> XML-style prompt content
  -> /v1/chat/completions
```

The client must enforce:

- history truncation
- summarization
- repo snippet selection
- tool-result compaction
- max prompt token budget

The runtime should still fail cleanly if prompt tokens exceed `max_context`.

Current risk to close:

```text
Transformer.generate currently builds:
Tensor(tokens + [0] * (self.max_context - len(tokens)))
```

So oversized prompts need an explicit runtime guard before model invocation.

## Cache Strategy

There are four separate caches. Do not conflate them.

| cache | owner | safe reuse rule |
|---|---|---|
| model file cache | runtime/operator | reuse by path + size + sha/hash metadata |
| compiled kernel cache | runtime | reuse by model/backend/code/runtime fingerprint |
| prompt/KV prefix cache | runtime | reuse only for shared token prefix in one loaded model session |
| repo/context index cache | client | reuse by repo path + git commit + index config |

The artifact-cache policy in `docs/pure-machine-search-artifact-cache-scope-20260630.md` should be reused for compiled-kernel and benchmark artifacts, but
repo indexing belongs to the client layer.

## Phase R0: Boundary Audit

Goal: document the current tinygrad server behavior and identify runtime/client leakage.

Build:

```text
extra/tinygrad_runtime_boundary_audit.py
```

Outputs:

```text
bench/tinygrad-runtime-boundary/latest.json
bench/tinygrad-runtime-boundary/summary.md
```

Required checks:

- existing `/v1/models` route
- existing `/v1/chat/completions` route
- current model load path
- current `max_context` handling
- current cache-prefix behavior
- current streaming behavior
- missing oversized-prompt guard
- missing load/unload/status controls

Verdicts:

```text
R0_PASS_BOUNDARY_PINNED
R0_BLOCKED_SERVER_ENTRYPOINT_AMBIGUOUS
```

## Phase R1: Runtime Status Endpoint

Goal: add read-only runtime introspection before adding lifecycle mutation.

Implement:

```text
GET /runtime/status
GET /runtime/models
GET /runtime/metrics
```

Acceptance:

- reports loaded model id/path
- reports `max_context`
- reports whether model is loaded
- reports last request prefill/decode metrics if available
- does not change `/v1/*`
- works while idle and after one chat request

Verdicts:

```text
R1_PASS_STATUS_ENDPOINTS
R1_BLOCKED_SERVER_STATE_NOT_CENTRALIZED
```

## Phase R2: Prompt Guard + Error Schema

Goal: make the runtime safe as a provider.

Implement:

- explicit prompt-token overflow guard before `Transformer.generate`
- consistent JSON error body
- provider-compatible HTTP status codes
- request id in errors

Required errors:

```text
model_not_loaded
unknown_model
context_length_exceeded
generation_cancelled
runtime_busy
internal_runtime_error
```

Acceptance:

- oversized prompt returns a structured error, not a tensor shape failure
- normal `/v1/chat/completions` still streams
- OpenAI-compatible clients receive parseable JSON on non-stream failures

Verdicts:

```text
R2_PASS_PROVIDER_ERROR_CONTRACT
R2_BLOCKED_STREAM_ERROR_COMPATIBILITY
```

## Phase R3: Model Registry

Goal: stop hardcoding the model list directly in server code.

Implement:

```text
runtime_models.json
```

or:

```text
~/.config/tinygrad/runtime_models.json
```

Minimum fields:

- id
- path or URL
- architecture if known
- quant if known
- default max_context
- tags
- enabled

Acceptance:

- `/v1/models` reads from registry
- existing built-in model aliases still work as fallback
- no model load behavior changes yet

Verdicts:

```text
R3_PASS_MODEL_REGISTRY_READONLY
R3_BLOCKED_MODEL_ALIAS_COMPAT
```

## Phase R4: Load / Unload Lifecycle

Goal: allow proprietary app to select model -> load -> infer -> unload.

Implement:

```text
POST /runtime/load
POST /runtime/unload
POST /runtime/warmup
```

Policy:

- one loaded model per runtime process for first cut
- reject concurrent load during generation
- unload must release model references and clear KV/prefix cache
- warmup should be explicit and report cost

Acceptance:

- load a registry model
- run `/v1/chat/completions`
- unload
- status returns unloaded
- reload same model
- route still works

Verdicts:

```text
R4_PASS_MODEL_LIFECYCLE_SINGLE_MODEL
R4_BLOCKED_GPU_MEMORY_NOT_RELEASED
R4_BLOCKED_CONCURRENT_REQUEST_STATE
```

## Phase R5: Streaming Compatibility

Goal: make streaming robust enough for OpenCode and proprietary clients.

Checks:

- SSE formatting
- final chunk
- usage accounting
- `finish_reason`
- client disconnect handling
- cancellation

Implement:

```text
POST /runtime/cancel
```

Acceptance:

- OpenAI-compatible streaming parser accepts chunks
- non-stream response mode works if requested
- cancellation stops generation and leaves runtime usable

Verdicts:

```text
R5_PASS_STREAMING_COMPAT
R5_BLOCKED_CANCELLATION_STATE
```

## Phase R6: Client-Side Context Contract

Goal: define the prompt envelope the proprietary app should produce.

Build doc/schema:

```text
docs/tinygrad-client-context-envelope-v1.md
```

Recommended XML-style sections:

```xml
<runtime>
  <model id="qwen3-8b-q4k" max_context="4096"/>
</runtime>

<session_summary>
...
</session_summary>

<repo path="/home/ubuntu/tinygrad-arkey" commit="...">
  <file path="tinygrad/llm/model.py" start="1551" end="1600">
...
  </file>
</repo>

<task>
...
</task>
```

Acceptance:

- schema names required sections
- describes token-budget priority order
- explicitly says runtime treats this as plain text
- includes citation format for local files

Verdicts:

```text
R6_PASS_CONTEXT_ENVELOPE_SPEC
```

## Phase R7: Repo Index Adapter Boundary

Goal: keep repo loading outside tinygrad while making it easy for the proprietary app to plug in.

Define client-side interface:

```text
index_repo(path, commit?)
search_repo(query, filters)
pack_context(hits, token_budget)
invalidate_repo(path)
```

Acceptance:

- no tinygrad runtime dependency
- can be implemented by proprietary app, OpenCode references, or a separate local service
- output is message content for `/v1/chat/completions`

Verdicts:

```text
R7_PASS_REPO_CONTEXT_BOUNDARY
```

## Phase R8: Kernel / Runtime Cache Integration

Goal: avoid paying repeated warmup/JIT cost when selecting model -> inference.

Use the artifact-cache model from `docs/pure-machine-search-artifact-cache-scope-20260630.md`.

Acceptance:

- runtime reports cache hits/misses
- warmup output includes compiled-kernel count or equivalent proxy
- cache key includes model path/hash, tinygrad code hash, backend target, relevant env flags
- correctness/speed artifacts are not reused for promotion unless fingerprint matches

Verdicts:

```text
R8_PASS_RUNTIME_CACHE_OBSERVABILITY
R8_BLOCKED_COMPILE_CACHE_NOT_ADDRESSABLE
```

## Phase R9: Provider Compatibility Gate

Goal: prove the runtime can be used by both OpenCode-style and proprietary clients.

Build:

```text
extra/tinygrad_provider_compat_gate.py
```

Checks:

- `GET /v1/models`
- non-stream chat completion
- stream chat completion
- max_tokens stop
- context overflow error
- `/runtime/load` then `/v1/chat/completions`
- `/runtime/unload`

Manual OpenCode config example:

```json
{
  "provider": {
    "tinygrad": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "tinygrad",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1"
      },
      "models": {
        "qwen3-8b-q4k": {}
      }
    }
  }
}
```

Acceptance:

- gate passes against local tinygrad server
- OpenAI-compatible client can stream
- proprietary runtime controls remain outside `/v1/*`

Verdicts:

```text
R9_PASS_PROVIDER_COMPAT
R9_BLOCKED_OPENAI_COMPAT_SURFACE
```

## Phase R10: Production Hardening

Goal: make the runtime boring to operate.

Required decisions:

- one request at a time vs queue
- one model per process vs multi-model
- unload while generating policy
- memory pressure behavior
- request timeout
- startup model auto-load
- config file location
- logs/metrics format

Acceptance:

- documented operational policy
- no silent concurrent cache corruption
- clear rollback to current CLI server

Verdicts:

```text
R10_PASS_OPERATIONAL_POLICY
```

## Suggested Execution Order

Do this in order:

```text
R0 boundary audit
R1 read-only status
R2 prompt guard + errors
R3 model registry
R4 load/unload lifecycle
R5 streaming/cancel
R6 context envelope spec
R7 repo-index boundary
R8 runtime cache observability
R9 provider compatibility gate
R10 hardening
```

Do not start with repo indexing. That belongs to the client and can be built independently once the runtime provider boundary is stable.

## Success Definition

This roadmap is successful when:

```text
OpenCode can call tinygrad through /v1/chat/completions
and
the proprietary app can select/load/unload models through /runtime/*
and
repo/session/context logic remains outside tinygrad
and
the runtime exposes enough status/cache metrics to avoid reloading/regenerating unnecessarily
```

The final product split should be:

```text
tinygrad:
  local inference runtime

proprietary app / OpenCode:
  agent shell, context manager, repo loader, tools, UX
```

