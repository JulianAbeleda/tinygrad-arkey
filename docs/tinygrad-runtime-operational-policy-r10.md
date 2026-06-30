# Tinygrad Runtime Operational Policy (R10)

Date: 2026-06-30

Status: This documents the operational policy of the tinygrad runtime server as implemented in
`tinygrad/llm/cli.py`. It is the "Phase R10: Production Hardening" deliverable of
`tinygrad-runtime-client-separation-roadmap-20260630.md`. The goal is a runtime that is boring to operate:
predictable, hard to corrupt, and easy to roll back.

## What the runtime is

One process that serves two HTTP surfaces on one port:

- `/v1/*` — the OpenAI-compatible inference surface, for clients like OpenCode, AI-SDK tools, or anything that
  already speaks the OpenAI chat/completions protocol.
- `/runtime/*` — lifecycle and introspection controls, for the proprietary app or a local operator.

The runtime owns model load, tokenizer, KV cache, prefill/decode, sampling, and GPU memory. It does **not** own
sessions, repo context, tools, or prompt packing. Those live above the boundary, in the client.

## Decisions

### One request at a time

The server is multi-threaded so that read-only controls (`/runtime/status`, `/runtime/metrics`) and
`/runtime/cancel` stay responsive while a generation is running. But only **one generation runs at a time**. A
single `gen_lock` serializes generation and any model-mutating control. If a second generation arrives while one
is in flight, it gets `429 runtime_busy` immediately rather than corrupting the shared model/KV cache.

Rationale: there is one loaded model with one KV cache and one prefix cache. Two concurrent generations would
interleave writes into that shared state and produce garbage. Refusing the second request is the safe default.

### One model per process

The first cut loads exactly one model at a time. Loading a new model unloads the current one first. Multi-model
hosting (several models resident, routed by request `model` id) is intentionally out of scope for now — it
multiplies GPU memory and complicates the cache story. Run one process per model if you need several.

The request `model` field on `/v1/chat/completions` is **echoed back** but does not switch models. The runtime
always uses the currently loaded model. Selecting a different model is an explicit `/runtime/load` call. This
keeps behavior deterministic and avoids surprise multi-GB loads triggered by a chat request.

### Unload / load while generating

Model-mutating controls — `/runtime/load`, `/runtime/unload`, `/runtime/warmup`, `/runtime/cache/clear` — take
the same `gen_lock` as generation. If a generation is in flight they return `429 runtime_busy`. You cannot pull
the model out from under a running request.

### Cancellation

`/runtime/cancel` sets a flag that the active generation checks between tokens. The generation stops at the next
token boundary, reports cleanly, and leaves the runtime usable (next request works normally). Because the flag is
only checked between yielded tokens, a cancel that arrives during the initial prefill/compile of a cold request
takes effect once the first token would have been produced. Warm requests cancel within one token.

Over the wire, a cancelled stream ends with the standard OpenAI `finish_reason: "stop"`. The fact that it was
cancelled is recorded in `/runtime/metrics` as `last_finish_reason: "cancelled"` for observability.

### Memory pressure

Single-model policy bounds resident memory to one model plus its KV cache (sized by `max_context`). `/runtime/unload`
drops references and runs a GC pass so the GPU buffers can be released before the next load. KV size for a given
model/context is reported via `/runtime/status` (`kv_cache_tokens`) and the model-file size via `/runtime/cache`.

### Request timeout

The runtime does not impose a wall-clock timeout on a generation; `max_tokens` (or `max_completion_tokens`) bounds
the output, and the client can abort via `/runtime/cancel` or by disconnecting (a client disconnect mid-stream is
caught and ends the generation without crashing the server). Operators who want a hard timeout should enforce it
in the client or a reverse proxy.

### Prompt overflow

Every generation runs an explicit guard before the model: if the tokenized prompt is `>= max_context` it returns
`400 context_length_exceeded` with a structured error, instead of a tensor-shape crash. The client is responsible
for truncation/summarization to fit the budget (see `tinygrad-client-context-envelope-v1.md`).

### Startup model auto-load

By default `--serve` preloads the model named by `--model` (a built-in alias or a local GGUF path) and warms the
JIT before accepting requests. Pass `--no-preload` to start with no model and let the client drive the first load
via `/runtime/load`. Either way the registry is available immediately at `/v1/models` and `/runtime/models`.

### Config file location

The model registry is read from `~/.config/tinygrad/runtime_models.json` by default, overridable with `--registry`
or the `TINYGRAD_RUNTIME_MODELS` env var. An example is in `extra/runtime_models.example.json`. Built-in aliases
in `cli.py` always seed the registry as a fallback, so the file is optional.

### Logs and metrics format

Per-request lines go to stderr (prefill tok/s, decode tok/s, in/out token counts, total time). Structured metrics
are available as JSON at `/runtime/metrics` (last prefill/decode tok/s, last prompt/completion tokens, cached
prefix tokens, load/request counts) and at `/runtime/status` (loaded model, max_context, backend/target, warmup
state, busy flag). Cache observability is at `/runtime/cache` (kernel cache db path/size, model file size, prefix
cache token count).

## Error contract

All failures return a parseable JSON body so OpenAI-compatible clients don't choke on non-stream errors:

```json
{ "error": { "message": "...", "type": "<type>", "code": "<type>", "request_id": "req-..." } }
```

| type | HTTP | when |
|---|---|---|
| `model_not_loaded` | 409 | a generation was requested with no model loaded |
| `unknown_model` | 404 | `/runtime/load` got a model id not in the registry and no `path` |
| `context_length_exceeded` | 400 | prompt tokens >= `max_context` |
| `runtime_busy` | 429 | a generation or mutation is already in progress |
| `invalid_request` | 400 | malformed JSON, bad content part, unsupported prompt shape |
| `generation_cancelled` | 499 | reserved for cancel-as-error flows |
| `internal_runtime_error` | 500 | unexpected failure (e.g. a model failed to load) |

## Rollback

The change is contained to `tinygrad/llm/cli.py` plus additive docs and `extra/` scripts. The previous behavior
(preload a model, serve `/v1/models` and `/v1/chat/completions`, benchmark, interactive chat) is preserved:
`--serve`, `--benchmark`, `--warmup`, `--remote-metrics`, and the interactive REPL all work as before. To roll
back, revert the `cli.py` commit; nothing else in the repo depends on the new runtime state.

## Acceptance (R10)

- [x] Documented operational policy (this file).
- [x] No silent concurrent cache corruption — `gen_lock` + `429 runtime_busy` enforce one-at-a-time.
- [x] Clear rollback to the current CLI server — single-file change, additive everything else.

Verdict: **R10_PASS_OPERATIONAL_POLICY**
