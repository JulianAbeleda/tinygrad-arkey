# Tinygrad Runtime / Client Separation — Implementation Status

Date: 2026-06-30

Status: implementation result for `tinygrad-runtime-client-separation-roadmap-20260630.md`. This records what was
built on the tinygrad side, the phase verdicts, and how to run/verify it.

## Summary

The tinygrad server in `tinygrad/llm/cli.py` is now a two-surface runtime:

- `/v1/*` — OpenAI-compatible inference for clients (OpenCode, AI-SDK, llama.cpp-style tooling).
- `/runtime/*` — lifecycle + introspection controls for the proprietary app / local operator.

Runtime state is centralized in a `RuntimeState` object held by the server. The server is threaded so controls
stay responsive during generation; a single `gen_lock` enforces one generation/mutation at a time. All previous
behavior (preload, benchmark, interactive REPL, `--remote-metrics`) is preserved.

## Endpoints

OpenAI-compatible (`/v1/*`):

```
GET  /v1/models                 # reads the registry
POST /v1/chat/completions        # stream + non-stream, usage accounting, finish_reason
POST /v1/completions             # legacy text completion (string prompt)
```

Runtime controls (`/runtime/*`):

```
GET  /runtime/status             # loaded model, max_context, backend/target, warmup, busy, last metrics
GET  /runtime/models             # full registry rows
GET  /runtime/metrics            # last prefill/decode tok/s, prompt/completion tokens, counts
GET  /runtime/cache              # kernel cache db, model file, prefix cache observability
POST /runtime/load               # {model, path?, max_context?, warmup?}
POST /runtime/unload
POST /runtime/warmup
POST /runtime/cancel             # stops the active generation at the next token
POST /runtime/cache/clear        # clears the prompt/KV prefix cache
```

## Phase verdicts

| phase | deliverable | verdict |
|---|---|---|
| R0 | boundary audit (`extra/tinygrad_runtime_boundary_audit.py`) | **R0_PASS_BOUNDARY_PINNED** (19/19 surface checks, no leakage) |
| R1 | read-only status endpoints | **R1_PASS_STATUS_ENDPOINTS** |
| R2 | prompt guard + structured error contract | **R2_PASS_PROVIDER_ERROR_CONTRACT** |
| R3 | model registry (`extra/runtime_models.example.json`, `build_registry`) | **R3_PASS_MODEL_REGISTRY_READONLY** |
| R4 | load / unload / warmup lifecycle, single-model | **R4_PASS_MODEL_LIFECYCLE_SINGLE_MODEL** |
| R5 | streaming compatibility + cancel | **R5_PASS_STREAMING_COMPAT** |
| R6 | client context envelope spec (`tinygrad-client-context-envelope-v1.md`) | **R6_PASS_CONTEXT_ENVELOPE_SPEC** |
| R7 | repo-index adapter boundary (`tinygrad-repo-index-adapter-boundary-v1.md`) | **R7_PASS_REPO_CONTEXT_BOUNDARY** |
| R8 | runtime cache observability (`/runtime/cache`) | **R8_PASS_RUNTIME_CACHE_OBSERVABILITY** |
| R9 | provider compat gate (`extra/tinygrad_provider_compat_gate.py`) | **R9_PASS_PROVIDER_COMPAT** (11/11) |
| R10 | operational policy (`tinygrad-runtime-operational-policy-r10.md`) | **R10_PASS_OPERATIONAL_POLICY** |

## How to run

Start the server (preload a model and warm the JIT):

```
python -m tinygrad.llm.cli --serve 8000 -m qwen3:8b --max_context 4096
# or a local GGUF:
python -m tinygrad.llm.cli --serve 8000 -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
# or start empty and load via the API:
python -m tinygrad.llm.cli --serve 8000 --no-preload
```

Verify the boundary and provider compatibility:

```
python extra/tinygrad_runtime_boundary_audit.py            # R0: static + live audit
python extra/tinygrad_provider_compat_gate.py \
  --base-url http://127.0.0.1:8000 --model qwen3:8b        # R9: 11-check gate
```

Point a client (OpenCode) at it:

```json
{
  "provider": {
    "tinygrad": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "tinygrad",
      "options": { "baseURL": "http://127.0.0.1:8000/v1" },
      "models": { "qwen3-8b-q4k": {} }
    }
  }
}
```

## Validation performed (2026-06-30)

Live-tested against a server preloaded with `Qwen3-0.6B-Q8_0` on AMD gfx1100:

- R9 gate: **11/11 PASS** (models list, non-stream chat, stream chat, max_tokens→length, context overflow error,
  completions, metrics, load, chat-after-load, unload).
- Error contract: `model_not_loaded` (409), `unknown_model` (404), `context_length_exceeded` (400),
  `runtime_busy` (429), `invalid_request` (400) all return structured JSON.
- Concurrency: `/runtime/status` responsive during generation; concurrent generation → 429; `/runtime/cancel`
  stops generation and leaves the runtime usable.
- R0 audit: **19/19** surface checks present, no client-concern leakage.

## What is intentionally NOT in the runtime

Repo indexing, retrieval, file edits, tool execution, session memory, summarization, and prompt packing all live
above the boundary in the client. See the R6 envelope spec and R7 adapter boundary spec.
