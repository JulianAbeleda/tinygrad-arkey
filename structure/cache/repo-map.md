> **⚠ STALE (2026-06-21)** — this cache predates the active-surface reduction (`docs/perf-probe-active-surface-reduction-result-20260621.md`, 291 perf files deleted) and lists removed scripts. Regenerate before relying on it; current authority is `docs/current-project-state-handoff-20260621.md`.

# Repo Map

Last updated: 2026-06-16
Purpose: compact file ownership map for agents before reading source. Doc map: `../../docs/README.md`.

## Root

- `README.md` - upstream tinygrad README + a fork-orientation header (entry points).
- `tinygrad/` - the (AMD-only) framework core. `extra/` - fork's decode primitives, kernels, and search tooling.
- `test/external/` - the fork's test suite. `bench/**` - generated artifacts (gitignored).
- `structure/` - human/role layer + this cache. `docs/` - engineering docs (see `docs/README.md`).

## Entrypoints

- `tinygrad/llm/cli.py`
  - owns: the LLM CLI — interactive chat, `--benchmark N` (decode tok/s), `--serve` (OpenAI-compat).
  - key symbols: `main`, `LLMServer`, `SimpleTokenizer`.
- `tinygrad/llm/model.py`
  - owns: the decode model + QK quantized-primitive install/demotion (the hot path).
  - key symbols: `Transformer.from_gguf`, `QKConfig`, `Q4KPrimitiveLinear`/`Q6KPrimitiveLinear`,
    `_install_q4k/q6k_primitives`, `_demote_q6k_to_q4`, `_q4k_policy`, `generate`/`logits`.

## Core Modules

- `extra/q4_k_gemv_primitive.py`, `extra/q6_k_gemv_primitive.py`
  - owns: the custom decode GEMV/GEMM kernels (UOp + raw-C). key: `q4k_gemv_partial_kernel`, `parse_opt`.
- `extra/qk_quantize.py` - owns: the from-scratch Q4_K quantizer (`quantize_q4_k`); the only quant format (no sub-4-bit).
- `extra/qk_flash_decode.py` - owns: flash-decode attention (`FLASH_DECODE`); long-context, exact.
- `extra/qk_search_spec.py` - owns: the machine-search schema authority (`SearchRow`/`Constraints`/`AcceptedPolicy`).
- `extra/qk_nll_eval.py` - owns: teacher-forced decode-path dNLL quality gate.
- `extra/qk_demote_search.py` - owns: the demotion search orchestrator (first dogfood of the scaffold).
- `extra/llm_eval_common.py` - owns: shared IO + scoring SSOT for the `llm_*`/`qk_*` tooling.

## Data And State

- `*.gguf` weights at `/home/ubuntu/models/` (not in repo). KV cache + repacked primitive storage live in
  VRAM (`QK_PRIMITIVE_STORAGE=shared` views the GGUF in place; `storage_bytes=0`).
- `bench/**` accepted-policy + result artifacts (gitignored; durable ones force-added).

## UI Or Interface

- `tinygrad/llm/cli.py` - CLI + OpenAI-compatible HTTP server. Decode is driven by env flags (see repo-cache).

## Tests And Scripts

- `test/external/` - the fork suite (`pytest test/external/ -q` → 237 pass / 56 skip). Golden/byte-proof +
  reproduce-from-artifact tests (skip-if-absent when `bench/**` artifacts aren't present).
- `extra/q4_k_profile_report.py` - DEBUG=2 decode kernel-bucket profiler. `extra/qk_two_queue_probe.py` -
  the overlap-feasibility probe (re-fire gate for the 2nd-compute-ring build).

## Documentation

- **`docs/README.md`** - the doc map (canonical entry). **`docs/amd-decode-banked-20260616.md`** - current
  decode state + lever map. `structure/Development/session-handoff.md` - running engineering log.
- `structure/INDEX.md` - project purpose + first pointers. `structure/Purpose/` - LLM role boot layer.
- `structure/Development/` - principles, overrides, the active machine-search direction plan.

## Read Strategy

For most development tasks:

1. Read `structure/INDEX.md` (→ points to `docs/README.md` for the engineering work).
2. For the decode/quant work: `docs/README.md` → `docs/amd-decode-banked-20260616.md` → the owning `extra/qk_*`
   / `tinygrad/llm/model.py` source.
3. Read `structure/cache/repo-cache.md` for stable facts + boundaries before editing.
4. Then read only the owning source file or doc.
