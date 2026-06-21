# Repo Cache

Last updated: 2026-06-21 (regenerated after the active-surface reduction + whole-repo principles audit)
Analyzed source: `tinygrad/llm/`, `extra/qk_*`/`llm_*`, `test/external/`, `docs/`, `structure/`.
Scope: stable implementation context for token-saving orientation. Full doc map: `../../docs/README.md`.

> **Current state lives in the handoff, not here.** For live numbers, decided policies, and closed lanes read
> `../../docs/current-project-state-handoff-20260621.md` (canonical). This cache holds only *stable* facts (project
> shape, run commands, boundaries, file ownership) that do not change every time a perf number moves — so it does
> not re-drift. Whole-repo file classification: `../../bench/qk-repo-principles-cleanup/inventory.json`.

## Project Shape

- `tinygrad-arkey`: an **AMD-only hard-fork of tinygrad** for **quantized LLM decode** (Q4_K/Q6_K GGUF) on
  RDNA3 (RX 7900 XTX, gfx1100). Non-AMD backends (NV/CUDA/METAL/QCOM/CL/WEBGPU/DSP) were pruned.
- User-facing surface: `tinygrad/llm/cli.py` (chat / `--benchmark` / OpenAI-compat `--serve`).
- The fork's value is **decode primitives + a closed machine-search/evaluator lifecycle** in `extra/qk_*`.
- What it is NOT: a general multi-backend tinygrad; a training framework (inference/decode focus).

## Core Facts

- Python 3.12, deps via `uv` (`uv sync --extra testing_minimal --extra costmodel`); interpreter `.venv/bin/python`.
- Run decode: `DEV=AMD .venv/bin/python -m tinygrad.llm.cli -m <gguf> --warmup --benchmark N`.
- Models: `/home/ubuntu/models/Qwen3-{1.7B,4B,8B,14B,32B}*.gguf`. The default-model literal has one source of truth:
  `extra/qk_paths.py:DEFAULT_MODEL_GGUF` (the tooling's machine-local fixture; the CLI itself takes `-m`).
- Key runtime files: `tinygrad/llm/model.py` (`Transformer.from_gguf`, `QKConfig`, Q4K/Q6K primitive install +
  demotion, flash-decode route), `tinygrad/llm/cli.py`, `tinygrad/llm/gguf.py`, the kernels
  `extra/q4_k_gemv_primitive.py` / `q6_k_gemv_primitive.py`, the quantizer `extra/qk_quantize.py`, and
  `extra/qk_flash_decode.py` (flash-decode attention).
- **Machine-search/evaluator lifecycle (live):** `extra/qk_decode_eval.py` (W==D evaluator + promotion/refutation
  authority), `qk_lifecycle_search_loop.py` (generate→evaluate→prune), `qk_candidate_template_gen.py`,
  `qk_harness_contract.py` (artifact stamping + comparator SSOT), `qk_nll_eval.py` (dNLL quality gate),
  `qk_clock_pin.py` (the single GPU perf-state boundary), `qk_modes.py` (the `Verdict` enum SSOT),
  `qk_policy_consistency_check.py` (canonical-doc guard). Ledgers: `bench/qk-decode-eval/`,
  `bench/qk-lifecycle-search/`. Harness rules: `bench/qk-decode-eval/HARNESS_GUIDE.md`.
- Env flags (names only): `DEV`, `JIT`, `Q4K_PRIMITIVE`/`Q6K_PRIMITIVE` (auto-on for AMD GGUF), `Q6K_COVER_MORE`,
  `Q6K_DEMOTE_FFNDOWN`/`QK_DEMOTE_TENSORS`, `FLASH_DECODE`/`FLASH_L`/`FLASH_VARIANT`, `QK_GENERATED_POLICY`,
  `QK_PRIMITIVE_STORAGE`, `PREFILL_V2`/`PREFILL_SERVER_PROFILE` (opt-in), `Q8_FFN_HANDWRITTEN` (opt-in).
  Generated artifacts under `bench/**` are gitignored (durable verdict docs force-added).

## Main Flow

1. `cli.py` loads a GGUF → `Transformer.from_gguf` (`model.py`) installs Q4_K/Q6_K decode primitives
   (path-aware, default-on for AMD), applies any demotion/policy.
2. `model.generate` runs the JIT'd decode loop (symbolic `start_pos`); primitives swap the GEMV kernels;
   flash-decode (`FLASH_VARIANT=gqa_coop_vec`, `FLASH_L=128`) serves attention at ctx ≥ threshold.
3. Machine search (when used): `qk_search_spec` rows / generated candidates → `qk_decode_eval` (correctness →
   local A/B → whole-decode W==D → policy) → schema'd verdict → ledger/refutation. No defaults change from a harness.

## Key Boundaries

- **AMD-only**; non-AMD paths gated/removed. Env-ordering invariant: set `DEV`/`JIT`/QK flags **before**
  `from tinygrad import ...` (loaders import tinygrad lazily to preserve this; light tooling like
  `qk_harness_contract`/`qk_paths` avoids importing tinygrad).
- **Do NOT run BEAM / risky schedule search on gfx1100** (hangs). Generated policies are **never a runtime default**.
  Lossy quant demotions / q8 FFN are **dNLL-gated, default-off**. Exact wins (Q6_K coverage, flash) are byte-identical.
- Dangerous-power surface (custom kernels, raw queue/ring, GPU clock pin) stays contained + gated. The GPU
  perf-state mutations live in exactly one boundary: `extra/qk_clock_pin.py` (see `coding-principles`).

## Verification

- `.venv/bin/python -m pytest test/external/ -q` → **299 pass / 57 skip** (the 57 skips = artifact-absent reproduce
  tests, by design; `bench/**` is gitignored).
- Canonical-doc guard: `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_policy_consistency_check.py` (PASS).
- `.venv/bin/python -m py_compile <file>`; `git diff --check`.
- Decode smoke: the `cli --benchmark` command above.

## Current Direction

See `../../docs/current-project-state-handoff-20260621.md` (canonical). In brief, do not re-open: **bounded decode
work is RESTED** (only remaining lever = the north-star full `flash_attn_tile` lifecycle/codegen); the decode
headline is the **curve ~86→~61 tok/s ≈ ~67% llama**, never the bare `87.6`; **global `PREFILL_V2` default OFF**
(opt-in `auto`/server profile); **q8 FFN opt-in, default-off**. Method authority:
`../Development/performance-primitive-research-principles.md`.

## Known Risks

- `bench/**` gitignored → reproduce/golden tests skip-if-absent on a fresh tree (not regressions).
- Dangerous-power builds (overlap 2nd compute ring in `ops_amd.py`; sub-4-bit quantizer+kernel) are deliberate-scope,
  not bolt-on.

## Update Rule

Update when command routing, architecture boundaries, core data contracts, storage layout, verification
commands, or important file ownership change. Keep live perf numbers in the handoff, not here. No transient logs,
secrets, or session content.
