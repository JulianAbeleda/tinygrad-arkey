# Repo Map

Last updated: 2026-06-21 (regenerated after the active-surface reduction + whole-repo principles audit)
Purpose: compact file ownership map for agents before reading source. Doc map: `../../docs/README.md`.
Whole-repo file classification SSOT: `../../bench/qk-repo-principles-cleanup/inventory.json` (every tracked file →
KEEP_CORE / LIVE_TOOLING / LIBRARY_HELPER / DOC_AUTHORITY / TEST / ARCHIVE_PROVENANCE / vendor).

## Root

- `README.md` - upstream tinygrad README + a fork-orientation header (entry points).
- `tinygrad/` - the (AMD-only) framework core (vendor; `tinygrad/llm/` is the fork's runtime).
  `extra/` - fork's decode primitives, kernels, and the evaluator/search lifecycle (top-level `qk_*`/`q*_k_*`/`llm_*`
  are fork; the `extra/` subdirs `thunder`/`gemm`/`sqtt`/`amdpci`/… are upstream).
- `test/external/` + `test/unit/` - the fork's test suites. `bench/**` - ledgers (force-added) + gitignored artifacts.
- `structure/` - human/role layer + this cache. `docs/` - engineering docs (see `docs/README.md`).

## Entrypoints

- `tinygrad/llm/cli.py` — the LLM CLI: interactive chat, `--benchmark N` (decode tok/s), `--serve` (OpenAI-compat).
  Key symbols: `main`, `LLMServer`, `SimpleTokenizer`.
- `tinygrad/llm/model.py` — the decode model + QK quantized-primitive install/demotion + flash-decode route (hot
  path). Key symbols: `Transformer.from_gguf`, `QKConfig`, `Q4KPrimitiveLinear`/`Q6KPrimitiveLinear`,
  `_install_q4k/q6k_primitives`, `_demote_q6k_to_q4`, `_q4k_policy`, `generate`/`logits`.

## Core Modules

- `extra/q4_k_gemv_primitive.py`, `extra/q6_k_gemv_primitive.py` — the custom decode GEMV/GEMM kernels (UOp + raw-C).
- `extra/qk_quantize.py` — the from-scratch Q4_K quantizer (`quantize_q4_k`); the only quant format (no sub-4-bit).
- `extra/qk_flash_decode.py` — flash-decode attention (`FLASH_DECODE`, `gqa_coop_vec`); long-context, exact.
- `extra/qk_layout.py`, `extra/qk_paths.py` — shared libs (GGUF layout/quant helpers; `portable_path` +
  `DEFAULT_MODEL_GGUF`). Most-imported helpers in the fork.

## Evaluator / Search Lifecycle (the closed loop)

- `extra/qk_decode_eval.py` — the W==D decode evaluator + promotion/refutation authority.
- `extra/qk_lifecycle_search_loop.py` — generate → evaluate → prune loop on the evaluator.
- `extra/qk_candidate_template_gen.py` — the candidate-spec 'generate' step.
- `extra/qk_harness_contract.py` — artifact stamping (13-field contract) + the `DECODE_COMPARATOR` SSOT mirror.
- `extra/qk_nll_eval.py` — teacher-forced decode-path dNLL quality gate.
- `extra/qk_modes.py` — the `Verdict` enum (verdict-string SSOT). `extra/qk_search_spec.py` — `SearchRow`/
  `Constraints`/`AcceptedPolicy` + the `SearchSpace`/`Phase`/… choice enums.
- `extra/qk_clock_pin.py` — the single GPU perf-state (clock-pin) boundary.
- `extra/qk_policy_consistency_check.py` — the canonical-doc guard.
- Ledgers/contracts: `bench/qk-decode-eval/` (`candidates.json`, `schema.json`, `binding_templates.json`,
  `HARNESS_GUIDE.md`) + `bench/qk-lifecycle-search/` (`search_*`, `refutations.json`, `evaluator_contract.json`).

## Data And State

- `*.gguf` weights at `/home/ubuntu/models/` (not in repo). KV cache + repacked primitive storage live in
  VRAM (`QK_PRIMITIVE_STORAGE=shared` views the GGUF in place; `storage_bytes=0`).
- `bench/**` accepted-policy + result artifacts (gitignored; durable verdicts force-added).

## Tests And Scripts

- `test/external/` + `test/unit/` — the fork suites (`pytest test/external/ -q` → 299 pass / 57 skip). Golden/byte-proof
  + reproduce-from-artifact tests (skip-if-absent when `bench/**` artifacts aren't present).
- `extra/q4_k_profile_report.py` — DEBUG=2 decode kernel-bucket profiler.

## Documentation

- **`docs/current-project-state-handoff-20260621.md`** — ⭐ canonical current state (read first).
- **`docs/README.md`** — the doc map (navigation entry). **`docs/provenance-index-20260621.md`** — supersession map
  for the 650+ historical docs (topic → current authority). **`bench/qk-repo-principles-cleanup/inventory.json`** —
  whole-repo file classification. `structure/Development/performance-primitive-research-principles.md` — method.
- `structure/INDEX.md` - project purpose + first pointers. `structure/Purpose/` - LLM role boot layer.

## Read Strategy

1. Read `structure/INDEX.md` (→ `docs/README.md` for engineering work).
2. For current state: `docs/current-project-state-handoff-20260621.md` → the owning `extra/qk_*` /
   `tinygrad/llm/model.py` source. For historical context: `docs/provenance-index-20260621.md`.
3. Read this cache + `structure/cache/repo-cache.md` for stable facts + boundaries before editing.
4. Then read only the owning source file or doc (use `inventory.json` to find the owner).
