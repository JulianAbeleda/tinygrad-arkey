# Repo Audit — 2026-06-16

Full principles-based audit of the project-authored surface (extra/ 137 files /
26.3k LOC, tinygrad/ +2.5k vs upstream, test/external/ 130 files / 10.7k, docs
68). Measured against [coding-principles.md](coding-principles.md) (esp.
"Reducing Code The Right Way") + [tinygrad-coding-overrides.md](tinygrad-coding-overrides.md).
Conducted by 8 parallel subsystem auditors. READ-ONLY; nothing changed.

## Headline

**The repo is large mostly because ~6,800 LOC of dead one-off probe scripts from
concluded campaigns were never deleted** — exactly the anti-re-sprawl rule's
target. Their verdicts are recorded in `session-handoff.md` and they are
git-preserved, so deletion is safe and principle-aligned.

The principles' own thesis is vindicated: **line count is not the metric**.
The collapsible *knowledge-duplication* is small (~300–420 byte-safe LOC). The
divergent extractors / different-dataset builders are correctly NOT a DRY target
(every auditor independently confirmed this). So the fix is **delete concluded
experiments + a few single-source-of-truth consolidations**, not a rewrite.

## A. Dead concluded-probe deletions (the big lever) — ~6,800 LOC

Each has: zero live importers (or artifact-test-only), a recorded verdict in the
handoff, and full git history. Per the overrides: "one-off probes that have
reached a verdict are deleted once their conclusion is recorded."

| area | LOC | confidence | notes |
|---|---:|---|---|
| `qk_semantic_codegen` v1–v4 + 4 verdict files | ~1,950 | high | delete `codegen_v3` only after un-wiring `qk_flywheel_shadow.py:279` |
| q4_k kernels: 8 zero-import + 4 artifact-test-only probes | ~1,130 | high | marlin_w1b/w2, matmul_decoded, loop_search, opt/policy sweeps, loop_dataset(_smalln), beam_warmstart, track0, batch_ceiling… |
| q4_k kernels: batched_b0 + 3 packed_tile probes | ~1,030 | medium | confirm `test_qk_packed_tile`/`test_qk_batched_b0` don't import the module first |
| bench long-tail: 17 probes (decode_*/memory_access/prefetch/vdot4/wmma_w1/speculative/profile_pmc/quant_sensitivity/gemm_b1/integer_vector_load + `_flash_bench`/`_prefill_bench`/`_s0_safety`/`_flash_verify_model`) + `amd_vdot_smoke` + `weekly_commits_table` | ~2,026 | high | zero importers |
| dead internal kernel variants in `q4_k_gemv_primitive.py` (hoist/coop/fused-intdot) | ~95 | high (byte-safe) | removable once dead probe callers go; live decode path untouched |
| tinygrad core dead-gated branches: `GGUF_Q4K_WIDE`, `GQA_ATTN`, `Q4K_BATCHED` (+ warm-start/BEAM hooks ~50 if loop abandoned) | ~40–90 | high | concluded-negative levers wired into hot paths |
| tests: 14 kernel probe verdict-tests (delete committed-assertion halves; keep pure-logic spec halves) | ~300–350 | high | |
| tests: `test_qk_ansor_transition` codegen v1–v4 methods | ~230 | high | pairs with the semantic deletes |

**Subtotal: ~5,240 high-confidence + ~1,030 medium + ~530–580 tests ≈ ~6,800 LOC.**

## B. Highest-VALUE finding (low LOC, high importance): remove `Ops.QK_BLOCK_DOT`

`Ops.QK_BLOCK_DOT` is a **privileged core UOp op** wired into `uop/__init__.py`,
`uop/ops.py`, `uop/spec.py`, and the C renderer (`cstyle.py`) — for a probe the
handoff records **rejected** (`qk_block_dot_microbench_rejected`, −30% to −90%),
emitted by zero live code. A new dangerous-power core op for an abandoned
experiment violates both "contain dangerous power" and the anti-re-sprawl rule.
**Delete the op + its 4 core touchpoints** (~10 LOC, but the real win is one fewer
privileged boundary in tinygrad core). Highest-priority cleanup.

## C. Genuine knowledge-duplication — byte-safe consolidations (~300–420 LOC)

Real single-source-of-truth violations (NFC, byte-provable). Distinct from the
divergent extractors, which are correctly left alone.

1. **`llm_json_rejection_sample` bypasses `llm_generate`** — re-implements the env
   setup + model load + generate loop that `llm_generate` was just extracted to
   own (the one place the extraction isn't used). Route it through
   `configure_process_env`/`load_model_and_tokenizer`/`generate_one`. ~40 LOC. **Highest-value single fix** (closes the bypass).
2. **Opt-string + load-width parsing rule duplicated 3×** (`cost_model`,
   `dataset_v1`, `feature_enrich`) — one regex rule, three owners → extract one
   `parse_opts()`/`parse_load_width()`. ~35 LOC.
3. **`_load_json`/`_read_json` + id-validated jsonl reader** — 3–4 byte-identical
   copies in `llm_*` while `llm_eval_common` owns the canonical versions. ~30–55.
4. **matrix/roofline/scorecard/gap JSON+`_fmt`+`LLAMA_REFS`+`_git_commit`
   helpers** — duplicated across the surviving bench tools; route through
   `llm_eval_common`/`qk_experiment_matrix`. ~70 LOC.
5. **q4_k_bench device-metric driver** (`_classify`/regex/`_measure`) duplicated
   across 4 live generation/audit tools → one `q4k_bench_driver`. ~80 LOC.
6. **Q4_K/Q6_K scale-min unpack + dequant** duplicated `tinygrad/llm/gguf.py` ↔
   `extra/qk_layout.py` — the GGML block format is real duplicated knowledge;
   make `gguf.py` the source, `qk_layout` import it. ~6 load-bearing lines.
7. **staged-shadow `v0/v2` constants → rows in the existing `STAGED_BATCHES`
   table** (`shadow.py`); collapse the 14-way CLI enum. ~40 LOC.
8. **shadow re-implements jsonl helpers** `eval_common` owns. ~10 LOC.
9. **test fixtures**: `REPO`/`read_jsonl`/`triage_row` factory copy-pasted across
   36 test files → one `test/external/_qk_testutil.py` (no `conftest.py` exists).
   ~120 LOC net.
10. small: `_majority` (cost_model↔triage_eval), dead branches in `feature_enrich`
    / `targeted_outcomes`, `adapter` summary_markdown table. ~30 LOC.

## D. Structural (single-source-of-truth / dangerous-power) — not LOC-driven

- **~22 stringly-typed env flags** (`Q4K_*`/`Q6K_*`/`FLASH_*`/`QK_*`) read ad hoc
  in `model.py`, validated only by scattered `if x not in (...)` raises →
  centralize into one validated `QKConfig` dataclass parsed at `from_gguf`
  (preserves the AMD env-ordering invariant; makes invalid states unrepresentable).
- **BEAM/warm-start module-global hooks** (`postrange.py`/`search.py`
  `_WARMSTART_OPTS`/`_BEAM_CANDIDATE_FILTER`/`_BEAM_SCHEDULE_LOG`) poked from
  `extra/` probes via private mutation → a documented `set_beam_hooks()` boundary,
  or remove with the abandoned loop.
- Dangerous power is otherwise **well-contained**: `_dp4a` gated emit, inline asm
  behind `q4_k_safety` gates, flash-decode single entry point, `QKPrimitiveBudget`
  validation, the `decode_enabled` decode/prefill orthogonality — all exemplary.

## E. Hold — needs sign-off (changes bytes/evidence, not byte-provable)

- **adapter dataset pipeline scaffold** (~135–160 LOC) — real dup, but **blocked
  on adding golden regen tests** for the `training-data-*` artifacts first (goldens
  currently cover only `qk_flywheel_*`). Add goldens → then byte-provable.
- **dataset-chain data-model rewrite** (~1,500) and **verdict/shadow fold** (~900)
  — change the `plus` corpus the cost-model/shadow conclusions rest on → evidence
  change, not a refactor. Do only with explicit sign-off + revalidation.
- mechanism-classifier table unify (~20), qwen `--preset` flag replacing the 2
  alias wrappers (~36), signal/v0 question-pool unify (~50) — inputs-changes.

## F. What is already good (do NOT touch)

`llm_generate` (deep module, lazy import preserves env-ordering), `qk_modes`
(enums = encode-invariants done right), `qk_paths.portable_path` (canonical
portability boundary), `qk_layout`/`qk_quantize` (decoder vs encoder — different
operations, legitimately separate), `q4_k_safety` (risky-search gate), the
golden/reproduce-from-artifact tests (`test_flywheel_*_golden`), `assemble_row`
(single row-schema authority), the dual cost-model backends, the leakage-audit
centralization. The campaigns respected DRY-is-knowledge: dequant lives once,
Marlin probes import `_q4k_weight` rather than re-deriving it.

## Phased execution plan (each its own commit, never red, verified)

1. **Delete dead probes** (A) in grouped commits by area (`[test]` for extra/
   probes + their tests; `[codegen]`/`[nn]` for the tinygrad core op + gated
   branches). Un-wire `shadow.py:279` before deleting `codegen_v3`. Confirm the 4
   medium-confidence packed_tile/batched tests don't import their module. Run full
   suite after each group. **~6,800 LOC, the bulk of the win.**
2. **Remove `Ops.QK_BLOCK_DOT`** (B) from core — `[codegen]`.
3. **Byte-safe consolidations** (C) as separate NFC commits, each byte-proven
   (golden/regen/fixed-seed). Start with the `rejection_sample → llm_generate`
   bypass (#1). **~300–420 LOC.**
4. **Structural** (D): `QKConfig` typed flags; guard the BEAM globals.
5. **Hold** (E) until signed off.

Total defensible reduction: **~7,000–7,500 LOC (~25%+ of the project surface)**,
~6,800 of it via deleting concluded, git-preserved probes — and the durable win
is the overrides' anti-re-sprawl rule now being enforced so it doesn't re-grow.
