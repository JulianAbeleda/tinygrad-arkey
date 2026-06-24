# FILE_INDEX — active core (repo front door)

The fork's **active/core surface**: **100 ● live** files (KEEP_CORE / KEEP_LIVE_TOOLING / KEEP_LIBRARY_HELPER). Everything else is current docs/tests or historical provenance (kept under `docs/archive/` + dated `bench/<arc>/`).

Generated from `bench/qk-repo-principles-cleanup/inventory.json` (HEAD `7648f72a3`, 2026-06-24) — do not hand-edit. Regenerate: `build_repo_inventory.py` then `build_folder_indexes.py`.

**Repo totals:** ARCHIVE_PROVENANCE **1311** · KEEP_DOC_AUTHORITY **182** · KEEP_TEST **115** · KEEP_LIVE_TOOLING **64** · KEEP_CORE **29** · KEEP_LIBRARY_HELPER **7**

Drill-down: `extra/FILE_INDEX.md` · `tinygrad/llm/FILE_INDEX.md` · `bench/FILE_INDEX.md` · `docs/README.md` (canonical state) · `docs/provenance-index-20260624.md` (archive map).

## Core runtime — decode/prefill CLI, model, gguf (6)

| file | recommendation | LOC | description |
|---|---|---:|---|
| `tinygrad/llm/__init__.py` | KEEP_CORE | 1 | (no docstring) |
| `tinygrad/llm/__main__.py` | KEEP_CORE | 2 | (no docstring) |
| `tinygrad/llm/chat.html` | KEEP_CORE | 39 | HTML |
| `tinygrad/llm/cli.py` | KEEP_CORE | 279 | (no docstring; def remote_pressure_snapshot (+5 more)) |
| `tinygrad/llm/gguf.py` | KEEP_CORE | 188 | ggml packs each iq grid entry as N bytes (N=4 for uint32 grids, N=8 for uint64 grids) in a single word. See ggml-common. |
| `tinygrad/llm/model.py` | KEEP_CORE | 1497 | Prefill v2 (opt-in, default off; decode 100% untouched when off). Concrete-ubatch fp16 prefill that lets |

## qk_* tooling & q4k/q8 primitives (62)

| file | recommendation | LOC | description |
|---|---|---:|---|
| `extra/amd_warp_reduce.py` | KEEP_LIVE_TOOLING | 48 | Shape-safe warp/lane primitives for AMD (gfx1100, wave32) custom kernels. |
| `extra/cl_android.sh` | KEEP_CORE | 5 | source extra/cl_android.sh |
| `extra/llm_adapter.py` | KEEP_LIVE_TOOLING | 161 | (no docstring; def expand_lora_targets (+5 more)) |
| `extra/llm_eval_common.py` | KEEP_LIBRARY_HELPER | 166 | (no docstring; def read_jsonl (+12 more)) |
| `extra/llm_generate.py` | KEEP_LIBRARY_HELPER | 106 | Shared LLM generation core for the flywheel rollout/eval tooling. |
| `extra/llm_json_scorer.py` | KEEP_LIVE_TOOLING | 119 | (no docstring; def wilson_interval (+2 more)) |
| `extra/mesa/lvp_nir_options.sh` | KEEP_CORE | 25 | shell script |
| `extra/nvJitLink.h` | KEEP_CORE | 54 | C/C++ header |
| `extra/q4_k_gemv_primitive.py` | KEEP_LIBRARY_HELPER | 852 | (no docstring; def parse_opt (+22 more)) |
| `extra/q4_k_safety.py` | KEEP_LIVE_TOOLING | 27 | (no docstring; def q4k_device_label (+3 more)) |
| `extra/q4k_mmvq_handwritten.hip` | KEEP_CORE | 79 | ---- Q4_K GGUF layout: 256 weights/block, 144 bytes ---- |
| `extra/q4k_w4a16_handwritten.hip` | KEEP_CORE | 55 | W4A16: fp16 activation x int4 weight, dequant to fp, FMA. No q8 pack. (byte-identical to fp path) |
| `extra/q6_k_gemv_primitive.py` | KEEP_LIVE_TOOLING | 261 | (no docstring; def parse_opt (+6 more)) |
| `extra/q8_ffn_fast_artifact_probe.py` | KEEP_LIVE_TOOLING | 436 | (no docstring; def hip_norm_source (+12 more)) |
| `extra/q8_ffn_graph_route.py` | KEEP_LIVE_TOOLING | 99 | (no docstring; class Q8ArtifactRunner (+4 more)) |
| `extra/q8_ffn_handwritten_oracle.py` | KEEP_LIBRARY_HELPER | 276 | (no docstring; def q8_blocks (+4 more)) |
| `extra/q8_ffn_hcq_artifact.py` | KEEP_LIVE_TOOLING | 252 | (no docstring; def make_buffer (+4 more)) |
| `extra/q8_ffn_oneblock_route.py` | KEEP_LIVE_TOOLING | 241 | (no docstring; def realized_buf (+10 more)) |
| `extra/qk_asm_scheduler.py` | KEEP_LIVE_TOOLING | 432 | ASM instruction scheduler for the prefill GEMM -- Increment 0: IR + dependency DAG + identity proof. |
| `extra/qk_candidate_template_gen.py` | KEEP_LIVE_TOOLING | 219 | Candidate-template generation layer v0 — the 'generate' step of the lifecycle-search loop. |
| `extra/qk_clock_pin.py` | KEEP_LIVE_TOOLING | 68 | Reusable GPU clock pin for reproducible decode timing (RX 7900 XTX / gfx1100). |
| `extra/qk_decode_eval.py` | KEEP_LIVE_TOOLING | 297 | Decode evaluation harness — the project's first machine-search evaluator (infrastructure, NOT kernel work). |
| `extra/qk_decode_fused_flash_tile_ab.py` | KEEP_LIVE_TOOLING | 102 | Phase 2 Candidate A (decode-latency-hiding scope): fully-fused flash-decode tile prototype A/B. |
| `extra/qk_decode_mmvq_graph_route.py` | KEEP_LIVE_TOOLING | 183 | (no docstring; class FixedLaunchRunner (+6 more)) |
| `extra/qk_decode_mmvq_hip_interpose.ver` | KEEP_CORE | 14 | linker version script |
| `extra/qk_decode_mmvq_kernarg_capture.cpp` | KEEP_CORE | 203 | Decode MMVQ P2 capture shim. |
| `extra/qk_decode_mmvq_p3_q4_correctness.py` | KEEP_LIVE_TOOLING | 170 | P3 standalone correctness for imported llama Q4_K MMVQ through tinygrad HCQ. |
| `extra/qk_decode_mmvq_p5_lifecycle_probe.py` | KEEP_LIVE_TOOLING | 238 | P5 lifecycle probe for imported llama Q4_K MMVQ. |
| `extra/qk_decode_runtime_overhead.py` | KEEP_LIVE_TOOLING | 85 | Arc 4 Phase 0: decode host/runtime overhead accounting. Cleanly isolates per-token host-sync overhead without |
| `extra/qk_decode_warp_flash_tile_ab.py` | KEEP_LIVE_TOOLING | 133 | VECTOR_FLASH_DECODE_TILE lever #2: warp-cooperative q.k flash tile (llama flash_attn_tile structure). |
| `extra/qk_ffn_gemv_warp_ab.py` | KEEP_LIVE_TOOLING | 80 | Local A/B for the lossless q4k_gemv_warp work-decomposition variant vs the default q4k_gemv_partial, at the FFN |
| `extra/qk_ffn_gemv_warp_wd.py` | KEEP_LIVE_TOOLING | 77 | W==D for the lossless q4k_gemv_warp FFN gate/up route (Q4K_GEMV_WARP=1) vs the default, in-process interleaved A/B |
| `extra/qk_flash_decode.py` | KEEP_LIVE_TOOLING | 353 | Approach B: custom Flash-Decoding kernels for batch-1 GQA decode attention. |
| `extra/qk_fused_flash_concrete_gate_ab.py` | KEEP_LIVE_TOOLING | 214 | Fused-Flash CONCRETE-shape decode-attention gate vs gqa_coop_vec (local A/B). |
| `extra/qk_fused_softmax_v_tail_ab.py` | KEEP_LIVE_TOOLING | 161 | Path A — fused online-softmax+V TAIL candidate vs gqa_coop_vec (local A/B). |
| `extra/qk_harness_contract.py` | KEEP_LIVE_TOOLING | 160 | Shared evaluator-contract helper for performance-claiming harnesses. |
| `extra/qk_layout.py` | KEEP_LIBRARY_HELPER | 183 | (no docstring; class GGUFInfo (+23 more)) |
| `extra/qk_lifecycle_search_loop.py` | KEEP_LIVE_TOOLING | 218 | Lifecycle-search loop v0 — the first closed generate -> evaluate -> prune loop on top of the decode evaluator. |
| `extra/qk_llama_fattn_kernarg_capture.cpp` | KEEP_CORE | 110 | Route B B1.1 capture — LD_PRELOAD shim to capture llama.cpp's flash_attn_tile / combine / mask_to_KV_max |
| `extra/qk_llama_flash_attn_tile_hcq_ab.py` | KEEP_LIVE_TOOLING | 297 | Route B B1.2-1.4 — vendored llama flash_attn_tile launched through tinygrad's HCQ, local A/B vs gqa_coop_vec. |
| `extra/qk_llama_flash_attn_tile_oracle_ab.py` | KEEP_LIVE_TOOLING | 157 | Llama flash_attn_tile REFERENCE ORACLE — does llama's decode attention tile beat gqa_coop_vec STANDALONE? |
| `extra/qk_matmul_pv_diagnostic_ab.py` | KEEP_LIVE_TOOLING | 183 | Matmul-PV diagnostic candidate vs gqa_coop_vec (local A/B). |
| `extra/qk_modes.py` | KEEP_LIVE_TOOLING | 137 | Centralized enum definitions for kernel `mode` and `prompt_format` values. |
| `extra/qk_nll_eval.py` | KEEP_LIVE_TOOLING | 70 | Teacher-forced decode-path NLL evaluator — the quality gate for the B3 demotion search. |
| `extra/qk_north_star_flash_attn_tile_ab.py` | KEEP_LIVE_TOOLING | 137 | North-star flash_attn_tile decode candidate — LOCAL A/B vs the current winner gqa_coop_vec. |
| `extra/qk_owned_flash_decode.hip` | KEEP_CORE | 284 | Route B B3 — OWNED hand-authored decode-attention tile for TINYGRAD's KV layout (not vendored llama code). |
| `extra/qk_owned_flash_decode_amdgcn_b3.py` | KEEP_LIVE_TOOLING | 233 | Route B B3 — OWNED hand-AMDGCN decode-attention tile (tinygrad KV layout), local A/B vs gqa_coop_vec. |
| `extra/qk_owned_flash_decode_graph_node.py` | KEEP_LIVE_TOOLING | 232 | Route B B4 -- the EXTERNAL-precompiled-AMDGCN-kernel-as-JIT-graph-node capability. |
| `extra/qk_paths.py` | KEEP_LIBRARY_HELPER | 16 | Single source of truth for the default Qwen3-8B-Q4_K_M weights path used by the fork's decode/eval |
| `extra/qk_policy_consistency_check.py` | KEEP_LIVE_TOOLING | 112 | Canonical policy/headline consistency guardrail (docs hygiene; no GPU, no kernels). |
| `extra/qk_prefill_blas_ceiling.cpp` | KEEP_CORE | 339 | C++ source |
| `extra/qk_prefill_blas_sequence.cpp` | KEEP_CORE | 50 | Baseline experiment (no tinygrad dependency added): time the FULL per-layer prefill matmul SEQUENCE with rocBLAS, |
| `extra/qk_prefill_bridge_shim.cpp` | KEEP_CORE | 47 | EBT-1 Lane-A shim: run a rocBLAS fp16 GEMM directly on tinygrad-owned (HCQ/KFD) VRAM pointers, no copies. |
| `extra/qk_prefill_graph_gemm_route.py` | KEEP_LIVE_TOOLING | 119 | (no docstring; def route_pf16_graph_gemm) |
| `extra/qk_quantize.py` | KEEP_LIVE_TOOLING | 85 | fp16/fp32 weights -> Q4_K block bytes (the inverse of q4_k_reference). Port of llama.cpp's |
| `extra/qk_split_kv_economics_audit.py` | KEEP_LIVE_TOOLING | 328 | Split-KV economics audit for decode-attention candidates (durable, reusable). |
| `extra/qk_tensile_hcq_launch.py` | KEEP_LIBRARY_HELPER | 125 | TPE-3 — minimal HCQ launch proof: run the selected rocBLAS Tensile ffn_gate/up kernel from tinygrad HCQ on |
| `extra/qk_tensile_inmodel.py` | KEEP_LIVE_TOOLING | 92 | A3 in-model Tensile prefill route (research-only, PREFILL_TENSILE_GEMM=1). Install-once, robust routing. |
| `extra/qk_tensile_kernarg_capture.cpp` | KEEP_CORE | 74 | TPE-3/5 (capture) — LD_PRELOAD shim: intercept hipModuleGetFunction (to map hipFunction_t -> kernel symbol) and |
| `extra/qk_tensile_kernarg_capture_all.cpp` | KEEP_CORE | 43 | Variant-capture shim: like qk_tensile_kernarg_capture.cpp but captures EVERY distinct dispatched kernel SYMBOL |
| `extra/qk_tensile_runtime.py` | KEEP_LIVE_TOOLING | 86 | TPE-7b — TensileRunner: a runtime object conforming to the HCQGraph protocol, so the extracted Tensile kernel can |
| `extra/qk_tensile_solution_sweep.cpp` | KEEP_CORE | 37 | Host-only rocBLAS: enumerate ALL solutions for the gateup GEMM (m=512,n=12288,k=4096, HHS) and dispatch each once, |

## Machine-search & audit builders (24)

| file | recommendation | LOC | description |
|---|---|---:|---|
| `bench/qk-active-surface-reduction/build_docs_index.py` | KEEP_LIVE_TOOLING | 93 | Roadmap #5 — mechanized docs supersession index (consolidation, NO deletion). |
| `bench/qk-active-surface-reduction/build_inventory.py` | KEEP_LIVE_TOOLING | 152 | Phase 0 reference-graph inventory for the perf-probe active-surface reduction. |
| `bench/qk-decode-eval/HARNESS_GUIDE.md` | KEEP_LIVE_TOOLING | 159 | Decode Harness Best Practices |
| `bench/qk-decode-eval/README.md` | KEEP_LIVE_TOOLING | 66 | qk-decode-eval — the decode machine-search evaluator |
| `bench/qk-decode-eval/binding_template_schema.json` | KEEP_LIVE_TOOLING | 51 | Schema for bench/qk-decode-eval/binding_templates.json — the evaluator-binding contract that a north-star decode-attenti |
| `bench/qk-decode-eval/binding_templates.json` | KEEP_LIVE_TOOLING | 263 | JSON: schema, date, comment, split_kv_economics_contract_v1, templates |
| `bench/qk-decode-eval/candidates.json` | KEEP_LIVE_TOOLING | 375 | JSON: schema, comment, thresholds_default, candidates, suites |
| `bench/qk-decode-eval/schema.json` | KEEP_LIVE_TOOLING | 87 | Machine-readable verdict artifact emitted by extra/qk_decode_eval.py per candidate run. |
| `bench/qk-decode-eval/summaries/latest.json` | KEEP_LIVE_TOOLING | 49 | JSON: date, git_commit, rows |
| `bench/qk-docs-archive/run_archive.py` | KEEP_LIVE_TOOLING | 146 | 2026-06-24 docs declutter: move all non-current docs/*.md into docs/archive/ and rewrite live-surface |
| `bench/qk-lifecycle-search/candidates.json` | KEEP_LIVE_TOOLING | 965 | JSON: schema, date, commit, scope_doc, candidates, refutation_memory |
| `bench/qk-lifecycle-search/evaluator_contract.json` | KEEP_LIVE_TOOLING | 47 | JSON: schema, date, evaluator, registry, run_schema, purpose |
| `bench/qk-lifecycle-search/generated_candidates.json` | KEEP_LIVE_TOOLING | 289 | JSON: schema, method, rows, summary |
| `bench/qk-lifecycle-search/policy_exports.json` | KEEP_LIVE_TOOLING | 43 | JSON: schema, note, policies |
| `bench/qk-lifecycle-search/refutations.json` | KEEP_LIVE_TOOLING | 452 | JSON: schema, entries, candidate_pruning, validation |
| `bench/qk-lifecycle-search/runner_bindings.json` | KEEP_LIVE_TOOLING | 131 | JSON: schema, rows, validation |
| `bench/qk-lifecycle-search/search_candidates.json` | KEEP_LIVE_TOOLING | 115 | JSON: schema, date, comment, candidates, suites |
| `bench/qk-lifecycle-search/search_policy.json` | KEEP_LIVE_TOOLING | 44 | JSON: schema, date, comment, closed_lanes, forbidden_promotions, promotion_intents_requiring_owner |
| `bench/qk-lifecycle-search/search_schema.json` | KEEP_LIVE_TOOLING | 78 | Machine-readable artifact emitted by extra/qk_lifecycle_search_loop.py per search run. |
| `bench/qk-lifecycle-search/summary.md` | KEEP_LIVE_TOOLING | 43 | Primitive lifecycle search - 2026-06-19 |
| `bench/qk-lifecycle-search/template_schema.json` | KEEP_LIVE_TOOLING | 35 | Schema for bench/qk-lifecycle-search/templates.json — the candidate-template registry consumed by extra/qk_candidate_tem |
| `bench/qk-lifecycle-search/templates.json` | KEEP_LIVE_TOOLING | 63 | JSON: schema, date, comment, templates, suites |
| `bench/qk-repo-principles-cleanup/build_folder_indexes.py` | KEEP_LIVE_TOOLING | 188 | Per-folder FILE_INDEX.md generator — so each folder shows which files are LIVE vs provenance + a description. |
| `bench/qk-repo-principles-cleanup/build_repo_inventory.py` | KEEP_LIVE_TOOLING | 366 | Whole-repo principles cleanup audit — reference-graph inventory builder (2026-06-21). |

## Root config & entry docs (8)

| file | recommendation | LOC | description |
|---|---|---:|---|
| `.gitignore` | KEEP_CORE | 103 | (file) |
| `.python-version` | KEEP_CORE | 2 | (file) |
| `README.md` | KEEP_CORE | 219 | <!-- |
| `pyproject.toml` | KEEP_CORE | 266 | TOML config |
| `spec/README.md` | KEEP_CORE | 2 | Run `./render.sh` whenever you update tinyspec.tex to regenerate tinyspec.pdf. |
| `spec/render.sh` | KEEP_CORE | 11 | shell script |
| `spec/tinyspec.pdf` | KEEP_CORE | 1094 | PDF |
| `spec/tinyspec.tex` | KEEP_CORE | 458 | LaTeX source |
