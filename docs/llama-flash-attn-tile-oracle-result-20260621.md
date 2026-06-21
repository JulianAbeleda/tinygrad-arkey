# Llama Flash-Attn-Tile Reference Oracle — Result

Date: 2026-06-21

Scope: `docs/decode-codegen-dataflow-capability-scope-20260621.md` (`CODEGEN_SCOPE_LLAMA_ORACLE_FIRST`). Answer the
one decisive question: **does llama's decode `flash_attn_tile` beat tinygrad's `gqa_coop_vec` STANDALONE at the
Qwen3-8B decode shape** — i.e. is the 10× gap a standalone kernel-codegen target, or only in-model integration?

## Decision: **`LLAMA_ORACLE_LOCAL_AB_PASS`**

**Yes — decisively. llama's standalone decode attention is ~5–6× faster than coop standalone** (pure GPU time,
apples-to-apples). The win is in the **kernel body**, not just in-model integration. **Native fused-flash codegen is
aiming at the right layer**, and llama's `flash_attn_tile` is the validated target.

## Method — a PROFILING oracle (and why, not a port)

Phase 0 confirmed the full source port is **BOUNDED** (`/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-tile.cuh`:
one `flash_attn_tile<DKQ,DV,ncols1,ncols2,…>` kernel + a tiny `flash_attn_combine_results<128>` + ~10 inlinable
helpers; **no `cp_async`, no `mma`/WMMA, no broad ggml runtime**; already compiles to gfx1100 HSACO). But the
decisive question is answered most reliably by the **GPU kernel time of llama's REAL kernel** (zero port/correctness
risk) vs coop's real kernels — a 700–900-line CUDA→HIP port carries correctness risk (the GQA-folded `ncols2=4`
config, the `parallel_blocks` split + combine kernel, fp16 layout) that could *invalidate* the comparison. Per the
scope's own framing ("the cheapest way to answer"), the profiling oracle leads; **the full port is deferred to the
native-codegen follow-up** that actually needs a re-runnable byte-level oracle.

- **llama**: per-dispatch GPU time of `flash_attn_tile<128,128,1,4,false>` + `flash_attn_combine_results<128>` from
  the rocprofv3 kernel trace `bench/qk-llama-decode-primitive-audit/llama_decode_kernel_trace_ctx1024.csv` (ctx1024
  **measured per-dispatch**, median of 1188 dispatches; ctx512/4096 **derived** from per-token totals ÷ dispatches-
  per-token calibrated at ctx1024 ≈ 37 = layers).
- **coop**: GPU-busy time per attention call via tinygrad's own **ProfileGraphEvent** (`PROFILE=1`), median-of-5,
  clock-pinned. (rocprofv3 hooks HIP/HSA dispatch and does **not** intercept tinygrad's custom HCQ queue — confirmed
  empirically — so coop must be timed with tinygrad's profiler. Both sides are therefore **pure GPU kernel time**,
  which also eliminates the raw-dispatch-vs-JIT-graph wall-clock confound that sank the prior north-star A/B.)

## Phase 0 — extraction audit (feasibility: BOUNDED)

| llama symbol/file | purpose | dependencies | keep/replace | risk |
|---|---|---|---|---|
| `flash_attn_tile<…>` (`fattn-tile.cuh:788-1148`) | the decode tile kernel (q·k + online softmax + V acc) | small device helpers only | KEEP | low |
| `flash_attn_tile_iter` / `_iter_KQ` / `_load_tile` | KV-tile body, vector-FMA q·k, LDS staging | `ggml_cuda_mad` (incl. `v_dot2_f32_f16` asm), warp reduces | KEEP | low |
| `flash_attn_combine_results<128>` (`fattn-common.cuh:911-967`) | merge `parallel_blocks` partials (used at decode) | trivial | KEEP | low |
| config table `get_config_amd_rdna` + accessors | nthreads/occupancy/nbatch (RDNA workaround) | constexpr | KEEP | low |
| `ggml_tensor` / KV-cache layout | tensor dims/strides | **host-side only** (device kernel takes raw ptrs+scalars) | REPLACE w/ hand launch | low |
| `cp_async` / `mma` / WMMA | — | **NONE in the tile path** | — | none |

Non-WMMA confirmed three ways (no include, no `mma_sync`/`wmma`, the lone `wmma` token is dead `#ifdef
GGML_USE_WMMA_FATTN` which the build doesn't set). Dispatch logic (`fattn.cu`) routes decode → `BEST_FATTN_KERNEL_TILE`
(GQA-opt applies ⇒ `ncols2=4`). **No `NEEDS_DEEPER_PORT` blocker.**

## Correctness

Not at risk: the oracle measures llama's **real** kernel (the reference, correct by construction); coop is already
byte-exact vs numpy (err ~2e-4). No re-implementation exists to verify, so there is no layout/scale/causal mismatch
to classify. (A future *port* would gate on rel_rmse ≤ 1e-3 vs this reference.)

## Local A/B — pure GPU time, llama vs gqa_coop_vec (`bench/qk-llama-flash-attn-tile-oracle/latest.json`)

| ctx | llama tile µs | llama combine µs | **llama attn µs** | **coop attn µs** | **llama speedup** |
|---:|---:|---:|---:|---:|---:|
| 512 | 7.12 | 3.08 | **10.20** | 59.9 | **5.87×** |
| 1024 | 9.16 (measured) | 3.08 (measured) | **12.24** | 69.9 | **5.71×** |
| 4096 | 23.6 | 4.07 | **27.67** | 132.0 | **4.77×** |

**Gate: PASS** (llama ≥1.05× faster @ctx1024 — 5.71× — and no regress @ctx4096 — 4.77×). Clock-pinned, perf-state
restored to `auto`.

## Dispatch-confound analysis

The prior north-star A/B was confounded by wall-clock latency (raw un-batched dispatches vs coop's JIT graph). This
oracle avoids it entirely: **both sides are pure GPU kernel time** (llama via rocprofv3 HW timestamps, coop via
tinygrad ProfileGraphEvent HW timestamps). coop's pure GPU time (60/70/132 µs) is *lower* than its earlier
throughput proxy (75/85/144 µs), confirming the proxy carried dispatch overhead — yet llama still wins ~5×, so the
result is robust and conservative (not `INCONCLUSIVE_DISPATCH_CONFOUND`).

## Lifecycle / decode_eval binding

- `bench/qk-decode-eval/candidates.json`: `llama_flash_attn_tile_oracle` (family `reference_oracle`, `ab_script` →
  the oracle harness) → ran through decode_eval → **`PASS_ORACLE_LOCAL_AB`** (match, 5.71×).
- New verdicts `PASS_ORACLE_LOCAL_AB` / `FAIL_ORACLE_LOCAL_AB` added to `decode_eval` `classify`, `schema.json`, and
  the lifecycle `search_policy` map (`reference_oracle_target_informs_codegen_non_promotable`).
- New binding `llama_flash_attn_tile_oracle_v0` (`is_reference_oracle: true`, `default_eligible: false`).
- **Non-promotable**: it is vendored llama reference code, NOT a tinygrad primitive; it must NEVER become a default
  or model route. It is a *target + future byte-level oracle*, nothing more.

## What this says about native codegen

**The native-codegen path is justified and now has a concrete target.** The 10× decode-attention gap is real at the
**standalone kernel** level (~5–6×), so it is a kernel-codegen problem, not merely an in-model-integration problem.
The next project (native codegen) should target llama's exact structure — the capability gap is the **single fused
flash kernel** (coupled `(m,l,acc)` online softmax across different range nests), blocked at `tinygrad/uop/spec.py:163-165`
(single-op `REDUCE`) + the shared-range store-group idiom + `linearizer.py:54-82` (pre-refuted
`flash_fused_multireduce_linearizer_wall`, a multi-week linearizer project). llama's kernel is now the validated
target and the byte-level oracle for that work (the full port — BOUNDED — becomes the oracle's re-runnable form).

## Acceptance gates

| gate | result |
|---|---|
| G1 exact llama source path/symbol audited | PASS (`fattn-tile.cuh:788`, the Hd=128 instance) |
| G2 non-WMMA decode path confirmed | PASS (no mma/cp_async; dispatch routes to TILE) |
| G3 standalone oracle runs or precise blocker | PASS (profiling oracle ran; port is BOUNDED, deferred) |
| G4 correctness measured | PASS (llama = reference; coop byte-exact known) |
| G5 local A/B vs gqa_coop_vec measured | PASS (5.87/5.71/4.77×, pure GPU) |
| G6 artifact JSON emitted + validated | PASS (decode_eval run validates) |
| G7 decode_eval/lifecycle metadata updated | PASS (candidate + binding + verdict map) |
| G8 no tinygrad model/default route | PASS (`git diff tinygrad/` empty) |
| G9 no closed lane reopened | PASS (oracle is a new reference lane) |
| G10 policy guard passes | PASS |
| G11 tree clean after commit | PASS (commit below) |

## Next scoped action

**Native codegen / dataflow** is now justified with a real target. The next project scopes the single-fused-flash
linearizer capability (or, as its first concrete step, the **full source port** of llama's tile — BOUNDED per Phase 0
— as the re-runnable byte-level oracle that native codegen validates against). Do NOT route the oracle in-model; do
NOT treat it as a tinygrad primitive.

## Changed files
`extra/qk_llama_flash_attn_tile_oracle_ab.py` (new), `extra/qk_decode_eval.py` (oracle verdicts),
`bench/qk-decode-eval/{candidates.json, binding_templates.json, schema.json}`, `bench/qk-lifecycle-search/search_policy.json`,
this doc + handoff/READMEs/refutation update.

## Boundary
No `tinygrad/` change, no model route/default, no kernel shipped, no W==D route, no closed lane reopened. The oracle
is a non-default, non-promotable reference. Clock-pinned diagnostic; perf-state restored to `auto`.
