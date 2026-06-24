# Route B B1 — Vendored llama tile via tinygrad HCQ: local A/B result

Date: 2026-06-21

Executes **Route B B0–B1** of `docs/decode-attention-route-a-route-b-full-execution-scope-20260621.md` (the
escape-hatch-first de-risk): launch the *vendored* llama decode-attention kernel **through tinygrad's HCQ runtime** and
A/B it vs `gqa_coop_vec`, to answer one question the rocprofv3 oracle cannot — **does the llama-class tile win when
DISPATCHED BY tinygrad's runtime, not just in llama's own trace?** Non-promotable; stops before B2 (W==D) / any model
route / owned kernel.

## Decision: **`PASS_ORACLE_LOCAL_AB` — GPU-kernel win CONFIRMED (2.96×), but W==D is launch-overhead-gated (B2 caveat)**

The vendored llama `flash_attn_tile<128,128,1,4,false>` + `flash_attn_combine_results<128>`, launched via tinygrad's
HCQ on tinygrad Buffers, is **value-correct (rel_max 1.3e-3)** and **wins 2.96× by GPU-busy time** @ctx1024
(llama 15.1µs vs `gqa_coop_vec` 44.6µs) — clears the ≥1.5× de-risk target. **But the same launch path is ~2.5× SLOWER
by WALL time** (148µs vs 58µs): the GPU win is eaten by the per-call launch overhead of **two raw HCQ dispatches**
(tile + combine) vs the comparator's single jit graph. So the escape hatch is worth integrating **at the kernel level**,
but a **W==D win (B2) REQUIRES folding the launches into the model JIT graph** (one dispatch) — the recurring
"isolated kernel wins don't transfer to in-model integration" finding ([[inference-perf-measured-map]]).

## What was built (full capture-and-replay, no `tinygrad/` change)
- **`extra/qk_llama_fattn_kernarg_capture.cpp`** — LD_PRELOAD shim. ggml launches the tile via the chevron
  `kernel<<<>>>` → `hipLaunchKernel` (PDL off; `common.cuh:1639`). Hooks `__hipRegisterFunction` (host-ptr→symbol) +
  `hipLaunchKernel` (geometry + per-arg `void**` values). Three real bugs resolved en route:
  (1) **versioned-symbol interposition** — `libggml-hip` imports `hipLaunchKernel@hip_4.2`; an unversioned preload
  can't preempt it → a `--version-script` exporting the `hip_4.2` node; (2) **static-init crash** —
  `__hipRegisterFunction` fires from librocblas's `_dl_init` *before* the shim's global ctors → construct-on-first-use
  maps; (3) **warmup masking** — keyed capture per distinct KV length so llama-bench's short warmup (KV 256) doesn't
  block the real `-d 1024` decode (KV 1280). Captured: `flash_attn_tile<…,1,4,0>` grid(1,20,8)/block(32,4,1),
  parallel_blocks=20, scale=1/√128, GQA 32/8, fp16 K/V, 37 args fully decoded → `bench/qk-llama-hcq-tile/capture_decode_ctx1024.json`.
- **`extra/qk_llama_flash_attn_tile_hcq_ab.py`** — HCQ replay + A/B. `NamedAMDProgram` (Tensile precedent
  `qk_tensile_hcq_launch.py`) loads the on-disk gfx1100 `.co` (already extracted, bare ELF), resolves both `.kd`
  symbols via symtab walk, rebuilds the **464-byte kernarg**: 37 explicit args at AMD-ABI offsets + the **COV5 hidden
  block** (`hidden_block_count`/`group_size`/`grid_dims` at metadata offsets 208/220/272 — tinygrad doesn't populate
  these, but the vendored kernel reads `gridDim.y`=parallel_blocks; offsets decoded from the `.co` `amdhsa.kernels`
  msgpack), patches the 8 pointers to tinygrad Buffers, launches tile + combine (dynamic LDS = parallel_blocks×8 for
  combine). Buffers laid out byte-identically to the captured ggml strides; numpy GQA-softmax reference.

## Measurements (clock-pinned, @ctx1024 decode shape, KV 1280-padded / 1024-valid)
| metric | llama tile+combine (HCQ) | gqa_coop_vec | ratio |
|---|---|---|---|
| **GPU-busy** (signal timestamps / ProfileGraphEvent) | **15.1 µs** | 44.6 µs | **2.96× llama-faster** |
| wall (per call) | 148 µs (2 raw HCQ dispatches) | 58 µs (1 jit graph) | 0.39× — llama **slower** |
| correctness vs numpy | rel_max 1.3e-3 (fp16 K/V) | — | OK |

- **GPU-busy is the kernel-win authority** (the de-risk's question); **wall is launch-overhead-bound** and is NOT the
  gate — but it is the decisive B2 signal. The 2.96× is below the oracle's standalone "5.7×" partly because this is a
  fairer in-runtime comparison and llama processes the padded KV=1280 (vs coop's 1024) and pays the combine.
- Artifact: `bench/qk-llama-hcq-tile/latest.json` (stamped, CONFORMS); `decode_eval --candidate
  reference_oracle_hcq_llama_tile` → **`PASS_ORACLE_LOCAL_AB`** (non-promotable by construction; family short-circuit).

## Implication for the route decision
- **Capture-and-replay is feasible** — the NEEDS_DEEPER_PORT risk did **not** materialize: the wide (39-arg / 464-byte
  / fastdiv / strides) kernarg is handled by *capture* (record real bytes) + *VA-patch* + *COV5-hidden-fill*, not a
  broad ggml runtime port. A llama-class decode tile runs correctly under tinygrad's runtime.
- **The kernel genuinely wins on GPU time** → the escape hatch (B3 owned hand-AMDGCN, or B2 vendored W==D) is aiming at
  a real ceiling.
- **The gating risk shifts to launch integration.** Before authoring an owned kernel (B3) or claiming a W==D win (B2),
  the launches MUST be graph-integrated (tile+combine folded into the model JIT graph / a single dispatch) — otherwise
  the per-call overhead (~2× the GPU time) inverts the win. **B2's first sub-question is now precise: can the HCQ tile
  launches be folded into the decode JIT graph so the 2.96× GPU win survives to whole-decode W==D?**

## Boundary / guardrails honored
Vendored kernel **non-promotable**; no W==D, no model/default route, no owned kernel in this task. No `tinygrad/` change
(shim + replay harness + JSON registration + docs only). No closed-lane reopen (WMMA decode / MMVQ / FLASH_L / fused
tail / matmul-PV / warp tile). Local A/B is GPU-time diagnostic, **not** a benchmark headline. `gqa_coop_vec` is the
comparator SSOT. Next: B2 (vendored W==D, owner-gated) only if the launch-integration question above is answered;
otherwise the de-risk already tells the owner the kernel-level win is real and the integration cost is the true gate.
