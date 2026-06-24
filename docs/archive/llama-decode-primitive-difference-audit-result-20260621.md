# Llama Decode Primitive Difference Audit — Result

Date: 2026-06-21

Scope: `docs/llama-decode-primitive-difference-audit-scope-20260621.md`

## Decision

**`WMMA_FLASH_DECODE` is REFUTED.** llama's decode attention is **not** a tensor-core/WMMA body. The decode gap is
**dominantly attention**, but via a **non-WMMA vector `flash_attn_tile`** whose advantage is *parallel
decomposition + LDS staging*, not matrix hardware. The audit therefore returns a **corrected verdict:
`VECTOR_FLASH_DECODE_TILE` (fundable, non-WMMA)** — attention is the gap and is actionable, but the next project
is a high-occupancy vector flash-decode tile, **not** a WMMA build and **not** `REST_DECODE`. `MMVQ_REOPEN` is
rejected (wall parity); `TOOLING_BLOCKED` is rejected (full visibility achieved). Default decode behavior NOT
changed; no kernels built (audit only).

## 1. Source primitive inventory (Phase 1)

| role | llama primitive (source) | tinygrad primitive | status |
|---|---|---|---|
| Q4_K/Q6_K matvec | `mul_mat_vec_q` + `vec_dot_q[46]_K_q8_1`, **int8 `dp4a`→`__builtin_amdgcn_sudot4` (V_DOT4)** (`mmvq.cu`,`vecdotq.cuh`,`common.cuh:698`) | native Q4_K/Q6_K coop/gemv GEMV (fp16 path) | **wall parity** (+0.24 ms) |
| q8_1 activation producer | `quantize_q8_1` once/matvec into a pool buf, reused; gate+up **fused** (one q8 feeds both) (`mmvq.cu:1186`,`ggml-cuda.cu:4118`) | per-op activations; q8 opt-in only | structural diff, not wall gap |
| decode flash attention | **`flash_attn_tile<128,128,1,4>` — pure vector `ggml_cuda_mad` FMA** (`fattn-tile.cuh`) | `gqa_coop_vec` (6 UOp kernels) | **the gap (attention)** |
| stream-k / fixup / combine | `flash_attn_combine_results` (KV-split merge); stream-k fixup only on the MMA path (not decode) | flash_gmax/den/combine | llama merges more splits |
| RMSNorm / RoPE / residual | `norm.cu` / `rope.cu` / `binbcast.cu` (separate ops) | separate UOp kernels | tinygrad rmsnorm faster |
| graph lifecycle | **HIP graph** captured once, replayed/token (`ggml-cuda.cu:4278`) | TinyJit HCQ graph replay | equivalent |

## 2. Refreshed llama decode runtime trace (Phase 2 — rocprofv3 kernel-trace, decode-only, per token)

| ctx | decode ms/tok (GPU) | mmvq (weight-GEMV) | attention (tile+combine+fixup) | rmsnorm | rope | q8-quant | other |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 9.63 | **7.78 (80.7%)** | 0.41 (4.3%) | 0.53 | 0.28 | 0.35 | rest |
| 1024 | 9.97 | **7.69 (77.1%)** | 0.51 (5.1%) | 0.61 | 0.37 | 0.34 | rest |
| 4096 | 12.91 | **7.69 (59.6%)** | 1.23 (9.5%) | 1.22 | 0.90 | 0.34 | rest |

Resource shape (rocprofv3, exact): `flash_attn_tile` LDS **10752 B**, VGPR **128**, workgroup 32×4 (128 threads),
grid scales with ctx (**parallel_blocks 48 → 80 → 144** per the per-call traces). `mul_mat_vec_q` LDS 0/512,
VGPR 24-40, wg 32×1/32×2. WMMA `flash_attn_ext_f16` (LDS scratch, VGPR 256, MI16×16) appears **only n=72 =
prefill**, never decode. **Tooling is sufficient** — LDS/VGPR/SGPR/scratch/grid/workgroup/timing all captured.

## 3. Attention primitive diff (Phase 3) — the central question answered

| feature | llama decode (`flash_attn_tile`) | tinygrad `gqa_coop_vec` | implication |
|---|---|---|---|
| tensor-core / WMMA | **NO** (rocWMMA off by default `CMakeLists.txt:219`; T=1 below threshold `4>8` false `fattn.cu:519`; vector `ggml_cuda_mad`) | no | **WMMA_FLASH_DECODE refuted** |
| per-layer time @1024 | **~9.2 µs** | ~97 µs | **~10× faster** |
| KV-split parallelism | **parallel_blocks = 48/80/144** (scales with ctx) | **fixed 8** (FLASH_L=128) | tinygrad occupancy-starved at T=1 |
| block decomposition | block-per-kv-head(8) × parallel_blocks × 4 query-heads packed in `ncols2` | 8 kv-heads × 8 splits ≈ 64 blocks | llama fills GPU (8×48-144 blocks) |
| query-head parallelism | 4 heads packed in columns, all kv-heads parallel | consolidated to 8 kv-heads | both reuse K/V, llama keeps more parallel work |
| q·k mapping | once per (key,column), vector FMA from LDS, no redundancy | matmul (also once) | parity on redundancy |
| P·V mapping | vector FMA accumulate from LDS | coop V-reuse | parity-ish |
| online softmax | registers + small cross-warp `__shared__` | split across kernels | llama keeps it in-kernel |
| LDS K/V staging | **yes, ~10.7 KB tile, loaded once/step, reused across columns** | minimal | the occupancy+reuse combination |
| ctx-slope | flat-ish (parallel_blocks absorbs KV growth) | grows 22→32% of wall | the slope gap is the fixed-split starvation |

**Verdict:** llama's decode-attention win is a **well-engineered vector tile**: it fills the GPU at T=1 by
splitting the KV dimension into many parallel blocks (48-144, scaling with ctx) while staging K/V once in LDS and
packing the 4 GQA query heads into columns. It is **not** a tensor-core trick. tinygrad's `gqa_coop_vec` uses a
**fixed 8-way split (~64 blocks)** → occupancy-starved at T=1, ~10× slower per layer. The earlier fused-tile
refutation consolidated the **wrong way** (to kv-heads, serial G) — this is **new evidence** that the actionable
structure is *more* KV-split parallelism, not less.

## 4. MMVQ primitive diff (Phase 4) — do NOT reopen

| role | tinygrad | llama | gap | reopen? |
|---|---|---|---|---|
| all weight-GEMV (decode) | 7.93 ms/tok @1024 (558-863 GB/s, Del 0) | 7.69 ms/tok (int8 dp4a + q8 reuse) | **+0.24 ms (parity)** | **NO** |

llama's MMVQ is a *structural* body advantage (native int8 `sudot4`/V_DOT4 over 4/6-bit weights + q8_1 activation
reuse + gate/up fusion), but it does **not** translate to a wall-time gap: both paths are HBM-bandwidth-bound on
the weight read at similar effective GB/s, so tinygrad's fp16-path GEMV is already at llama parity in ms/token
(confirmed Del 0, refreshed here). Reopening MMVQ would chase a structural curiosity, not the gap. **Closed.**

## 5. Tooling gap report (Phase 5) — NOT blocked

| needed observation | tool | status |
|---|---|---|
| kernel name → role | demangled symbols + counts | OK |
| LDS/VGPR/SGPR/scratch/grid/workgroup | rocprofv3 `--kernel-trace --output-format csv` | **OK (all columns present)** |
| per-kernel timing/duration | rocprofv3 timestamps | OK |
| attention body / dataflow | llama source (`fattn-tile.cuh`, `fattn.cu`) | OK |
| per-kernel instruction mix / PC-stall | rocprofv3 counters return 0 on HIP (multiplex) | missing, **but not needed** — resource shape + grid + source body already name the difference |

The difference is fully observable: the attention gap is the grid/parallel-block decomposition + LDS staging,
visible directly in the trace + source. **Not TOOLING_BLOCKED.**

## 6. Decision and next scope

**`VECTOR_FLASH_DECODE_TILE` (corrected from the refuted `WMMA_FLASH_DECODE`).** The decode gap is attention
(+3.0 ms/tok @1024 ≈ 64% of the gap, growing with ctx), and the actionable primitive is a **non-WMMA vector
flash-decode tile** matching llama: **(a) many KV-splits (parallel_blocks scaling with ctx, ~48-144) to fill the
GPU at T=1**, **(b) LDS K/V tile staged once and reused**, **(c) GQA query-head column-packing (keep query-head
parallelism, not kv-head consolidation)**, **(d) register online-softmax + a many-split combine.** This directly
attacks the occupancy starvation that the prior fused-tile attempt got backwards.

- It is **not** WMMA (no tensor-core codegen needed) → more tractable than the WMMA hypothesis.
- It is **not** a reopen of a refuted lane without evidence: the refuted fused-tile consolidated to kv-heads with
  fixed/too-few splits; the **new trace evidence** shows the win is *more* KV-split parallelism + query-head
  packing — the opposite structure.
- **Recommended follow-up:** an implementation scope for a high-occupancy vector flash-decode tile, with a hard
  first gate — *a standalone decode-shape tile with parallel_blocks ≫ 8 + LDS staging must beat `gqa_coop_vec` by
  ≥1.05× @ctx1024 before any integration*; if it cannot (e.g. the UOp combine over many splits or the LDS body
  can't match llama's vector FMA), then `REST_DECODE`.

If the project does not want to fund another attention build, the fallback is `REST_DECODE` (~67% llama steady,
q8 opt-in) — but the audit has now **named a concrete, non-WMMA, observable, evidence-backed lever**, so this is
not the "nothing remains" case.

## Artifacts

- `bench/qk-llama-decode-primitive-audit/decode_kernel_trace.json` (per-ctx decode breakdown + resource shapes + findings)
- `bench/qk-llama-decode-primitive-audit/llama_decode_kernel_trace_ctx1024.csv` (raw rocprofv3 trace)
- lifecycle ledger updated (Phase: WMMA refuted, vector-tile lever opened).

## Constraints honored

No model.py/default change; no codegen; no kernel built. WMMA_FLASH_DECODE claimed **only to refute it** (the gate
required matrix/tensor-core evidence, which is absent). MMVQ not reopened (parity, not launch-count). Launch count
not used as a proxy (timing + resource shape + source body). Clean-wall/rocprof trace at matched ctx.
