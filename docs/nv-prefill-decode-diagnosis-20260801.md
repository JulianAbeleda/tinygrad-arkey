# NV prefill vs decode: why prefill is ~1% of its roofline

Date: 2026-08-01

Investigation on the RTX 5090 (sm_120), Qwen3-8B-Q4_K_M, branch `nvidia-bringup-20260731`. Question:
why is prefill slow on NV, what path does each phase actually take, and where is the recoverable
time. No code changed; this doc records measurements and the root cause.

## 1. BoltBeam ceilings (fresh NV run, 2026-08-01)

Run dir: `/home/ubuntu/boltbeam-runs/llama-8b-nv-20260801/`. Hardware facts used: 1792 GB/s,
180 TFLOPS.

- **Decode is bandwidth-bound.** Raw ceiling 383.58 tok/s (2.607 ms/token, 4.67 GB packed
  weights/token). Compute ceiling would be 11,892 tok/s; it is irrelevant.
- **Prefill pp512 is compute-bound.** Floor 37.47 ms -> ceiling 13,664 tok/s. Role split:
  `ffn_gate_up` 55.0%, `attn_qo` 18.3%, `ffn_down` 13.8%. **This role split is BoltBeam's
  modeled FLOP-share, not a measurement.** The measured kernel-share is section 3; the two
  agree on ranking, not on magnitude (BoltBeam's model has no dequant cost, which is where
  the scalarized GEMMs actually spend their time).

## 2. Measured NV vs the ceilings

| metric | measured | ceiling | % of ceiling |
| --- | ---: | ---: | ---: |
| decode | 158.2 tok/s (764.8 GB/s) | 383.6 tok/s | 41% |
| prefill pp512 | ~101-115 tok/s (ttft 5.07s incl. JIT capture; warm steady 4.34-4.54s) | 13,664 tok/s | ~1% |

AMD reference on the same codebase (`boltbeam-runs/qwen3-8b-current-20260713`): prefill measured
3,881 tok/s, decode 117.1 tok/s. **NV prefill is ~34x behind AMD on the same model and code path
shape.** Same-session AMD llama reference (`docs/prefill-current-state.md`, 2026-07-24,
`llama-bench -fa 1 -ngl 99`): 8B pp512 = 3,347 +/- 242 tok/s, ours 3,727 (+11.4%).

Bench artifact: `/tmp/qwen3-8b-p5-final2.json` (decode median 158.21, prefill 101.0,
`prefill_overlay_promotion: "no-promoted-candidate"`, first tokens byte-identical to the SDPA
baseline).

**Paired llama.cpp baseline (same machine, same model, same session, CUDA build
`ac4cddeb0`, `llama-bench -ngl 99`, verbose run confirms `n_layer = 36` all on CUDA0).**

Prefill sweep at depth:

| context | llama.cpp tok/s | tinygrad NV tok/s | BoltBeam ceiling tok/s |
| --- | ---: | ---: | ---: |
| pp128 | 7,733 | - | - |
| pp256 | 11,542 | - | - |
| pp512 | 14,250 | ~101-115 | 13,664 |
| pp1024 | 14,633 | - | - |
| pp2048 | 14,342 | - | - |
| pp4096 | 13,801 | - | - |

Decode at depth (gen 10, `-d`):

| depth | llama.cpp tok/s | tinygrad NV tok/s |
| --- | ---: | ---: |
| d512 | 237.1 | 158.2 |
| d2048 | 225.7 | - |
| d4096 | 217.0 | - |

Prefill is the 100x+ gap and it is the same root cause as section 4: llama's Q4_K GEMMs reach
the tensor/int-dot unit (q8_1 quantize + int8 `dp4a` MMQ, zero `mma` kernels - traced, campaign
scope section 1a); ours stay scalarized. Decode is a smaller 1.6x gap inside the same
bandwidth-bound regime (their per-token GEMV kernels are more efficient).

## 3. What path each phase takes

**Decode — flash-decode route (`rollout_jit_flash`, `use_flash=True`).** Uses CUDA `fdot2` /
`warp_shfl_xor` providers; per-token kernels `q4k_g3_lanemap_gemv_*`, `q6k_gen_coop_*`,
`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128`. Healthy: 41% of a bandwidth ceiling
that is a hard floor for packed Q4_K.

**Prefill — `prefill_v2` + concrete-KV + fused custom-kernel attention + dense Q4_K and Q6_K
GEMM routes.** The fused attention kernel (`nv_sm120_q16_grid_hd128_loop_attention`) is *not* the
problem: 46 TFLOPS, ~98us/call, 36 dispatches = 3.6ms, **0.4% of traced GPU time**. The problem
is the six scalarized GEMM routes that dominate prefill:

| kernel | role | calls | total | per call | share |
| --- | --- | ---: | ---: | ---: | ---: |
| `r_16_64_8_16_4_4_48_2_2_2_16_2` | ffn_down Q6_K | 18 | 319.1ms | 17.7ms | 33.2% |
| `r_16_256_8_16_4_3_16_4_2_8_4` | ffn_gate_up Q4_K | 72 | 302.3ms | 4.2ms | 31.5% |
| `r_16_64_8_16_4_4_48_4_2_16_2` | ffn_down Q4_K | 18 | 140.3ms | 7.8ms | 14.6% |
| `r_16_64_8_16_4_4_16_4_2_16_2` | attn_qo Q4_K (q+o) | 72 | 129.1ms | 1.8ms | 13.4% |
| `r_16_16_8_16_4_4_16_2_2_2_16_2` | attn_v Q6_K | 18 | 27.9ms | 1.55ms | 2.9% |
| `r_16_16_8_16_4_4_16_4_2_16_2` | attn_k Q4_K + attn_v Q4_K | 54 | 25.4ms | 0.47ms | 2.6% |
| `nv_sm120_q16_grid_hd128_loop_attention` | fused attention | 36 | 3.6ms | 0.10ms | 0.4% |
| `r_1187_32_4_16_2_2_2_4_8` | lm_head (output Q6_K) | 1 | 2.9ms | 2.9ms | 0.3% |

Source: `/tmp/nv_kernel_times.log` (TIMING=2, 2,041 kernel lines), re-aggregated 2026-08-01.
**Correction to the first edition of this table**: the earlier share column (47.1/21.9/20.1,
summing to "89%") was computed against a 641.9ms subtotal that silently excluded every
ms-denominated kernel. The `ffn_down Q6_K` route logs `tm 17.63ms/...` and was parsed as ~0;
it is the single most expensive kernel in the trace. Correct shares use the full 961.0ms
total. Q4_K routes are 62.1% of traced time; **Q6_K routes are 36.4%** (ffn_down Q6_K 33.2% +
attn_v Q6_K 2.9% + lm_head 0.3%) - the Q6_K share was previously estimated from llama's 13%
and is now measured from our own trace. Role labels were derived from the GGUF tensor quant
layout (36+36 Q4_K gate/up, 18+18 Q4_K/Q6_K ffn_down, 18+18 Q4_K/Q6_K attn_v, 36+36 attn_q/o,
Q6_K output) plus the route-plan policy `(parts, opts)` signatures.

## 4. Root cause: the prefill GEMMs are scalarized

I pulled the generated CUDA for the gate_up route out of the JIT cache
(`compile_nvcc__sm_120_e3b0c442_22`, key containing `r_16_256_8_16_4_3_16_4_2_8_4`). It is **fully
scalarized**: no `mma.sync`, no `dp4a`, no `half2`, no vector loads. The kernel dequantizes Q4_K
in registers (byte shifts/masks), casts to float, and accumulates with scalar `float` FMAs
(`float buf0[48]`, 12 scalar half->float products per inner step). That is CUDA-core scalar code,
not tensor cores: ~6-24 TFLOPS on a card whose tensor-core peak is ~180 TFLOPS. All six GEMM
routes use the same scalarized lowering (the Q6_K routes are the slowest per byte: 17.7ms/call
against ffn_down Q4_K's 7.8ms). 98.2% of traced GPU time is in these six routes.

## 5. Why NV fell onto the scalar path while AMD did not

AMD's hot prefill kernel (`E_4_96_32_4_2_2_2_2_4_2_127_2`, 736us/call for the same gate_up shape)
is a **promoted BoltBeam WMMA full-kernel candidate**. The HIP cache key starts
`candidate:boltbeam.full_kernel_candidate.v1:7e37ad6e13de...`, and `7e37ad6e` is the
`ffn_gate_up` legacy identity in
`tinygrad/llm/generated/prefill_wmma_lds_dbuf_candidate_set.json`; the source has 177
`__builtin_amdgcn_wmma` references. AMD prefill runs the promoted tensor-core candidate set.

That promoted artifact is **AMD gfx1100-only** (`profile: qwen3_8b_q4k_m_gfx1100`,
`target: {AMD, gfx1100, wave32}`, `wmma_f32_16x16x16_f16`), and `promoted_candidate_set()`
(`tinygrad/llm/prefill_candidate_runtime.py:157`) pins it with
`_PINNED_COMPACT_ARTIFACT_TARGET` — non-AMD targets raise "compact target is unsupported". A CUDA
sm120 candidate set already exists in the repo
(`bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-sm120-v1/candidate-set.json`, 4
entries covering all 4 roles, `cuda_mma_f32_8x16x16_f16_lds2_static`, `wmma_f32_8x16x16_f16`) but
it is **not the promoted artifact**, so `automatic_promoted_prefill_graph_policy` returns None on
NV and the census records `prefill_overlay_promotion: "no-promoted-candidate"`. Prefill then runs
the ordinary tinygrad linear path, which emits the scalarized kernels in section 4.

## 6. Lowering is not the blocker anymore

The two gates that once blocked the NV-typed schedule are fixed by `948b26318` (C5): the
accumulator carrier `vec(8)` vs `vec(4)` mismatch (`kernel_pipeline.py:181`) and the CUDA operand
lane-layout derivation (`kernel_lds.py:171`). Both NV buffer shapes (buffer2 bc=2 active LDS
40960; buffer1 bc=1 active LDS 20480) now admit, compile, dispatch, and measure with
`max_abs_error 0.0`, bit-identical across three rounds, shared accesses inside declared
allocations, 96.67% coverage under the zero-init lower bound. Recorded in
`docs/bringing-up-a-new-target-20260731.md` §9: "The next NV slice is promotion work, not
lowering: bench-row/census wiring for the measured buffer kernels and the remaining quant
routes."

## 7. Caveat on the wall-time split

Warm prefill wall time is ~4.3s, while the cold trace sums to ~961ms of GPU kernel time. The
cProfile of a warm run (`/tmp/nv_exec_profile.log`) shows the host blocked in GPU `wait()` for
4.3s of 4.39s, and nvidia-smi showed 100% util bursts: the warm path is launch/sync-bound on top
of the slow kernels (~2,000 kernels per iteration with per-kernel gaps). Both halves point the
same direction: replace the scalarized GEMM path with the tensor-core candidates that already
exist for sm120.

## 8. Fix direction (not implemented)

Promote the sm120 candidate set into the production graph-GEMM path so NV prefill runs the
`cuda_mma` kernels the same way AMD runs its WMMA candidates: per-target promoted artifact
selection, census/bench-row wiring, correctness (first-token digits) and AMD control re-runs,
then a measured head-to-head. Scoped in
`docs/task_workflow/input/nv-prefill-gemm-promotion-scope-20260801.md`.
