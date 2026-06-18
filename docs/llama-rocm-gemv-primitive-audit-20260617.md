# llama.cpp ROCm GEMV (MMVQ) primitive audit — Qwen3-8B Q4_K/Q6_K decode (2026-06-17)

Audit only, no kernel built. llama build b9592, ROCm 7.2.4, gfx1100 (RX 7900 XTX). Every claim cites file/fn.

## Batch-1 decode GEMV path

- Dispatch: `ggml/src/ggml-cuda/mmvq.cu` `get_vec_dot_q8_1_cuda(type)` → per-quant `vec_dot_*_q8_1`
  (`GGML_TYPE_Q4_K → vec_dot_q4_K_q8_1`, `GGML_TYPE_Q6_K → vec_dot_q6_K_q8_1`). Batch-1 decode (ne[1] small)
  uses **MMVQ** (mul_mat_vec_q), not MMQ. [measured: source]
- **The dot is `ggml_cuda_dp4a`** (`vecdotq.cuh` lines ~126/150/184/.../377 for Q4_K/Q2_K/Q6_K etc.): on RDNA3
  `ggml_cuda_dp4a` → **`__builtin_amdgcn_sdot4`** (`common.cuh:697`) — the hardware **packed int8 dot (v_dot4,
  4 INT8 MACs/instruction)**. So Q4_K and Q6_K decode dots are int8 dp4a. [measured: source]
- **Activations are quantized to q8_1 (int8) ONCE per source activation** and reused across all linears reading
  it (the MMVQ kernel takes a q8_1-quantized `y`; quantization is a separate `quantize_row_q8_1` pass on the
  activation, not per-weight-tile). [inferred from MMVQ taking q8_1 `y` + the single quantize pass]
- Dequant strategy: weights stay quantized; only the **scales/mins** are applied in fp after the int dot
  (`d8[i] * (dp4a(vi,u) * sc - ...)`, vecdotq.cuh ~377) — i.e. int dot first, fp affine on the small sums. No
  per-weight fp dequant. [measured: source]
- Block/layout: K-quant super-blocks (256 weights); `get_int_b4`/`get_int_b2` read packed nibbles as int32
  words; no transpose/repack at runtime (uses the GGUF K-quant layout directly). [measured: source]
- Kernels per linear role: **1** MMVQ kernel per linear (+ the shared activation quantize, amortized across
  same-input linears). lm_head is the same MMVQ path at a large output (vocab). gate/up and q/k/v are
  **separate** linears (no weight fusion); the win is the shared q8_1 activation, not fused weights. [inferred]

## The one concrete structural difference vs tinygrad

| | llama (MMVQ) | tinygrad (shipped Q4_K primitive) |
|---|---|---|
| dot | **int8 dp4a** (`__builtin_amdgcn_sdot4`, ~1.35 VALU/weight) | **fp dequant + scalar fp dot** (~4.06 VALU/weight; `q4_k_gemv_primitive.py:45-61`) |
| activation | quantized to **q8_1 once**, reused across same-input linears | fp16 (default); the gated `Q4K_VDOT` path quantizes q8_1 **per-linear** (`model.py:142`) |
| affine | int dot → fp scale/min on small sums | per-weight fp `d*sc*q − dmin*mn` |

tinygrad HAS a dp4a path (`Q4K_VDOT=1` → `q4k_q8_1_vdot_builtin_partial_kernel`, `__builtin_amdgcn_udot4`,
cstyle.py:398) but it is default-OFF and only fires for parts==1 roles.

## Why the dp4a gap does NOT cash out in tinygrad (measured, this session)

`Q4K_VDOT=1` in-model W==D (the trusted method): ctx128 49.3→49.8 (+1.0%), ctx512 37.0→37.3 (+0.8%) — **null**,
despite the dp4a kernel being ~1.77× faster standalone (prior `dp4a-d0`). Reasons:
1. **Per-linear q8_1 activation quant** (`model.py:142`): q/k/v share the attn-norm output, gate/up share the
   ffn-norm output, but `Q4K_VDOT` re-quantizes per linear → 3×/2× redundant quant overhead that offsets the
   dot savings. llama amortizes (quantize once). [inferred — the structural difference + null e2e]
2. **Partial coverage**: only 163/199 linears are parts==1 (vdot-eligible); the 36 split-K roles (the big
   ffn_down/lm_head) stay fp. [measured]
3. The earlier "dp4a null = decode is latency-bound" claim was refuted (decode is GPU-bound, W==D); yet e2e is
   STILL null → the blocker is the quant overhead + coverage, not launch latency. [measured]
