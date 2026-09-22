# llama.cpp reference: attention residual and the fp32 block-output buffer in the Qwen3-8B Q4_K_M decode path

Date: 2026-08-11. Source tree: `/home/ubuntu/env/llama.cpp` (master).
Scope: 1-token CUDA decode (the tinygrad NV ground-truth case). No repo code was
modified. Tinygrad side cited from `/home/ubuntu/tinygrad-arkey` (branch
`nvidia-bringup-20260731`). Model: `Qwen3-8B-Q4_K_M.gguf` (n_layer=36,
n_embd=4096, all norms stored F32; `attn_q`/`attn_k`/`attn_output`/`ffn_gate`/
`ffn_up` Q4_K, `attn_v`/`ffn_down` Q6_K -- confirmed from the GGUF tensor
info section).

## 1. Short answers

1. Where is the attention residual add built, and is it absorbed into the `wo`
   GEMV epilogue? **Yes, absorbed like the FFN one.** The add is
   `ffn_inp = ggml_add(ctx0, cur, inpSA)` at
   `src/models/qwen3.cpp:116` (`cur` is the `wo` output, `inpSA` is the block
   input). For blocks 0..34 the add is the immediate successor of the `wo`
   `GGML_OP_MUL_MAT`, so `ggml_cuda_try_fuse`'s generic bias-add branch
   (`ggml/src/ggml-cuda/ggml-cuda.cu:4175-4236`) matches and dispatches the
   `wo` GEMV as fused mmvq with `fusion_data.x_bias = inpSA`
   (`ggml-cuda.cu:4216-4230`). The mmvq epilogue adds the residual row in fp32
   and stores fp32 (`ggml/src/ggml-cuda/mmvq.cu:640-644, 668`).
2. Is there ANY copy/cast/`ggml_cont` of the fp32 block-output buffer between
   blocks? **No.** `qwen3.cpp` contains no `ggml_cpy`/`ggml_cont`/`ggml_cast`
   at all (grep is empty). The block output is passed by node reuse:
   `inpL = cur` (`qwen3.cpp:139`), next block does `inpSA = inpL`
   (`qwen3.cpp:72`). The only `ggml_cpy` calls in `src/llama-graph.cpp` are
   recurrent-state models (L2761, L2845), unrelated to Qwen3.
3. CRITICAL - next block attention input: the attention GEMV input is the
   `attn_norm` output, which is **F32** (RMS norm output type == input type via
   `ggml_dup_tensor`, `ggml/src/ggml.c:3124, 1858-1860`; the CUDA RMS norm
   kernel asserts F32 in/out, `ggml/src/ggml-cuda/norm.cu:484-485`). The claim
   that llama.cpp feeds fp16 into attention GEMVs is **false for this tree**:
   the mmvq fusion path requires `src1->type == GGML_TYPE_F32`
   (`ggml-cuda.cu:2550`, `mmvq.cu:1126`). There is no fp32->fp16 cast of the
   residual stream anywhere; fp16 appears only at the KV-cache store boundary.
4. Attention residual cast: llama.cpp **never** renders an fp32->fp16 cast of
   the residual stream. The `wo` epilogue add output is fp32 (fp32 store,
   `mmvq.cu:668`), and `ffn_norm` consumes that fp32 add node directly via
   RMS norm (`qwen3.cpp:120-123`).
5. Tinygrad: the two remaining kernels are `E_32_32_4_fab82d40...` (49 fp32
   block-output copies) and `E_32_32_4_0a5eb0ac...` (36 fp32->fp16 attention
   casts). llama.cpp implies `fab82d40` is **eliminable entirely** (it is a
   value-transparent fp32->fp32 boundary copy; llama.cpp has zero such copies)
   and `0a5eb0ac` is a **tinygrad ABI artifact**: llama.cpp feeds fp32 into
   attention GEMVs, so the fp16 attention ABI (and its cast) only exists
   because tinygrad's Q4K decode kernels consume fp16 activations. The cast
   cannot be dropped as a no-op (lossy `cvt.rn.f16.f32`), but it can be
   eliminated by switching the Q4K attention input ABI to fp32 (matching
   llama.cpp) or folded into the norm/GEMV epilogue with bitwise-identical
   fp16 bytes. Details in section 7.

## 2. Graph construction (per block)

`src/models/qwen3.cpp:71-139`:

- L72: `ggml_tensor * inpSA = inpL;` - the block input is a **node reuse** of
  the previous block's output (no copy, no view-copy).
- L75-77: `cur = build_norm(inpL, attn_norm, NULL, LLM_NORM_RMS, il);` -
  attention RMS norm over the fp32 block output. `build_norm` lowers
  `LLM_NORM_RMS` to `ggml_rms_norm(ctx0, cur, eps)`
  (`src/llama-graph.cpp:1139`), whose result is `ggml_dup_tensor(ctx, a)`
  (`ggml.c:3124`) - **same dtype as the input (F32)** - followed by
  `ggml_mul(ctx0, cur, mw)` with the F32 norm weight (`llama-graph.cpp:1153`),
  also dtype-preserving (`ggml.c:2229`).
- L83-84: `build_qkv(model.layers[il], cur, ...)`; Qwen3 has separate wq/wk/wv
  (no fused wqkv in the GGUF), so `build_qkv` takes the separate path
  (`llama-graph.cpp:1199-1233`): `Qcur = build_lora_mm(layer.wq, cur, ...)` ->
  `ggml_mul_mat(ctx0, wq, cur)` (`llama-graph.cpp:1201, 1076`). **The Q/K/V
  GEMV input is `cur`, the F32 `attn_norm` output.**
- L108-110: `build_attn(inp_attn, model.layers[il].wo, wo_b, wo_s, ...)`
  (KV-cache overload, `llama-graph.cpp:2285-2358`). Flash attention is on by
  default (`LLAMA_FLASH_ATTN_TYPE_AUTO`, `src/llama-context.cpp:3366, 500`);
  `build_attn_mha` produces an **F32** attention output
  (`ggml.c:5396`, `ggml_flash_attn_ext` result tensor is `GGML_TYPE_F32`),
  reshaped to 2D at `llama-graph.cpp:2104`. The `wo` GEMV is
  `cur = build_lora_mm(wo, cur, wo_s)` (`llama-graph.cpp:2349`) ->
  `ggml_mul_mat(ctx0, wo, cur)` (`llama-graph.cpp:1076`), src1 F32.
- L112-115: last layer only (`il == n_layer-1 && inp_out_ids`):
  `ggml_get_rows` on both `cur` and `inpSA` (token selection).
- L116: `ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);` - **the
  attention residual add node**, dtype F32 (add result is
  `ggml_dup_tensor(ctx, a)`, `ggml.c:2028`).
- L120-123: `build_norm(ffn_inp, ffn_norm, NULL, LLM_NORM_RMS, il)` - the next
  norm consumes the fp32 add node.
- L133: `cur = ggml_add(ctx0, cur, ffn_inp);` - FFN residual add (covered by
  `llama_ffn_residual_reference.md`).
- L139: `inpL = cur;` - the block output node is passed to the next block
  as-is.

No `ggml_cast` exists anywhere in `qwen3.cpp`; the K/V tensors become fp16 only
at the KV-cache store (`cpy_k`/`cpy_v` -> `ggml_set_rows`,
`src/llama-kv-cache.cpp:1323, 1358`, cache type F16 default
`llama-context.cpp:3377-3378`), which is a scatter into the cache, not a
residual-stream cast.

## 3. CUDA fusion: the `wo` GEMV absorbs the attention residual

Same generic machinery as the FFN case. The CUDA graph-compute loop calls
`ggml_cuda_try_fuse` per node (`ggml-cuda.cu:4424`). The bias-add branch
(`ggml-cuda.cu:4175-4236`) matches any `MUL_MAT, ADD` pair where the add's
`src[0]`/`src[1]` is exactly the mul_mat node (`ggml-cuda.cu:4190-4196`) and
both operands are same-shape (`ggml-cuda.cu:4212`). For Qwen3 blocks 0..34 the
`wo` mul_mat's only consumer is the add at `qwen3.cpp:116` (guaranteed by the
post-order graph build, same argument as the FFN doc), so the add is
classified as an `x_bias` and dispatched as
`ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, bias_node, &fusion_data)` with
`fusion_data.x_bias = inpSA` (`ggml-cuda.cu:4216-4227`); the **dst is the add
node** (`ffn_inp`), and the mul_mat node itself is elided.

Eligibility (`ggml_cuda_should_fuse_mul_mat_vec_q`, `ggml-cuda.cu:2541-2572`):
quantized `src0` (Q4_K `wo`), **F32 `src1`** (flash-attn output) and **F32
`dst`** (the add node), `src1->ne[1] <= MMVQ_MAX_BATCH_SIZE`, `dst->ne[1] == 1`
(single token), cc > Pascal, no split buffers. All true here. The host
launcher asserts `src1->type == GGML_TYPE_F32` and `dst->type ==
GGML_TYPE_F32` (`mmvq.cu:1126-1127`) - i.e. the fused attention GEMV path
**requires** fp32 activations.

Kernel epilogue (`mmvq.cu:475-478`, `mul_mat_vec_q` with `has_fusion=true`):
the residual row is prefetched per row (`mmvq.cu:522-542`), and the epilogue
is `result += x_biases[j];` (fp32 accumulator, `mmvq.cu:640-644`) with an fp32
store at `mmvq.cu:668`. This is bitwise the same arithmetic the standalone add
would perform.

## 4. When llama.cpp does render a standalone attention add

Only the last layer: `ggml_get_rows` at `qwen3.cpp:112-115` is inserted
between the `wo` mul_mat and the add, breaking MUL_MAT+ADD adjacency, so
`ggml_can_fuse` (`ggml/src/ggml-impl.h:669-695`) fails and the add falls back
to `ggml_cuda_op_add` (`ggml-cuda.cu:2864-2866`, kernel `k_bin_bcast` with
`op_add`, `ggml/src/ggml-cuda/binbcast.cu:13-34, 393`). Both add inputs
(get_rows results) are fp32; the add computes and stores fp32. This is the
same 1-of-36 standalone add the FFN doc already counted.

## 5. Between blocks: no copies, fp32 buffer passed by node reuse

There is no `ggml_cpy`/`ggml_cont`/`ggml_cast` in the Qwen3 between-block path
(empty grep over `qwen3.cpp`; the only `ggml_cpy` in `llama-graph.cpp` is the
recurrent-state path at L2761/L2845). The block output is the FFN residual-add
node (`qwen3.cpp:133`); `inpL = cur` (`qwen3.cpp:139`) hands that exact tensor
to the next block, which reads it as `inpSA = inpL` (`qwen3.cpp:72`) for both
the attention residual `x_bias` (fused into the next `wo` GEMV) and the next
`attn_norm` RMS input. The fused `wo`/`ffn_down` mmvq writes its result
straight into the add node's fp32 buffer, so the block output is materialized
exactly once, contiguous fp32, and consumed in place. ggml never inserts
implicit copies; buffer reuse across blocks is an allocator detail, never a
copy.

**Attention GEMV input dtype trace (Q3):** `inpL` (fp32) -> `attn_norm`
(`ggml_rms_norm`, output F32 by `ggml_dup_tensor`, `ggml.c:3124/1858-1860`;
CUDA kernel F32-only, `norm.cu:484-485`) -> `build_qkv` -> `ggml_mul_mat(wq,
cur)` with src1 = **F32**. The fp16 in the attention path is confined to the
KV cache (F16 store via `ggml_set_rows`) and flash attention's K/V operands
read from that F16 cache; the Q/K/V and `wo` GEMV activations are F32 end to
end. This is what makes the mmvq fusion possible at all.

## 6. Arithmetic per token (36 blocks, rows=4096, fp32)

Attention residual (`ffn_inp = ggml_add(cur, inpSA)`, `qwen3.cpp:116`):

- 35 absorbed into the `wo` Q4_K mmvq epilogues (`mmvq.cu:640-668`).
- 1 standalone `k_bin_bcast` add (last layer, `qwen3.cpp:112-115` breaks
  adjacency; `binbcast.cu:393`), plus 2 `ggml_get_rows` in that layer.
- fp32->fp16 casts of the residual stream: **0**.

Between-block copies/casts: **0**.

Net: 0 residual casts + 0 between-block copies for the whole 36-block path.

## 7. "So what" for tinygrad

Tinygrad's remaining cost rows (`scratchpad/m2c_arithmetic_validation.md`
section 3): `E_32_32_4_fab82d40...` fp32 copies x49 (~1.66-1.76 us) and
`E_32_32_4_0a5eb0ac...` fp32->fp16 casts x36 (~1.66 us), total ~146 us census.

Where tinygrad makes them (`tinygrad/llm/model.py`, `tinygrad/llm/decode_routes.py`):

- fp32 block output is produced at `model.py:694-698`: the callified block
  returns `ffn_out` (absorbed) or `(h + ffn_out).contiguous()`, and the
  `@function` wrapper forces `.contiguous()` on the call boundary
  (`model.py:698`). These `.contiguous()` calls are the `fab82d40` family
  (ledger role `ffn_residual_add_or_block_output_contiguous`,
  `extra/llm_research/decode/nv_fusion_population_ledger.py:75`).
- the next block's attention norm is emitted **fp16**:
  `_decode_rmsnorm(self.attn_norm, x, _fused_norm, dtypes.float16)`
  (`model.py:628`), whose fallback materializes a standalone cast
  `out.cast(dtypes.float16)` (`model.py:424`); the Q4K GEMV input prelude also
  casts `x[:, 0, :].reshape(binding.K).cast(dtypes.float16).contiguous()`
  (`decode_routes.py:132, 326`). The 36 `0a5eb0ac` casts are this fp16
  attention-input boundary (ledger role `attention_cast`,
  `nv_fusion_population_ledger.py:73`).
- the attention residual for the `attn_output` epilogue is passed as
  `residual_for_output=(x if _epi_residual else None)` (`model.py:644`,
  consumed at `model.py:954-960`, residual cast to fp32 at
  `decode_routes.py:113-118`) - tinygrad already keeps the residual slot fp32
  when it absorbs the attention add, exactly like llama.cpp's `x_bias`.

Explicit verdict:

1. `fab82d40` (49 fp32 block-output copies): **eliminable entirely.** It is a
   pure fp32->fp32 identity copy. llama.cpp proves the correct structure has
   zero copies: the block output *is* the fused GEMV's dst buffer, reused by
   pointer. Tinygrad's copies exist because `.contiguous()` cannot fold over
   the opaque program CALL/AFTER boundary (M2c root cause: "the cast folds,
   the COPY does not"). Folding the boundary is value-transparent
   (`m2c_arithmetic_validation.md` section 4).
2. `0a5eb0ac` (36 fp32->fp16 attention casts): **a tinygrad ABI artifact, not
   a llama.cpp operation.** llama.cpp feeds F32 into the attention GEMVs
   (Q3/Q4 above) and never renders this cast. Tinygrad emits it because its
   Q4K decode attention kernels consume fp16 activations (the fp16 decode
   norm at `model.py:628/424` and the fp16 GEMV prelude at
   `decode_routes.py:132`). Because the cast is a lossy
   `cvt.rn.f16.f32`, it cannot be deleted as a no-op (that would change
   values), and the fp32 residual slot must be preserved (the attention resadd
   epilogue reads fp32). Two llama.cpp-compatible exits: (a) switch the Q4K
   attention input ABI to fp32 like llama.cpp's mmvq (requires the kernels to
   consume fp32 - the larger, structurally faithful change), or (b) fold the
   identical cast into the producer/consumer epilogue so the fp16 bytes are
   bitwise-identical to the standalone kernel (M2c's conservative path). Either
   way the standalone cast **kernel** disappears; only the fp16 bytes survive
   under option (b).

## 8. Caveats / boundary conditions

- Batch > 1 decode: `dst->ne[1] != 1` disables the fusion
  (`ggml-cuda.cu:2559`); llama.cpp then renders 36 standalone attention adds
  (1 per block), still 0 copies and 0 residual casts.
- Flash attention off (`LLAMA_FLASH_ATTN_TYPE_DISABLED`): the non-flash
  `build_attn_mha` path (`llama-graph.cpp:2105-2168`) still produces F32
  Q/K/V and an F32 `kqv` output, so the `wo` GEMV input stays F32 and the
  fusion story is unchanged.
- `GGML_CUDA_DISABLE_FUSION=1` and cc <= Pascal disable the epilogue
  absorption (`ggml-cuda.cu:2554-2556`), leaving standalone adds.
- K/V cache type is F16 by default; the fp16 KV boundary is a cache-store
  property, not an attention-activation cast, and does not change any
  conclusion above.
- The tinygrad `0a5eb0ac`/`fab82d40` counts (36/49) are the M2b/M2c census
  figures from `m2c_arithmetic_validation.md`; they vary per arm by ~0.1 us
  histogram medians, not structurally.
