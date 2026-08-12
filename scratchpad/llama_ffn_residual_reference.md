# llama.cpp reference: FFN residual add in the Qwen3-8B Q4_K_M decode path

Date: 2026-08-11. Source tree: `/home/ubuntu/env/llama.cpp` (master, `ac4cddeb0`).
Scope: 1-token CUDA decode (the tinygrad NV ground-truth case). No repo code was
modified.

## 1. Short answers

1. Does llama.cpp render a separate elementwise add for `h + ffn_out`?
   **No, not for single-token decode.** The residual add is a real graph node
   (`ggml_add`), but the CUDA backend absorbs it into the `ffn_down` GEMV
   epilogue: the mmvq kernel computes `total + h[row]` in fp32 and stores fp32,
   bitwise the same expression the standalone add would lower. The same applies
   to the attention residual add, which is absorbed into the `wo` epilogue.
2. Does llama.cpp materialize a contiguous copy of the block output between
   blocks? **No.** The block output *is* the residual-add node, and the fused
   GEMV writes its result directly into that node's fp32 contiguous buffer. The
   next block reuses the same tensor by pointer (`inpSA = inpL`, `inpL = cur`).
   There is no `ggml_cpy` / `ggml_cont` anywhere in the Qwen3 between-block path.
3. CUDA kernels involved: fused `mul_mat_vec_q` (Q4_K/Q6_K mmvq with
   `ggml_cuda_mm_fusion_args_device.x_bias`), or the standalone elementwise
   `k_bin_bcast` add (`ggml_cuda_op_add`) only when fusion cannot apply.

## 2. Graph construction (per block)

`src/models/qwen3.cpp:71-139` (same pattern in classic `src/models/llama.cpp:180-187`):

- L72: `ggml_tensor * inpSA = inpL;` - the residual stream is a **node reuse**,
  not a copy or view-copy.
- L116: `ffn_inp = ggml_add(ctx0, cur, inpSA);` - attention residual add node.
- L125-130: `cur = build_ffn(...)` with `ffn_down` and `down_b == NULL`.
- L133: `cur = ggml_add(ctx0, cur, ffn_inp);` - **the FFN residual add node**.
- L139: `inpL = cur;` - the add node is passed to the next block as-is.

`build_ffn` (`src/llama-graph.cpp:1244`) ends with `build_lora_mm(down, cur)`
(L1385), which without LoRA is just `ggml_mul_mat(ctx0, down, cur)`
(`src/llama-graph.cpp:1072-1074`). Qwen3 has no `down_b`/`down_s`
(`src/models/qwen3.cpp:45-47`), so the node directly following the `ffn_down`
`GGML_OP_MUL_MAT` in topological order is the residual `GGML_OP_ADD` at
qwen3.cpp:133. Graph order is guaranteed by the post-order DFS visit in
`ggml_visit_parents_graph` (`ggml/src/ggml.c:6899`, appended via
`ggml_build_forward_expand` at `ggml/src/ggml.c:7000`): the mul_mat's only
consumer is the add, so they are adjacent in `cgraph->nodes[]`.

## 3. CUDA fusion: MUL_MAT + ADD absorbed into the GEMV epilogue

The CUDA graph-compute loop calls `ggml_cuda_try_fuse` per node
(`ggml/src/ggml-cuda/ggml-cuda.cu:3879`, invoked at `ggml-cuda.cu:4424`).
The generic "bias add" branch (`ggml-cuda.cu:4177-4236`) matches any
`MUL_MAT, ADD` pair where the add's `src[0]`/`src[1]` is exactly the mul_mat
node (`ggml-cuda.cu:4190-4196`) and both operands are same-shape
(`ggml-cuda.cu:4212`; 4096x1 vs 4096x1 here). For Qwen3's residual add this
holds, so it is classified as an "x_bias" and dispatched as
`ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, bias_node, &fusion_data)` with
`fusion_data.x_bias = ffn_inp` (`ggml-cuda.cu:4216-4227`). Note the **dst** is
the add node (`bias_node`), not the mul_mat node; the mul_mat node itself is
elided (loop skips `fused_node_count - 1` nodes).

Eligibility (`ggml_cuda_should_fuse_mul_mat_vec_q`, `ggml-cuda.cu:2541-2572`):
quantized `src0` (Q4_K/Q6_K), F32 `src1` and `dst`, `src1->ne[1] <=
MMVQ_MAX_BATCH_SIZE`, `dst->ne[1] == 1` (single-token), cc > Pascal, no split
buffers. All true for the Qwen3-8B Q4_K_M single-token decode.

Kernel epilogue (`ggml/src/ggml-cuda/mmvq.cu:475-478`, `mul_mat_vec_q` with
`has_fusion=true`):

- `x_bias` (the residual row vector, fp32) is prefetched per row at
  `mmvq.cu:522-542`.
- Epilogue at `mmvq.cu:640-644`: `result += x_biases[j];` (fp32 accumulator).
- Store at `mmvq.cu:668`: `dst[j*stride_col_dst + threadIdx.x] = result;`
  (fp32 store, contiguous).
- Host launcher `ggml_cuda_mul_mat_vec_q` (`mmvq.cu:1123-1159`) asserts the
  residual is F32 with `ne[0] == dst->ne[0]` (4096 == 4096).

This is byte-for-byte the arithmetic the standalone add would do
(`total + h[row]`, fp32 store). The workspace already replays this exact fused
kernel from llama.cpp's own cubin in
`scratchpad/llama_cuda_ffn_down_oracle.py` (entries
`mul_mat_vec_q<...Q4_K/Q6_K..., has_fusion=true>` from
`libggml-cuda.so.0.14.36.sm_120a.cubin`, rows=4096, k=12288, formula
`quantized_matvec(x) + x_bias`), so the fusion is empirically confirmed, not
just inferred from source.

## 4. When llama.cpp does render a standalone add

The `GGML_OP_ADD` fallback is `ggml_cuda_op_add`
(`ggml/src/ggml-cuda/binbcast.cu:393`, kernel `k_bin_bcast` with `op_add`,
`binbcast.cu:13-34`; dispatch at `ggml-cuda.cu:2864-2866`). It fires only when
fusion cannot: batch > 1 (`dst->ne[1] != 1`), cc <= Pascal,
`GGML_CUDA_DISABLE_FUSION` (`ggml-cuda.cu:3881`), split buffers, or a node
between the mul_mat and the add (only the last layer's attention add, see 6).

## 5. Between blocks: no copies, buffer passed by node reuse

There is **no** `ggml_cpy` / `ggml_cont` / `ggml_add_inplace` in the Qwen3
between-block path. `ggml_add_inplace` does not exist in the current tree at
all; the only `ggml_cpy` calls in `src/llama-graph.cpp` (L2761, L2845) are for
recurrent-state (RWKV/Mamba-family) models, unrelated to Qwen3. The residual
buffer is passed between blocks purely as `ggml_tensor *` reuse: the add node
at qwen3.cpp:133 becomes `inpL` (qwen3.cpp:139), next block does
`inpSA = inpL` (qwen3.cpp:72), and reads that same buffer as the `x_bias` of
the next fused `wo` GEMV and as the input of the next `ffn_norm` RMS_NORM.
ggml has no implicit copy semantics: every op reads its `src` buffers by
pointer, and within one backend the scheduler inserts no copies. The fused
mmvq kernel writes the block output straight into the add node's buffer, so
the block output is materialized exactly once, contiguous fp32, and consumed
in place. Buffer lifetimes are handled by ggml-alloc (buffer reuse across
blocks is an allocator detail, never a copy).

## 6. Arithmetic per token (36 blocks, rows=4096, fp32)

FFN residual (`ggml_add` after `ffn_down`):

- Graph nodes: 36 (one per block, qwen3.cpp:133).
- CUDA kernels: **0 standalone adds** - all 36 absorbed into the `ffn_down`
  Q4_K/Q6_K mmvq epilogues (`mmvq.cu:640-668`).
- Copies: **0**.

Attention residual (`ggml_add` after `wo`, qwen3.cpp:116):

- 35 absorbed into the `wo` mmvq epilogue (same mechanism).
- 1 standalone `k_bin_bcast` add per token: the last layer has
  `ggml_get_rows(..., inp_out_ids)` inserted between `wo` and the add
  (qwen3.cpp:112-115), which breaks the MUL_MAT+ADD adjacency. This
  `get_rows` (token selection) is the only copy-like node llama.cpp renders in
  the whole 36-block path, and it is in the last layer only, not between
  blocks.

Net: **0 add kernels + 0 copies per block** for the FFN residual; 1 standalone
add + 1 `get_rows` for the last layer only.

## 7. "So what" for tinygrad (why no 72-copy explosion)

Tinygrad's M2b candidate hit 72 `E_32_32_4_86a23e1a` copies (2 per block:
one `.contiguous()` right after the absorbed-add GEMV, one at the next block's
residual-input boundary - see `scratchpad/m2b_arithmetic_validation.md`
section 4). llama.cpp does not, because of two coupled properties:

1. **The fused kernel's destination is the residual-add node's buffer.** In
   llama.cpp the add at qwen3.cpp:133 is a first-class tensor; the fusion
   passes that tensor as the mmvq `dst` (`ggml-cuda.cu:4227`,
   `mmvq.cu:1124`), and the elided mul_mat never materializes. There is no
   separate "GEMV output buffer" that differs from "block-output buffer" -
   they are the same allocation, written once, contiguous fp32 by
   construction.
2. **Graph edges reuse that buffer by pointer.** The next block reads the same
   `ggml_tensor` as `inpSA` (qwen3.cpp:72) for both the attention-residual
   `x_bias` and the RMS norm. Since ggml never inserts implicit copies, the
   block output is consumed in place; nothing forces a contiguous re-buffer at
   either boundary.

In tinygrad, the epilogue absorption made the GEMV's own output buffer hold
`h + ffn_out`, which is a different logical tensor from the graph's
block-output residual tensor, so the scheduler re-materializes the fp32 block
output as a contiguous tensor at both boundary positions (2 per block). The
llama.cpp fix for that class of problem is the same fix M2b needs: make the
absorbing kernel's output buffer *be* the block-output tensor, so downstream
consumers read it directly instead of through `.contiguous()` copies.

## 8. Caveats / boundary conditions

- Batch > 1 decode: `dst->ne[1] != 1` disables the fusion
  (`ggml-cuda.cu:2559`); llama.cpp then renders 36 standalone adds (1 per
  block), still 0 copies.
- Prefill (n_tokens >> 1) uses mmq/mmf/cuBLAS paths; residual adds are
  standalone there too.
- `GGML_CUDA_DISABLE_FUSION=1` and cc <= Pascal disable the epilogue
  absorption (`ggml-cuda.cu:2554-2556`, `ggml-cuda.cu:3881`).
