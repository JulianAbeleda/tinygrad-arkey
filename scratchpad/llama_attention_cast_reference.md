# llama.cpp reference: attention-output fp32->fp16 cast and the tinygrad fp16 attention ABI

Date: 2026-08-11. Source tree: `/home/ubuntu/env/llama.cpp` (master,
`ac4cddeb0`). Tinygrad side: `/home/ubuntu/tinygrad-arkey` (branch
`nvidia-bringup-20260731`, HEAD `0dfe0ecac`). Scope: 1-token CUDA decode (the
tinygrad NV ground-truth case); read-only, no repo code touched. Model:
`Qwen3-8B-Q4_K_M.gguf` (36 layers, n_embd=4096, `attn_output` Q4_K). This
document answers what llama.cpp does (and does not) render between the
flash-attention output and the `wo` GEMV, and what that implies for the
tinygrad op `E_32_32_4_0a5eb0ac` (x36) sitting between
`flash_fused_gmax_combine_32_128` and `q4k_g3_lanemap_gemv_epi_resadd_4096_4096`.
Companion doc: `scratchpad/llama_attention_residual_reference.md` (sections
1.4/1.5 and 7 cover the residual stream; this doc covers the cast only).

## 1. Short answers

1. **Dtype of the flash-attention output before the `wo` GEMV: F32.**
   `ggml_flash_attn_ext` allocates its result as `ggml_new_tensor(ctx,
   GGML_TYPE_F32, 4, ne)` (`llama.cpp/ggml/src/ggml.c:5396`). The flash path
   in `build_attn_mha` reshapes that F32 tensor to 2D
   (`src/llama-graph.cpp:2104`) and `build_attn` feeds it straight into the
   `wo` GEMV: `cur = build_lora_mm(wo, cur, wo_s)`
   (`src/llama-graph.cpp:2349`) -> `ggml_mul_mat(ctx0, w, cur)`
   (`src/llama-graph.cpp:1076`), so **src1 of the `wo` mul_mat is the F32
   flash-attention output**. The Qwen3 graph builds this per block at
   `src/models/qwen3.cpp:108-110` (`build_attn(inp_attn,
   model.layers[il].wo, ...)`). The CUDA mmvq launcher enforces it:
   `GGML_ASSERT(src1->type == GGML_TYPE_F32)` (`ggml/src/ggml-cuda/mmvq.cu:1126`).
2. **Any cast/copy/cont between flash attn and the `wo` mmvq in the 1-token
   decode graph? No.** `qwen3.cpp` contains zero `ggml_cast`/`ggml_cpy`/
   `ggml_cont` (grep empty). Every hit in `llama-graph.cpp` is on a path this
   graph does not take; details in section 3. The only "op" between the F32
   flash output and the `wo` mul_mat is the view-only `ggml_reshape_2d`
   (`llama-graph.cpp:2104`), which moves no data.
3. **Where llama.cpp does convert fp32 -> fp16 in decode: the KV-cache store
   boundary only**, via the fused ROPE+VIEW+SET_ROWS path or the plain
   `ggml_set_rows` kernel. Both convert with the round-to-nearest-even
   `__float2half` family, i.e. PTX `cvt.rn.f16.f32`. Details in section 4.
   This is the rounding contract the tinygrad in-kernel cast must match.
4. **Tinygrad side confirmed:** `flash_fused_gmax_combine_kernel(..., 
   output_fp16=False)` stores `value = value.cast(dtypes.float16)` when
   `output_fp16=True` (`tinygrad/llm/flash_decode_attention.py:245`, kernel
   def at :206) and names the variant `flash_fused_gmax_combine_f16_{Hq}_{Hd}`
   (`flash_decode_attention.py:246`, spec property :451-455). The standalone
   cast `E_32_32_4_0a5eb0ac` is the same RNE fp32->fp16 conversion; the
   2026-08-02 measurement record captured its CUDA source as
   `*((half4*)(data0_4096+alu0)) = make_half4((half)val0.x, ...)` with digest
   `0a5eb0ac56c097a089f39541962d5d73b9bc613251a6320685824338d26b38c4`
   (`docs/task_workflow/input/m5-flash-combine-normalization-measurement-record-20260802.md:8-12`).
5. **Conclusion: bitwise-identical bytes.** The in-kernel fp16 combine store
   and the standalone cast are the same RNE conversion applied to the same
   fp32 value, so the fp16 bytes are identical (verified: token sha-256
   identical 3/3, `m5-flash-combine-normalization-measurement-record-20260802.md:50-52`).
   The fold previously measured non-landing only because the opaque program
   boundary materialized a new fp16->fp16 copy `E_32_32_4_3b0fcfbc` x36
   (`m5-flash-combine-normalization-measurement-record-20260802.md:33,36-40`)
   that replaced the absorbed cast 1:1. The M5 typed boundary (commit
   `d46cee681`, `tinygrad/llm/kernel_program.py` `_validated_typed_view` +
   `_cancel_lossless_fp16_roundtrip`) now prevents that copy
   (`docs/task_workflow/input/m5-typed-boundary-p0-implementation-record-20260803.md:24-33,53-60,65-79`),
   making the M2d fold byte-identical and landing-eligible. Details in
   sections 5-6.

## 2. Graph construction: F32 flash output into the `wo` GEMV

`src/models/qwen3.cpp` per block:

- L83-84: `build_qkv(model.layers[il], cur, ...)` produces Q/K/V from the F32
  `attn_norm` output (all F32; covered in the residual reference doc).
- L108-110: `cur = build_attn(inp_attn, model.layers[il].wo,
  model.layers[il].wo_b, model.layers[il].wo_s, Qcur, Kcur, Vcur, ...)`.
- `build_attn` (`src/llama-graph.cpp:2285-2358`): stores K/V into the cache
  (`cpy_k`/`cpy_v` at L2323-2324), reads K/V back from the cache (L2330-2331),
  and calls `build_attn_mha` (L2333). Flash attention is the default
  (`LLAMA_FLASH_ATTN_TYPE_AUTO`, `src/llama-context.cpp:3366`; enabled at
  L500).
- `build_attn_mha` flash branch (`llama-graph.cpp:2063-2104`): with a KV cache
  present, `k`/`v` are the cache tensors (F16 by default,
  `llama-context.cpp:3377-3378`), the F32-only casts at L2072-2078 are not
  inserted, and `cur = ggml_flash_attn_ext(ctx0, q, k, v, ...)` (L2080)
  produces an **F32** result (`ggml.c:5396`), reshaped 2D at L2104.
- Back in `build_attn`, L2349: `cur = build_lora_mm(wo, cur, wo_s)` ->
  `ggml_mul_mat(ctx0, wo, cur)` (`llama-graph.cpp:1076`) with **src1 = the F32
  flash output**. `ggml_mul_mat` allocates its result F32
  (`ggml/src/ggml.c:3242-3250`).

The Q4_K `wo` GEMV is dispatched through the mmvq path, whose host launcher
asserts F32 `src1` and F32 `dst` (`mmvq.cu:1126-1127`). No fp16 appears
anywhere between the flash output and the `wo` result.

## 3. The grep: no cast/copy/cont on this path

`grep -n "ggml_cast\|ggml_cpy\|ggml_cont"` over `src/models/qwen3.cpp`: **0
hits**. Over `src/llama-graph.cpp`, every hit is off-path for the Qwen3
1-token flash decode:

| line | construct | why it is not on this path |
| --- | --- | --- |
| L2072-2078 | `k`/`v` `ggml_cast(..., GGML_TYPE_F16)` | inside `build_attn_mha`, guarded by `k->type == GGML_TYPE_F32` / `v->type == GGML_TYPE_F32`; comment: "this can happen when KV cache is not used". With the F16 KV cache the tensors are already F16, so no cast node is created. |
| L2100 | `ggml_cont` (MLA branch) | inside `if (v_mla)`, and `build_attn` asserts `v_mla == nullptr` (L2298). |
| L2146, L2162 | `ggml_cont`/`ggml_cont_2d` | non-flash `build_attn_mha` branch; flash is on (L500) so this branch is not built. |
| L1562-1563, L1798, L1880, L2033 | MoE expert casts/cont, Granite embedding cont, ALiBi `pos_bias` cont | other architectures / pre-attention setup, not the Qwen3 attention path. |
| L2761, L2845 | `ggml_cpy` | recurrent-state models, not Qwen3. |

The single reshape between flash output and `wo` (`llama-graph.cpp:2104`) is a
view (`ggml_reshape_2d`), not a copy. Net: in llama.cpp the F32 flash output
reaches the `wo` GEMV with **no cast, no copy, no cont**.

## 4. The rounding contract: llama.cpp's fp32->fp16 conversions in decode

The only fp16 activations llama.cpp produces in the Qwen3 decode path are the
K/V cache rows. The conversion happens at the KV-store boundary, built as
`ggml_set_rows` (`src/llama-kv-cache.cpp:1323` for K, `:1358` for V), with
cache type F16 by default (`llama-context.cpp:3377-3378`).

Two CUDA implementations, both round-to-nearest-even:

1. **Fused ROPE+VIEW+SET_ROWS** (the live 1-token path): the graph-compute
   loop fuses the three nodes (`ggml/src/ggml-cuda/ggml-cuda.cu:3958-3965`;
   eligibility includes `set_rows->type` F32 or F16,
   `ggml-cuda.cu:3425`). The F32->F16 dispatch is
   `rope_norm_cuda<forward, float, half>` (`ggml/src/ggml-cuda/rope.cu:646-650`)
   and the store is `half2 v = make_half2(x0, x1)` (`rope.cu:91`), which
   converts each float through `__floats2half2_rn` (`cuda_fp16.h:348`) - RN.
2. **Unfused `ggml_set_rows` fallback**: the F16-dst dispatch
   (`ggml/src/ggml-cuda/set-rows.cu:240-249`) stores
   `dst_row_ptr[i00] = ggml_cuda_cast<half>(src0_row[i00])`
   (`set-rows.cu:171`), which lowers through the `__half(float)` constructor
   (`/usr/local/cuda/include/cuda_fp16.h:4698`) to `__float2half`
   (documented as the RN variant, `cuda_fp16.h:207`) == `__float2half_rn`
   (`cuda_fp16.h:231`).

Both spellings compile to PTX `cvt.rn.f16.f32` (round-to-nearest-even, RNE).
The only graph-level `ggml_cast` to F16 (the conditional K/V cast at
`llama-graph.cpp:2072-2078`) also goes through the CUDA convert kernel
(`ggml/src/ggml-cuda/convert.cu:680`, same `ggml_cuda_cast`), i.e. the same
RNE. **RNE fp32->fp16 is the rounding contract the tinygrad in-kernel cast
must match.**

## 5. Tinygrad parity: the same RNE cast, producer side

The tinygrad flash combine emits fp32 output by default and, under the
closed-default M5 gate, an fp16 variant:

- `tinygrad/llm/flash_decode_attention.py:206` -
  `flash_fused_gmax_combine_kernel(Hd, Hq, S, stride, output_fp16=False)`.
- `flash_decode_attention.py:244-246` - `value = final_acc / final_den`;
  `if output_fp16: value = value.cast(dtypes.float16)`, kernel named
  `flash_fused_gmax_combine_f16_{Hq}_{Hd}`.
- `flash_decode_attention.py:451-455` - `FlashCombineSpec.kernel_name` /
  `emit()` drive the same f16 variant; admission is the separate
  `decode_flash_combine_fusion` record (`combine_fusion_promoted` field at
  `flash_decode_attention.py:563`, `combine_fusion_admitted` at :572-573).

The standalone cast that the f16 variant replaces was captured in the
2026-08-02 record: a "pure fp32->fp16 elementwise RNE cast", CUDA source
`*((half4*)(data0_4096+alu0)) = make_half4((half)val0.x, ...)`, digest
`0a5eb0ac56c097a089f39541962d5d73b9bc613251a6320685824338d26b38c4`, sitting
between `flash_fused_gmax_combine_32_128` (fp32 `(Hq*Hd,)` combine output) and
the o-proj GEMV
(`m5-flash-combine-normalization-measurement-record-20260802.md:8-12`). In the
current per-block order the consumer is
`q4k_g3_lanemap_gemv_epi_resadd_4096_4096`
(`scratchpad/m2c_dependency_map.md:50`). The fp16 combine "store carries the
same RNE `(half)` cast"
(`m5-flash-combine-normalization-measurement-record-20260802.md:13-14`), and
the consumer prelude is `x[:, 0, :].reshape(binding.K).cast(dtypes.float16)
.contiguous()` (`tinygrad/llm/decode_routes.py:132,326`), which under the M5
typed ABI folds to a view of the combine AFTER (`decode_routes.py:143-146`).

`0a5eb0ac` is a **lossy** cast (`cvt.rn.f16.f32`, 10-bit mantissa), so it can
never be deleted as a no-op; it may only vanish by fusing the identical cast
into the producer/consumer epilogue so the fp16 bytes are bitwise-identical to
the standalone kernel (`scratchpad/m2c_arithmetic_validation.md:97-110`). That
is exactly what the f16 combine store does.

## 6. Conclusion for the M2d fold

**Bitwise contract:** the in-kernel fp16 store
(`flash_fused_gmax_combine_f16_{Hq}_{Hd}`) and the standalone cast
(`E_32_32_4_0a5eb0ac`) are the same RNE fp32->fp16 conversion of the same fp32
value, so the fp16 bytes are identical. The 2026-08-02 variant already proved
this empirically: token sha-256 identical at every depth, 3/3 reps
(`m5-flash-combine-normalization-measurement-record-20260802.md:47-52`).

**Why it did not land then:** the opaque `uop_program` CALL/AFTER boundary
materialized a new fp16->fp16 copy class `E_32_32_4_3b0fcfbc` x36 (~1.58 us
each), replacing the absorbed cast 1:1 and leaving net kernel count and net
time unchanged (`m5-flash-combine-normalization-measurement-record-20260802.md:33,36-40,54-60`).

**What changed since:** commit `d46cee681` (2026-08-03) added the M5 typed
boundary - typed output-layout declarations plus the fail-closed validator
`_validated_typed_view` (`tinygrad/llm/kernel_program.py:241-289`) with the
lossless fp16->fp32->fp16 cast-pair cancellation
`_cancel_lossless_fp16_roundtrip` (`kernel_program.py:222-238`), wired through
`_fold_typed_input_views` (`kernel_program.py:292-310`). With the ABI active
the attn_qo contiguous request folds to a zero-copy view of the fp16 combine
AFTER; the probe schedules only
`flash_fused_gmax_combine_f16_32_128` + `q4k_g3_lanemap_gemv_4096_4096`, while
the same probe without the ABI still schedules the copy
(`m5-typed-boundary-p0-implementation-record-20260803.md:53-60`). Census
confirms it: `E_32_32_4_3b0fcfbc` count 0, `E_32_32_4_0a5e` count 0, fp16
combine x36, legacy combine x0, kernels/token 1021 -> 985, tokens
byte-identical (`m5-typed-boundary-p0-implementation-record-20260803.md:65-79`).

**M2d fold verdict:** the copy obstacle that made the 2026-08-02 fold
non-landing is removed by the typed boundary. The remaining requirement is
process-level: the `decode_flash_combine_fusion` route record was explicitly
left closed (HARD STOP) at the P0 landing, opening it being a separate
decision (`m5-typed-boundary-p0-implementation-record-20260803.md:4-9`). On the
evidence in this document the fold is byte-identical and landing-eligible;
llama.cpp contributes no counter-evidence (it never renders this cast at all -
its attention GEMVs are F32 end to end, residual reference doc sections
1.4/1.5 and 7).

## 7. Caveats

- The fused store must stay RNE: any truncation (e.g. `cvt.rz.f16.f32`) would
  change fp16 bytes and trip the exact-logits SHA gate
  (`m2c_arithmetic_validation.md:105-110`).
- A validator rejection (gate closed, wrong `route_role`, non-pure view chain)
  falls back to the generic flat-buffer ABI, which materializes the old
  byte-identical copy - safe by construction (`kernel_program.py:241-244`).
- The fp16 attention ABI remains a tinygrad artifact; llama.cpp feeds F32 into
  attention GEMVs (`mmvq.cu:1126`). Switching the Q4K attention input ABI to
  fp32 would remove the cast entirely, but that is the larger, structurally
  faithful change (residual reference doc section 7); M2d takes the
  conservative bitwise-identical fold instead.
