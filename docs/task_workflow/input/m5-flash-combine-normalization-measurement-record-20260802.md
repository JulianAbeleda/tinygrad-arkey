# M5 flash-decode combine normalization measurement record

Date: 2026-08-02
Status: measured non-landing (tokens byte-identical, opaque-boundary copy replaces the absorbed cast 1:1)

## What landed

The post-flash-combine attention-output cast (`E_32_32_4_0a5e`, 36x/token) is a pure fp32->fp16
elementwise RNE cast (captured CUDA source: `*((half4*)(data0_4096+alu0)) = make_half4((half)val0.x, ...)`,
digest `0a5eb0ac56c097a089f39541962d5d73b9bc613251a6320685824338d26b38c4`) sitting between
`flash_fused_gmax_combine_32_128` (fp32 `(Hq*Hd,)` combine output) and the o-proj GEMV
`q4k_g3_lanemap_gemv_4096_4096`, whose prelude `_xv` casts to fp16 and contiguizes anyway.
M5 absorbs the cast at the combine: a NEW fp16 combine variant `flash_fused_gmax_combine_f16_{Hq}_{Hd}`
(store carries the same RNE `(half)` cast), additive under a NEW closed-default record
`decode_flash_combine_fusion` -- deliberately SEPARATE from M2's `decode_epilogue_fusion` record (NV
sm_120, Q6K in-kernel merge) and from M4's q4k record. `model.py` needed zero changes: the combine output
is fp16, so `out.reshape(B,Hq,T,Hd).cast(q.dtype)` folds (`Tensor.cast` returns `self` on dtype match) and
the o-proj GEMV reads the combine output directly.

Legacy combine byte-identity: `flash_fused_gmax_combine_32_128` UOp digest
`560ce2902832f4864e2776673061ca71f91acd685d4ca81e991a2bbade0e8fdf` and rendered HIP source
`sha256=c78e4651ad35 src_len=2212` unchanged (verified against a pristine HEAD worktree render).
pg3 q4k pins unmoved: `312422c73a49`/`27857cb8ca03`/`851760e2053c`/`39ddb717ddd4`.

## Kernel census (d512 prime token, DEBUG=2, Qwen3-8B-Q4_K_M, NV sm_120)

| class | baseline (gate closed) | variant (gate open) | delta |
| --- | ---: | ---: | ---: |
| total kernels/token | 1021 | 1021 | 0 |
| total kernel us/token | 6188 | 6183 | -5 |
| E_32_32_4_0a5e (fp32->fp16 cast) | 36 | 0 | -36 |
| flash_fused_gmax_combine_f16_32_128 | 0 | 36 | +36 |
| E_32_32_4_3b0fcfbc (NEW fp16->fp16 copy) | 0 | 36 | +36 |
| flash_fused_gmax_combine_32_128 (legacy) | 36 | 0 | -36 |

The fp16 combine absorbs the 36 cast kernels but the opaque `uop_program` boundary materializes a new
fp16->fp16 copy class (`E_32_32_4_3b0fcfbc`, 36x, ~1.58us; captured CUDA source is a plain `half4`
load/store copy). This is the same M3/M4 lesson: the fixed-ABI combine output cannot be re-laid-out to
the consumer's transposed view, so the generic scheduler inserts a per-input copy. Net kernel count and
net kernel time are unchanged (the copy costs the same ~1.6us the cast did).

## Wall tok/s (d512, 3 reps median, first token)

| metric | baseline | variant | delta |
| --- | ---: | ---: | ---: |
| tok/s | 173.175 | 173.21 | +0.04 (noise) |
| token sha256 | `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9` | identical (3/3) | identical |
| first token | 151936 | 151936 (3/3) | identical |

Byte-identity is confirmed -- the fp16 store cast and the absorbed generic cast are the same RNE
fp32->fp16 conversion of the same fp32 value. The variant does not win: the copy replaces the cast
1:1, so there is no wall-time or kernel-count improvement to promote.

## Verdict

Non-landing. Tokens are byte-identical but the variant adds a 36x fp16 copy class at the opaque
combine boundary (net zero kernels/us), so per the campaign rule it cannot land; the gate stays
closed and no target is promoted. The emitter code is correct and additive -- it is ready for
promotion when the opaque-kernel-boundary copy problem is resolved (decode-norm-fusion-paths-forward
paths 1/2/3).

## Promotion record

New, closed (`tinygrad/llm/generated/decode-flash-combine-route-policy.json`):
```json
{
  "schema": "boltbeam.route_policy.v1",
  "route": "decode_flash_combine_fusion",
  "promoted_targets": []
}
```

M2's record (`decode-epilogue-fusion-route-policy.json`, NV sm_120) is unchanged; M4's q4k record and
M3's norm record are unchanged.

## Deviations from brief

1. **Chosen option**: combine-emitter-writes-fp16 (new kernel name) over the o-proj GEMV prelude
   reading fp32. The prelude option would leave the fp32 combine ABI and still materialize the cast
   (or require a fp32-load GEMV variant); the combine option is the M4-shaped minimal change.
2. **No model.py change**: `out.reshape(...).cast(q.dtype)` folds to a view once the combine output is
   fp16, so only flash_decode_attention.py / decode_routes.py / model_route_plan.py / the record JSON
   changed. The gate is threaded through `FlashDecodeRouteConfig.evaluate` (decode_routes.py:267's
   call) -> `_FlashDecodeBinding.combine_fusion` -> `flash_decode_attention_route` ->
   `flash_decode_live_split_block_tile`.
3. **Diagnosis surprise was the copy, not the cast**: the E kernel IS a plain cast (as the design doc
   claimed); the surprise is that the fp16 combine output triggers a NEW fp16 copy at the opaque
   boundary instead of feeding the GEMV view chain copy-free. Recorded, not forced.
4. No d2048/d4096 measurement: the d512 result is already non-landing on the brief's own criterion
   ("adds copies"), so further depth measurements would not change the verdict.

## Files changed

- `tinygrad/llm/flash_decode_attention.py` -- `flash_fused_gmax_combine_kernel(..., output_fp16)`,
  `FlashCombineSpec.output_fp16`, `describe_flash_decode_attention(..., combine_fp16)`,
  `FlashDecodeAdmission.combine_fusion_promoted` + `combine_fusion_admitted`,
  `FlashDecodeRouteConfig.evaluate(..., combine_fusion_promoted)`,
  `flash_decode_live_split_block_tile(..., combine_fp16)` + combine `OutputSpec` dtype
- `tinygrad/llm/decode_routes.py` -- `decode_flash_combine_fusion_promoted` at bind,
  `_FlashDecodeBinding.combine_fusion`, `flash_decode_attention_route` passes `combine_fp16`
- `tinygrad/llm/model_route_plan.py` -- `load_decode_flash_combine_fusion_promotion` /
  `decode_flash_combine_fusion_promoted` (new closed record)
- `tinygrad/llm/generated/decode-flash-combine-route-policy.json` -- closed promotion record
- `test/unit/test_flash_combine_fusion_m5.py` -- 11 unit tests (spec validation, legacy combine
  hash/name byte-identity, new variant name, HIP+CUDA render arms, admission wiring)
- `test/unit/test_flash_combine_fusion_gate.py` -- closed-record gate tests (combine promotes
  nothing; M2 record keeps NV sm_120; M3/M4 records stay closed)
- `scratchpad/pg3_decode_rendered_source_equality.py` -- new `flash_fused_gmax_combine_f16_32_128`
  row in the render-equality baseline

## Test results

- `test_flash_combine_fusion_m5.py` + `test_flash_combine_fusion_gate.py`: 13 passed
- Related decode suites (production flash attention, M2/M4 gates, llm decode routes, route
  selection, buffer roles, model facts): 88 passed, 1 skipped, 0 failed
- `test_flash_decode_intrinsics_renderer_lowering.py`: 3 pre-existing Metal failures on this Linux
  box, identical on pristine HEAD (not M5)
- `scratchpad/pg3_decode_rendered_source_equality.py` (HIP arm): all legacy hashes unmoved, new
  fp16 combine row renders (`sha256=94d73c1e9650`)
