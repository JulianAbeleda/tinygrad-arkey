# M5 variant-reopen boundary P0 scope - fp16 combine output-layout/view-preservation

Date: 2026-08-03

Status: scope document only. This is the variant-reopen boundary P0 request for M5
(`nv-campaign-forward-review-amendment-20260803.md` section 2.2 item 3 of the corrected
sequence, section 4.1). It authorizes no implementation, no route-record change, and no
promotion. Branch boundary: tinygrad `nvidia-bringup-20260731`.

## 0. Why M5 is the first probe

Per the amendment's forced-open table, M5 is the only measured variant whose boundary
failure is a clean 1:1 substitution: absorbing the post-combine fp32->fp16 cast
(`E_32_32_4_0a5e`, 36x/token) at the combine replaces it one-for-one with a new fp16->fp16
copy (`E_32_32_4_3b0fcfbc`, 36x/token, ~1.58us median) at the opaque program boundary, with
zero wall gain and byte-identical tokens. There is no confounding wall win to argue over, so
a transport/contract P0 either removes the copy or it does not. M3 (144 input copies + 72
output materializations), M4 (126 opaque-boundary copies plus a second mechanism defect), and
Path 3 (+110 kernels/token) each carry additional mechanisms and stay closed regardless of
this P0's outcome (amendment section 4.1 item 4).

## 1. Boundary of record (amendment 2.2 item 1)

- Producer: `flash_fused_gmax_combine_f16_32_128`, the fp16 flash-combine variant
  (`flash_fused_gmax_combine_kernel(..., output_fp16=True)` in
  `tinygrad/llm/flash_decode_attention.py:205`; the store carries the same RNE `(half)`
  cast). It writes fp16 `(Hq*Hd,)` = `(4096,)` for Hq=32, Hd=128, S=48, executed through
  `KernelProgram` with `OutputSpec((Hq * Hd,), dtypes.float16)` and
  `execute_promoted_program` (`flash_decode_attention.py:526-529`).
- Consumer: the o-proj Q4K GEMV `q4k_g3_lanemap_gemv_4096_4096` (route_role `attn_qo`),
  whose activation prelude is
  `x[:, 0, :].reshape(binding.K).cast(dtypes.float16).contiguous()`
  (`tinygrad/llm/decode_routes.py:104`).
- Logical shape and dtype: producer writes a flat fp16 `(4096,)`; the consumer reads a
  logical `(1, 1, 32, 128)` -> flat fp16 `(4096,)` contiguous activation. No layout change
  is being asked for; only the materialization between the two is.
- Exact UOp chain with the variant open (as measured in the M5 record):

  1. `AFTER` of the fp16 combine `UOp.custom_kernel` call, dtype fp16, shape `(4096,)`;
  2. `flash_decode_live_split_block_tile` returns `out.reshape(Hq, Hd)` = `RESHAPE(AFTER)`,
     fp16 `(32, 128)` (`flash_decode_attention.py:529-530`);
  3. consumer prelude: `x[:, 0, :]` (slice) -> `reshape(4096)` -> `cast(fp16)` folds (dtype
     match, no kernel) -> `.contiguous()` materializes `E_32_32_4_3b0fcfbc` (fp16->fp16),
     because `has_buffer_identity()` follows `RESHAPE` but not `AFTER`
     (`tinygrad/uop/ops.py:999-1003`) and `UOp.custom_kernel` passes every non-`AFTER`
     argument through `.contiguous()` (`tinygrad/uop/ops.py:1264-1268`);
  4. `q4k_g3_lanemap_gemv_4096_4096` reads the resulting fp16 `(4096,)` PARAM.

  Legacy chain (gate closed): the fp32 combine `AFTER` (fp32 `(4096,)`) -> `RESHAPE(AFTER)`
  -> the same prelude, where `cast(fp16)` is now real work and emits the generic
  `E_32_32_4_0a5e` fp32->fp16 RNE cast kernel (digest
  `0a5eb0ac56c097a089f39541962d5d73b9bc613251a6320685824338d26b38c4`), which is itself a
  contiguous materialization, so no separate fp16 copy appears.

Net effect of the variant today: absorbs 36x `E_32_32_4_0a5e` casts, adds 36x
`E_32_32_4_3b0fcfbc` copies; zero kernel-count and zero kernel-us change.

## 2. Copy class of record (amendment 2.2 item 2)

- Class: `E_32_32_4_3b0fcfbc`, fp16->fp16 elementwise copy (captured CUDA source is a plain
  `half4` load/store pair).
- Count: 36x/token (one per combine output row group, d512 census).
- Median time: ~1.58us/kernel (M5 census; ~1.54us for the sibling M3/P0 class
  `E_32_32_4_3b0f`).
- Exists only with the variant open: the baseline census (gate closed) counts 0 instances of
  this class; the variant census (gate open) counts 36. The fp16 copy is a direct product of
  the opaque boundary re-materializing the fp16 combine output for the consumer's contiguous
  request, exactly like the M3/M4 output and input boundary copies.

## 3. Typed opt-in contract replacing the materialization (amendment 2.2 item 3)

The contract is output-layout/view-preservation on the combine output, consumer-specific and
closed-default:

1. The combine `KernelProgram` output declares its typed layout: fp16 `(Hq*Hd,)` row-major,
   logically viewable as `(Hq, Hd)` fp16 with no permutation, stride, or padding. The
   emitted kernel is unchanged (`flash_fused_gmax_combine_f16_32_128`); only the boundary's
   declared output ABI gains the layout/view metadata.
2. The o-proj Q4K GEMV program opts in with a typed input ABI mode on
   `KernelProgram`/`execute_promoted_program` that accepts a declared view of the combine
   output, so the prelude's `reshape(...).cast(fp16).contiguous()` folds to a view of the
   `AFTER` and no `E_32_32_4_3b0fcfbc` kernel is emitted. The opt-in is specific to the
   `attn_qo` o-proj consumer; `ffn_down`, `attn_kv`, and every other route keep the generic
   flat-buffer input ABI.
3. Closed-default: admission requires `FlashDecodeAdmission.combine_fusion_admitted` AND the
   consumer's explicit typed-ABI opt-in AND validator acceptance. The
   `decode_flash_combine_fusion` promotion record stays closed; nothing opens by default.
4. The default flat-buffer contract is unchanged. `UOp.custom_kernel`'s preserve-or-
   materialize rule and its default are NOT modified; the new ABI is a separate opt-in mode
   layered on the explicit `KernelProgram` boundary, not a change to `custom_kernel`'s
   default semantics (amendment section 5 and `decode-norm-fusion-paths-forward-20260802.md`
   sections 9.4-9.5).
5. Fail-closed admission with a validator. The validator must prove, before any copy-free
   binding is accepted: (a) the producer's declared output layout exactly matches the
   consumer's requested view (fp16, row-major, no permutation, no stride); (b) the consumer's
   `contiguous()` request folds to a view of the `AFTER` under the typed ABI; (c) the
   boundary is the single intended consumer (route_role `attn_qo`, o-proj GEMV); and (d)
   both the combine-fusion gate and the typed-ABI gate are open. Any mismatch, missing
   metadata, or gate-closed state rejects back to the legacy fp32 combine + generic cast
   route, which stays byte-identical.

## 4. Legacy route/hash byte-identity (amendment 2.2 item 4)

- Legacy combine `flash_fused_gmax_combine_32_128`: UOp digest
  `560ce2902832f4864e2776673061ca71f91acd685d4ca81e991a2bbade0e8fdf`; rendered HIP source
  `sha256=c78e4651ad35 src_len=2212`, unchanged (verified against a pristine HEAD worktree
  render in the M5 record). The fp16 variant is a NEW kernel name
  (`flash_fused_gmax_combine_f16_32_128`, rendered HIP `sha256=94d73c1e9650`), so no legacy
  hash moves.
- pg3 legacy 10-kernel HIP baseline rows (`scratchpad/pg3_decode_rendered_source_equality.py`,
  HIPRenderer gfx1100, render-only; re-derived with
  `PYTHONPATH=. .venv/bin/python scratchpad/pg3_decode_rendered_source_equality.py`):

  | kernel | sha256 |
  | --- | --- |
  | q4k_g3_lanemap_gemv_12288_4096 | 312422c73a49 |
  | q4k_g3_lanemap_gemv_4096_4096 | 27857cb8ca03 |
  | q4k_g3_lanemap_gemv_4096_12288 | 851760e2053c |
  | q4k_g3_lanemap_gemv_1024_4096 | 39ddb717ddd4 |
  | q6k_gen_coop_4096_12288 | cc38fbb3db92 |
  | q6k_gen_coop_151936_4096 | 5795e66a7292 |
  | q6k_gen_partial_1024_4096_4 | 344e1c388eeb |
  | q6k_vocab_scalar_reduce_151936_4096 | c708302aa2d2 |
  | flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128 | 66d4c4da3108 |
  | flash_fused_gmax_combine_32_128 | c78e4651ad35 |

  All ten rows must remain byte-identical under the P0. The pg3 file already carries an
  additive `flash_fused_gmax_combine_f16_32_128` row; if the typed ABI needs a rendered-AST
  row for the o-proj GEMV as consumed via the fp16 view, that row is added, never moved.

## 5. Fixed-depth wall and sha gate before the route record may change (amendment 2.2 item 5)

The `decode_flash_combine_fusion` record may change (promoted_targets gain the NV sm_120
target) only after ALL of the following hold on a same-session run:

- Fixed-depth wall: d512/d2048/d4096 W decode, `nmeas=20, reps=3`, median tok/s,
  `/tmp/path3_e2e_probe.py --depth N --nmeas 20 --reps 3 --no-fused-prefill`; must not
  regress the M2-on baseline (172.80 / 161.50 / 149.00 tok/s, `nv-decode-parity-final-20260802.md`).
  The P0's value is copy removal, not a wall win; a wall regression fails the gate even with
  the copy gone.
- Fixed-depth token sha: `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9`
  3/3 at every depth; first token `151936` 3/3 at every depth.
- Decode sha: `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe` unchanged.
- Bench census row: `prefill_overlay_promotion: candidate_set:sha256:1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f` unchanged.
- M5-specific census assertion: with the P0 active, the d512 kernel census must show
  `E_32_32_4_3b0fcfbc` count 0 (and `E_32_32_4_0a5e` count 0, `flash_fused_gmax_combine_f16_32_128`
  count 36, legacy `flash_fused_gmax_combine_32_128` count 0).

## 6. Settling command

1. Unit graph probe for the fp16 combine output view: a new unit test builds the fp16
   combine `AFTER`, its `RESHAPE(AFTER)` `(Hq, Hd)` view, and the `attn_qo` prelude
   (`x[:, 0, :].reshape(K).cast(fp16).contiguous()`), then asserts under the typed ABI that
   the consumer's contiguous request folds to a view with zero materialization (no
   `E_32_32_4_3b0fcfbc`-shaped copy kernel in the graph), and that the same probe WITHOUT
   the typed ABI still materializes the copy (the fail-closed contrast).
2. d512 census run asserting the copy class count goes to 0: d512 prime token, DEBUG=2,
   Qwen3-8B-Q4_K_M, NV sm_120, with the P0 active; assert `E_32_32_4_3b0fcfbc` count 0 in the
   per-token census (the M5-specific assertion in section 5).
3. Only after 1 and 2 pass, run the full section 5 gate (d512/d2048/d4096 wall + all shas +
   pg3 re-derive) before any record change is proposed.

## 7. Risk list

- Every opted-in consumer audited for flat-index assumptions. The o-proj Q4K GEMV indexes
  its activation flat at `binding.K`; a view-preserving input must not alter the emitted
  GEMV AST's index math or buffer roles. Audit the `attn_qo` path in
  `decode_routes.py:78-110` for any `contiguous()`, `numel()`, or flat-index assumption that
  would break under a view input, and pin the o-proj GEMV rendered source in pg3.
- The two boundary modes must not drift. The legacy fp32 combine + generic cast route and the
  fp16 combine + view route must remain distinct and independently pinned: separate kernel
  names, separate hashes, `combine_fusion_admitted` as the single gate, and the typed ABI
  active only for `attn_qo`. A wiring error that flips the default route moves legacy hashes;
  the section 4 byte-identity checks exist to catch exactly that.
- pg3 pins move deliberately. The ten legacy rows are immutable. Only additive rows (the fp16
  combine row already present, and a new o-proj view-consumption row if required) may be
  introduced; no legacy row hash or `src_len` may change.
- M4 decomposition stays separate. M4's k/v fp16-output piece overlaps M5's producer/output-
  layout problem but must not ride this P0; M4's combined record remains closed until its
  FFN-down SiLU/multiply recomputation defect is redesigned (amendment section 2.3).

## 8. HARD STOP

This scope authorizes no implementation and no record change. It is the variant-reopen
boundary P0 request for M5 only. A successful M5 P0 does NOT automatically reopen M3, M4, or
Path 3; each route needs its own d512/d2048/d4096 wall and sha record (amendment section 4.1
item 4). No route record is touched until the full section 5 gate passes, and even then the
record change is a separate, subsequent decision. This document is docs-only; no code and no
GPU use.

## 9. References

- `nv-campaign-forward-review-amendment-20260803.md` sections 2.2, 2.5, 4.1, 5
- `m5-flash-combine-normalization-measurement-record-20260802.md`
- `decode-norm-fusion-paths-forward-20260802.md` sections 9.4, 9.5
- `p0-72-copy-output-identity-verdict-20260802.md`
- `decode-gap-per-target-lever-scope-20260802.md` section 5.1 (pg3 HIP baseline)
- `nv-decode-parity-final-20260802.md` (wall baseline and pins)
- `tinygrad/uop/ops.py` (`has_buffer_identity`, `UOp.custom_kernel`)
- `tinygrad/llm/flash_decode_attention.py` (`flash_fused_gmax_combine_kernel`,
  `FlashCombineSpec`, `flash_decode_live_split_block_tile`)
- `tinygrad/llm/decode_routes.py` (o-proj Q4K GEMV prelude)
- `tinygrad/llm/generated/decode-flash-combine-route-policy.json` (closed record)
