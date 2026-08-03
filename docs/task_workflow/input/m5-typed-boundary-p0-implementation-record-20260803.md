# M5 typed-boundary P0 implementation record

Date: 2026-08-03
Status: infrastructure landed closed. d512 census gate PASS (copy class 0 with the ABI active),
fixed-depth wall gate PASS at all three depths, all pins held, pg3 legacy hashes byte-identical.
The `decode_flash_combine_fusion` route record is explicitly NOT changed (HARD STOP below);
opening the route is a separate subsequent decision per
`nv-beyond-parity-forward-scope-review-amendment-20260803.md` section 4.1 item 2 and the scope's
section 8.

## Protocol

All GPU runs on the RTX 5090 (NV sm_120), Qwen3-8B-Q4_K_M (`/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`),
fused prefill attention disabled (`tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()`),
every timing run wrapped in `flock /tmp/nv_gpu.lock -c "<cmd>"`. Open mode = gate forced open for NV
sm_120 via the per-role admission override pattern (`mrp._DECODE_FLASH_COMBINE_FUSION_PROMOTED_TARGETS
= frozenset({("NV", "sm_120")})`, same pattern as `/tmp/m4_decomp_probe.py`); closed mode = all gates
closed (the M2-on production default). Fixed-depth wall uses `/tmp/path3_e2e_probe.py`'s protocol
(`--depth N --nmeas 20 --reps 3`), median tok/s. Census = d512 prime token, DEBUG=2, per-token kernel
histogram.

## What landed

The P0 infra (typed output layout declaration + typed input ABI + fail-closed validator, closed-default)
plus the lossless fp16->fp32->fp16 cast-pair cancellation that makes the fold fire on the real model:

- `d46cee681` `[nn]` - kernel_program.py typed boundary (TypedLayout, DeclaredTypedOutput,
  TypedViewRequest, OutputSpec.typed_output, KernelProgram.typed_input_views,
  `_DECLARED_TYPED_OUTPUTS`, `_validated_typed_view` fail-closed validator, `_fold_typed_input_views`
  in `_execute_outputs`); flash_decode_attention.py declares the fp16 combine typed output
  (fp16 `(Hq*Hd,)` row-major, viewable `(Hq, Hd)`, `combine_fusion_admitted=combine_fp16`);
  decode_routes.py attn_qo o-proj issues the typed-ABI opt-in `TypedViewRequest(slot=1, fp16,
  (binding.K,), "attn_qo")`. ffn_down / attn_kv / every other route keep the generic flat-buffer ABI.
- `44725ad41` `[test]` - test/unit/test_m5_typed_boundary.py (20 tests: producer declaration, graph
  probe fold vs fail-closed contrast, validator rejections, GEMV AST byte-identity, decode_routes
  wiring).
- `1d3ef5d6b` `[nn]` - validator extension: cancel the exact lossless fp16->fp32->fp16 cast pair over
  pure movement legs (real o-proj chain: model.py:704 upcasts the fp16 combine output to fp32 because
  decode q is fp32, then the prelude casts back). The composed fp16 view of the AFTER then runs every
  existing fail-closed check unchanged. Lossy shapes (bf16 round trips, arithmetic between the casts,
  non-identity movement) reject to the generic ABI.
- `781ba1b5a` `[test]` - 7 new tests: real-model cast-pair fold (zero materialization) vs the
  without-ABI copy contrast, lossy bf16 / arithmetic-between-casts / data-moving-permute rejections,
  real-chain GEMV AST byte-identity, decode_routes wiring on the fp32 pipeline chain.

Files owned by this P0: `tinygrad/llm/flash_decode_attention.py`, `tinygrad/llm/decode_routes.py`,
`tinygrad/llm/kernel_program.py`, `test/unit/test_m5_typed_boundary.py`. No change to
`tinygrad/uop/ops.py` defaults, no `UOp.custom_kernel` default-semantics change, no generated route
policy JSON touched.

## Unit probe results

- `test/unit/test_m5_typed_boundary.py`: 27 passed. The graph probe (scope section 6.1) builds the
  fp16 combine AFTER, its `RESHAPE(AFTER)` `(Hq, Hd)` view, and the attn_qo prelude
  `x[:, 0, :].reshape(K).cast(fp16).contiguous()`: under the typed ABI the contiguous request folds
  to a view with zero materialization (no `E_32_32_4_3b0fcfbc`-shaped copy kernel in the scheduled
  graph: only `flash_fused_gmax_combine_f16_32_128` + `q4k_g3_lanemap_gemv_4096_4096`); the same
  probe WITHOUT the typed ABI still schedules the copy (`[combine, copy, GEMV]`, 1 copy of shape
  `(4096,)` fp16 -> fp16). The real-model chain (fp32 pipeline) folds identically, and the emitted
  o-proj GEMV function AST is byte-identical with and without the view (scope section 7 audit).
- Related decode suites: 96 passed, 1 skipped, 1 failure
  (`test_decode_norm_fusion_gate.py::test_spec_validate_rejects_bad_contracts`), confirmed
  pre-existing on pristine HEAD (stash check) - not this P0.

## Kernel census (d512, per-token, DEBUG=2)

| class | closed baseline | open (P0 active) | delta |
| --- | ---: | ---: | ---: |
| total kernels/token | 1021 | 985 | -36 |
| total kernel us/token | - | 6129.4 | - |
| E_32_32_4_3b0fcfbc (fp16->fp16 copy) | 0 | 0 | 0 |
| E_32_32_4_0a5e (fp32->fp16 cast) | 36 | 0 | -36 |
| flash_fused_gmax_combine_f16_32_128 | 0 | 36 | +36 |
| flash_fused_gmax_combine_32_128 (legacy) | 36 | 0 | -36 |

The M5-specific census assertion (scope section 5) holds: `E_32_32_4_3b0fcfbc` count 0, `E_32_32_4_0a5e`
count 0, fp16 combine count 36, legacy combine count 0. The P0's value is copy removal: the 36 opaque-
boundary fp16 copies are gone, so kernels/token drops 1021 -> 985. The value is byte-identical (token
sha holds at every depth, both modes).

## Fixed-depth wall (gate table, open mode, nmeas=20 reps=3 median tok/s)

| depth | baseline (M2-on) | open (P0 active) | delta |
| --- | ---: | ---: | ---: |
| d512 | 172.80 | 173.687 | +0.89 |
| d2048 | 161.50 | 162.214 | +0.71 |
| d4096 | 149.00 | 149.918 | +0.92 |

No regression at any depth. Raw reps (warmup artifact first rep excluded from the median):
d512 `[6.94, 174.47, 173.69]`, d2048 `[6.60, 162.21, 162.69]`, d4096 `[5.68, 149.92, 150.15]`.

## Pins

- Fixed-depth token sha `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9` 3/3 at
  d512/d2048/d4096, in both closed and open modes.
- First token `151936` 3/3 at every depth, both modes.
- Decode sha `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe` (model_e2e_bench.py,
  d512 prefill, 96 decode tokens) MATCHED in closed default and with the gate forced open; first
  token ids `50994, 82, 31109, 3508, 692, 2, 11162, 100, 254, 30317, 2655, 12080, 25, 576, 35264, 5624`
  MATCHED.
- Bench census row `prefill_overlay_promotion: candidate_set:sha256:
  1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f` MATCHED (prefill untouched).

## pg3 table (render-only, HIPRenderer gfx1100, byte-identical)

| kernel | pinned sha | re-derived |
| --- | --- | --- |
| q4k_g3_lanemap_gemv_12288_4096 | 312422c73a49 | 312422c73a49 |
| q4k_g3_lanemap_gemv_4096_4096 | 27857cb8ca03 | 27857cb8ca03 |
| q4k_g3_lanemap_gemv_4096_12288 | 851760e2053c | 851760e2053c |
| q4k_g3_lanemap_gemv_1024_4096 | 39ddb717ddd4 | 39ddb717ddd4 |
| q6k_gen_coop_4096_12288 | cc38fbb3db92 | cc38fbb3db92 |
| q6k_gen_coop_151936_4096 | 5795e66a7292 | 5795e66a7292 |
| q6k_gen_partial_1024_4096_4 | 344e1c388eeb | 344e1c388eeb |
| q6k_vocab_scalar_reduce_151936_4096 | c708302aa2d2 | c708302aa2d2 |
| flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128 | 66d4c4da3108 | 66d4c4da3108 |
| flash_fused_gmax_combine_32_128 | c78e4651ad35 | c78e4651ad35 |
| flash_fused_gmax_combine_f16_32_128 | 94d73c1e9650 | 94d73c1e9650 |

All ten legacy rows unmoved; the additive fp16 combine row holds (`94d73c1e9650`). No legacy row
hash or `src_len` moved; the o-proj GEMV rendered source is unchanged under the view (AST byte-
identity, unit-proven).

## Isolated-benefit probe (d512, same-session back-to-back, lock-held)

| arm | median tok/s | kernels/token |
| --- | ---: | ---: |
| closed (legacy fp32 combine + cast) | 172.154 | 1021 |
| forced-open (fp16 combine + typed view, copy gone) | 172.532 | 985 |

Benefit: +0.378 tok/s (+0.22%), net zero to small win. The 36 removed copies are ~57 us/token of a
~5.8 ms decode, so copy removal alone is sub-1% on this HBM-bound decode. This is a small win, not a
regression: the record decision now has the measured benefit number, and the gate table above holds
above the M2-on baseline at all three depths. Per the amendment, a route opening decision (if any) is
a separate subsequent step; this P0 does not open the route.

## Deviations from the scope brief

1. The scope's settled chain (section 1) described a direct fp16 view (fp16 AFTER -> RESHAPE ->
   prelude). The real model additionally carries a lossless fp16->fp32->fp16 cast pair (model.py:704
   `attn = out.reshape(B,Hq,T,Hd).cast(q.dtype)` with decode q = fp32, then the prelude casts back to
   fp16). The validator had to cancel exactly that pair over pure movement legs for the fold to fire
   on the real graph; the scope's section 6.1 unit probe still passes unchanged and every fail-closed
   check is preserved (the census reached `3b0fcfbc=0` only with this extension).
2. The repo authored-LOC ratchet (`sz.py`, 40,000) had 13 lines of headroom at HEAD; the validator
   extension was compressed to land at 39999/40000 budgeted lines. No behavior was moved out of scope;
   comments/documentation carry the reasoning.
3. `model_e2e_bench.py`'s prefill arm errors with the HEAD-known pre-existing
   `PACKED_FRAGMENT_LOAD` UOp verification failure (documented in the B3 characterization record);
   the decode-arm pins (decode sha, first ids, census row) are taken from the decode arm and hold in
   both closed and open modes.
4. Every 3-rep wall run shows a warmup first rep (~6 tok/s, JIT capture after model load); the
   protocol row is the median of reps, matching the baseline record methodology.

## HARD STOP

The route record is explicitly NOT changed: `tinygrad/llm/generated/decode-flash-combine-route-policy.json`
still carries `"promoted_targets": []`. The infrastructure lands closed on its own per the amendment
(4.1 item 2); copy removal alone is not an opening criterion, and any route opening is a separate,
subsequent decision gated on the full section 5 gate (fixed-depth wall + shas + pg3) and this record's
measured benefit number (+0.378 tok/s at d512, net zero to small win). This document authorizes no
promotion and no push; the parent pushes after review.
