# M3 fused decode RMSNorm - measurement record (NON-LANDING)

Status: measured non-landing on NV sm_120. The fused norm machinery is committed behind a
closed promotion record (`tinygrad/llm/generated/decode-norm-fusion-route-policy.json`) and
exercised by unit tests; no target is promoted. The M2 decode-epilogue record is untouched
and stays promoted on NV sm_120.

## What was built

`DecodeRMSNormSpec` + `emit_decode_rmsnorm_kernel` (`tinygrad/llm/decode_kernels.py`) plus the
model wiring (`_decode_rmsnorm`, `tinygrad/llm/model.py`): one opaque kernel per decode norm
(attn_norm, ffn_norm, q_norm, k_norm) replaces the generic reduce + epilogue pair, per
`l1-decode-plumbing-fusion-design-20260802.md` section 6 (norm family).

## Correctness (all pass)

- Fixed-depth decode sha256 `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9`,
  3/3 reps; first token `151936`, 3/3 reps (byte-identical to the M2 baseline).
- Kernel isolation probe: bitwise-identical rows for (1, 4096, 16 warps) and (1, 1024, 4 warps);
  q/k norms match to ~7e-07 (order-only sumsq deltas).
- The layer-0 e2e crash found during bring-up (`bad reshape: () -> (1, 1, 4096)`) was root-caused
  by graph instrumentation, not guessed: `_decode_rmsnorm` had passed `x.uop.base` (the embedding
  gather's raw producer carrying an internal `(1,1,1,4096,128)` vocab-block STAGE) into the opaque
  boundary, which materialized it at 5-D. Fix: pass the flat `(numel,)` view and index rank-1.

## Measured performance (NV, d512, Qwen3-8B-Q4_K_M)

| metric | M2 baseline (HEAD) | M3 fused | delta |
| --- | ---: | ---: | ---: |
| decode tok/s | 173.45 | 168.42 | -3% |
| kernels/token | 985 (M2 state) | 1093 | +108 |
| norm family kernels/token | 433 legacy | 289 fused + 144 copies | +144 copies |

Full name-level census diff (fused vs all-fusion-off): the norm fusion removes the 73
`r_16_256` reduces, 72+72 attn/ffn epilogues, 36+36 q/k reduces and the qk epilogue hashes, and
adds 144 `decode_rmsnorm_*` kernels plus 144 `E_32_32_4_86a2` / `E_8_32_4_dd98` boundary copies.
The `+144` copies are the exact gap between the doc's predicted -216 kernels and the measured
+108.

## Root cause of the regression

1. The opaque custom-kernel transport (`UOp.custom_kernel`) contiguous()s every non-identity
   input. The norm inputs are lazy function-arg producers (residual adds, qkv matmuls, embedding
   gather) with no buffer identity at trace time, so every fused call materializes one contiguous
   copy per token. A rank-3 pass-through was tried and does NOT elide the copy (still 144/token).
   The copies are structural to the boundary, not a reshape artifact.
2. The fused kernels are launch-bound at 3.2-5.0us (4.96us for `decode_rmsnorm_1_4096`), so the
   doc's llama-shaped 2.12us end-state is not reachable in this pipeline until the per-kernel host
   overhead work (decode-gap-per-target-lever-scope-20260802.md B3) lands. Even at 2.12us the 144
   copies (~1.5us each) would consume most of the win.

Net: correct, byte-identical tokens, but a measured regression, so the route is closed.

## Reopen conditions

- The boundary stops materializing per-call copies (view-passing opaque kernels: the emitter
  indexes the producer's logical shape and the transport preserves non-identity views), OR
- the per-kernel launch overhead work lands and the fused family beats the legacy pair at the
  campaign's fixed-depth protocol, AND
- a measured record with decode tok/s >= the M2 baseline at d512/d2048/d4096 and unchanged sha
  pins is attached to a review before the record flips.

## Controls

- `test/unit/test_decode_norm_fusion_gate.py`: closed default, checked-in record promotes
  nothing, M2 record still promotes NV, spec contracts, HIP+CUDA render arms.
- `scratchpad/pg3_decode_rendered_source_equality.py` gains `decode_rmsnorm_*` rows (HIP arm;
  Metal arm is macOS-only). Expected hashes are recorded at landing:
  - `decode_rmsnorm_1_4096` sha256 first-12: `2f3b80f7b426` (src_len 2820, ds_bpermute=5)
  - `decode_rmsnorm_32_128` sha256 first-12: `9cf696d384ba` (src_len 2041, ds_bpermute=5)
  - `decode_rmsnorm_8_128` sha256 first-12: `061dd2e554d0` (src_len 2039, ds_bpermute=5)
- Legacy pg3 rows and the M2 decode-epilogue gate tests are byte-identical (norm gate closed;
  legacy graph construction untouched).
