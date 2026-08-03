# Path 3 - semantic RMSNorm lowering (measurement record)

Status: measured NON-LANDING on NV sm_120. The semantic RMSNorm machinery is committed
behind a closed promotion record (`decode-rmsnorm-native-lowering-route-policy.json`,
`promoted_targets: []`) and exercised by unit tests; no target is promoted. Value parity is
proven end-to-end: the fixed-depth token sha is byte-identical with the native lowering
forced open at d512/d2048/d4096. The fused family regresses wall time ~1% because the
custom-kernel transport still materializes per-call input copies (the M3-class boundary
tax). Per the task protocol (path3-semantic-rmsnorm-task-20260802.md section 3), a
norm-family-only win is not a promotion, and this does not reopen anything.

## What was built

- `Ops.RMSNORM` semantic op + `RMSNormSpec` (dim, eps, out dtype, affine) with a validator,
  mirroring the `Ops.ATTENTION` precedent: src[0] is the ordinary fallback graph, src[1] is
  the input x, src[2] is the optional affine weight.
- `nn.RMSNorm.__call__` creates the marker via `Tensor._semantic_rmsnorm` ONLY when a
  load-time promotion flag (`_rmsnorm_native_promoted`, resolved once from
  `decode_rmsnorm_native_lowering_promoted`) is set; the ordinary expression stays the
  marker's first source (the universal fallback).
- `lower_rmsnorm_semantic` + `pm_rmsnorm_semantic` in `rangeify.py`: fail-closed
  re-admission (decode-shaped rows <= 32, dim % 32 == 0, fp16/fp32, contiguous
  MEMORY_SEMANTIC/buffer-backed x); admitted markers lower to ONE fused kernel
  (`rmsnorm_native_*`, the M3 emitter with a distinct name and the ordinary-graph
  intermediate cast in the epilogue) bound through the proven `UOp.custom_kernel`
  transport; everything else returns the ordinary source unchanged.
- Closed `boltbeam.route_policy.v1` record with `promoted_targets: []`; decode evidence
  never authorizes prefill (marker creation rejects rows > 32).

## Correctness (all pass)

- Fixed-depth token sha256
  `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9`, 3/3 reps, and first
  token `151936`, 3/3 reps, at d512/d2048/d4096 with the native lowering forced open AND
  with the default (closed) runtime - byte-identical to the M2 baseline.
- GPU isolation parity (NVRTC, CUDA): fp32 diff 0 (bitwise), fp16 diff 2.2e-3 (order-only
  sumsq delta), exactly one `rmsnorm_native_1_4096` kernel; prefill rows get NO marker.
- Unit tests: 24 passed (`test_rmsnorm_semantic_lowering.py` + `test_rmsnorm_native_lowering_gate.py`):
  marker admission, fail-closed fallback, single-kernel no-copy lowering for realized
  buffer inputs, M3 legacy key digest pin `19f2d6a8...`, native pin `533c1f05...`, closed
  default gate.
- pg3 rendered-source equality: `rmsnorm_native_1_4096` sha256 `965d8c5e084a` (src_len
  2780); M3 pins unmoved (`2f3b80f7b426` / `9cf696d384ba` / `061dd2e554d0`).

## Measured performance (NV sm_120, Qwen3-8B-Q4_K_M)

| depth | metric | default (M2 state) | forced-open Path 3 | delta |
| --- | --- | ---: | ---: | ---: |
| d512 | decode tok/s | 172.74 | 170.71 | -1.2% |
| d512 | kernels/token | 1021 | 1131 | +110 |
| d512 | kernel-us total | 6181 | 6311 | +130 |
| d2048 | decode tok/s | 161.47 | 159.34 | -1.3% |
| d2048 | kernels/token | 1021 | 1131 | +110 |
| d2048 | kernel-us total | 6582 | 6690 | +108 |
| d4096 | decode tok/s | 148.99 | 147.70 | -0.9% |
| d4096 | kernels/token | 1021 | 1131 | +110 |
| d4096 | kernel-us total | 7085 | 7196 | +111 |

Norm family (d512 census): 73x `rmsnorm_native_1_4096` @ 3.07us median - the attn/ffn/
output norms fuse; the PERMUTE-backed q/k norms stay on the ordinary path (see
deviations). The copies around the norms are present and measurable:

| census entry | default | forced-open | delta |
| --- | ---: | ---: | ---: |
| `rmsnorm_native_1_4096` | 0 | 73 | +73 |
| `E_32_32_4_0a5eb0ac...` (input copy class) | 36 | 108 | +72 |
| `E_32_32_4_f6721331...` (output materialization class) | 0 | 72 | +72 |
| `E_32_32_4_86a23e1a...` | 0 | 36 | +36 |
| `E_32_32_4_fab82d40...` | 72 | 73 | +1 |

The task protocol's no-copy census (norm family ~145 kernels with no copy/materialization
kernels around the norms, P0 `E_32_32_4_3b0f` count 0 on the native path) is NOT met: the
copies are present, so the protocol verdict is non-landing even though the sha gate holds.

## Root cause of the regression

1. The custom-kernel transport (`UOp.custom_kernel`) contiguous()s every non-identity
   input. The e2e norm inputs are lazy producer chains (residual adds, attention function
   outputs, the layer-0 embedding gather) with no buffer identity at lowering time, so each
   fused call materializes one contiguous copy per token. This is structurally identical to
   the M3 boundary tax, not a Path 3 artifact.
2. The no-copy alternative - passing the producer's `(1,1,4096)` MEMORY_SEMANTIC view
   straight into a manual `kernel.call` (rank-3 param, no reshape/contiguous) - crashes the
   `symbolic+reduce_collapse+debuf` pass: the lazy source collapses to `()` and the
   movement-op recompute raises `bad reshape: () -> (4096,)` / `bad expand: () -> (4096, 1)`
   from a RESHAPE/EXPAND whose producer collapsed. This is exactly the M3 reopen condition
   ("the emitter indexes the producer's logical shape and the transport preserves
   non-identity views") - still unbuilt in the scheduler.
3. The fused kernels are launch-bound at ~3.07us; even at that cost the copy tax dominates
   the norm-family win (the same conclusion as the M3 record).

## Verdict

NON-LANDING. The full fixed-depth protocol does not beat the M2 baseline at any depth
(-0.9% to -1.3% wall, +110 kernels/token). The semantic machinery is correct (byte-identical
tokens forced-open at all three depths), single-kernel and copy-free for buffer-backed
inputs (unit-proven), and closed by default. No promotion, no reopen.

## Reopen conditions

- The boundary stops materializing per-call copies (view-passing: the emitter indexes the
  producer's logical shape and the scheduler preserves non-identity views through
  symbolic), OR
- the per-kernel launch-overhead work lands (decode-gap-per-target-lever-scope B3) and the
  fused family beats the legacy pair at the fixed-depth protocol, AND
- a measured record with decode tok/s >= the M2 baseline at d512/d2048/d4096 and unchanged
  sha pins is attached to a review before the record flips.

## Deviations

- q/k norms: the lowering restricts the fused family to contiguous MEMORY_SEMANTIC /
  buffer-backed inputs (fail-closed), so the PERMUTE-backed q/k head norms stay ordinary.
  Net effect: fewer fused kernels and no q/k copies; the wall regression is smaller than
  M3's (-1% vs -3%).
- The input binding uses the proven `custom_kernel` transport rather than a manual
  `kernel.call`: the manual construction with a RESHAPE-over-lazy-producer call arg crashes
  the symbolic pass; the transport's CONTIGUOUS boundary is the only scheduler-stable
  binding available today.
- `DecodeRMSNormSpec` accepts x_rank in (1, 2, 3) (rank-2 added for the (1,dim) isolation
  probe shapes); the e2e lowering uses rank-1 exactly like M3.
