# NV reduce-output weight-identity wall bracket record (gates PASS, bracket NO-GO)

Status: **NO-GO for promotion, but the weight-identity fix is CORRECT and the
phase-6 route-efficiency mechanism is now understood end to end. Binding the
marker to a load-time materialized fp16 weight (commit `e00395b75`) removes
the per-body `8eeb0be1` materialization store completely (914 -> 896 kernels,
the last non-norms kernel family shift in the fused route), every gate still
PASSES, and the reverse control/candidate/control wall bracket again does NOT
promote: candidate 5.4204 ms/token vs control bracket 5.3977 ms/token (-22.6
us/token, -0.77 tok/s). The residual slowdown is intrinsic: the fused body's
own execution time regressed from 5.98 to 8.23 us because the load-time
persistent weight reads cold from DRAM, while phase-6's per-body store had
kept the weight L2-hot at the cost of the extra kernel. Measured both ways
the fused route costs ~8.2-8.5 us per norm, ~2 us more than the ordinary
split (reduce 3.81 + epilogue 2.18 = 5.99 us), so the +50 us promotion bar is
unreachable through this route. The +495.330 us norms row stays unbooked.**

Scope: `docs/task_workflow/input/nv-reduce-output-phase6-route-efficiency-scope-20260810.md`.
Branch `nvidia-bringup-20260731` (HEAD `e00395b75`). Campaign harness
`extra/llm_research/decode/nv_reduce_output_primitive_ab.py`, run 2026-08-10 on
the RTX 5090 (sm_120, Qwen3-8B-Q4_K_M at fixed depth 512) under the shared GPU
bench lock with a fresh process per arm; record artifact
`docs/task_workflow/output/nv-reduce-output-weight-identity-wall-bracket-20260810.json`.

## What changed

`tinygrad/llm/model.py` (commit `e00395b75`): the reduce-output marker now
prefers a load-time materialized fp16 weight
(`norm._decode_reduce_output_weight`) and falls back to `norm.weight` only
when the materialization is absent. The load path materializes the fp16
weight once per norm for attn/ffn/q/k norms and `output_norm`, mirroring the
M3 route's `_decode_fused_weight`. In phase 6, the marker received the lazy
fp16 cast over quantized storage; `lower_reduce_output_store` then emitted
one `w_buffer.after(w_buffer.store(weight))` materialization per fused body
(the measured `8eeb0be1` per-body store). Binding an identity buffer admits
the body with the weight as a plain PARAM, so the store family disappears
from the decode graph entirely.

## Evidence (2026-08-10, fresh processes under the GPU bench lock)

1. Phase 0 (NV render smoke) PASS: survives sm_120, 896 programs, 18 fused
   `reduce_output_rmsnorm_1_4096` bodies.
2. Phase 1 (exact full-logit gate) PASS: control and candidate logits SHA
   identical `70838f5237ce2cf215e937caed807c4827daa1336e69c3d6c396b1aad4434819`;
   token streams identical, shape `[32, 1, 151936]`, all rows finite, sampled
   token == argmax on every row.
3. Phase 2 (census gate) PASS: fused bodies 18 (control 0); rmsnorm_reduce
   56 -> 38 (drop 18); rmsnorm_epilogue 55 -> 37 (removed 18); q/k norm
   reduce roles unchanged at 36; kernels 936 -> 896 (honest net -40).
   `8eeb0be1` is absent from the candidate census; kernel_us 5903.45 ->
   5891.98 (phase 6) -> 5891.98 (weight-identity), i.e. the store removal
   saved the full per-store kernel sum.
4. Phase 3 (reverse wall bracket) NOT_PROMOTED, identical token streams
   (stream hash `f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905`
   across all three arms), no rejected high-contention samples:

| arm | median ms/token | tok/s | us/token vs bracket median |
| --- | --- | --- | --- |
| control A | 5.3988 | 185.22 | -21.5 (candidate slower) |
| candidate | 5.4204 | 184.49 | baseline |
| control B | 5.3966 | 185.30 | -23.7 (candidate slower) |

   tok/s conversion: control bracket 185.26, candidate 184.49, delta -0.77
   tok/s. Promotion requires candidate >= +50 us/token vs BOTH controls; the
   candidate is ~22.6 us/token SLOWER, so the norms row is not booked.

## Why the body regressed (fresh-process diagnosis)

The campaign's census histogram shows the fused body median jumped from
5.98 us (phase 6) to 8.225 us (weight-identity) with everything else flat.
The 18-kernel store removal was exactly offset. A fresh-process per-kernel
probe (same harness conditions, DEBUG=2 census timing) isolated the two
spellings:

| spelling | kernels | fused bodies | body median |
| --- | --- | --- | --- |
| load-time materialized weight (HEAD) | 896 | 18 | 8.145 us |
| phase-6 per-body store spelling | 914 | 18 | 6.160 us |

The phase-6 spelling's store kernel writes the 8KB weight immediately before
the body, so the body reads it L2-hot. The materialized weight is written
once at load; per token ~500 MB of quantized GEMV weights stream through L2
between norm uses, so the 18 x 8KB norm weights are evicted and the body's
affine epilogue reads them cold from DRAM (~+2 us of exposed latency). The
two spellings cost the same end to end (store 2.56 + body 5.98 = 8.54 us vs
body 8.23 us alone), so neither spelling can win.

## Conclusion and follow-up

- The weight-identity contract is correct and regression-protected:
  `test_reduce_output_helpers_bind_materialized_identity_weight` pins the
  marker binding, and the fp32 end-to-end contract
  (`test_fp32_end_to_end_marker_owns_no_cast_and_body_has_no_fp16_round_points`)
  pins the no-cast/no-fp16-round-point shape of the fp32 route.
- The fused route is at its ceiling: the body must reproduce the ordinary
  r_16_256 serial 256-element chain (32 lanes redundantly per warp) to keep
  the exact-logits gate bitwise, and that association plus launch overhead
  costs ~6.1 us best case vs the ordinary split's 5.99 us. There is no +50 us
  headroom in this route; both spellings land at parity or slightly worse.
- The q/k norms are the unbooked mass of the 495.330 us row: 36 q + 36 k
  reduces and 144 q/k epilogues are untouched because their consumers are not
  C6 chains. Any forward work on the norms row must open the q/k norm route
  (fp32 marker, no cast) and measure a body that beats the split by more than
  launch overhead, or land the single-kernel llama-shaped `rms_norm_f32`
  replacement on the view-preserving boundary.
- No policy promotion: `decode-reduce-output-rmsnorm-route-policy.json` stays
  `promoted_targets: []`; no model wiring change; no default flip.
