# NV reduce-output fp32 (q/k) route scope

Status: **SCOPE for the fp32 q/k norm route** (the unbooked mass of the
+495.330 us norms row).  Feasibility probed on 2026-08-10 (evidence below);
no code change yet.  The route replaces the ordinary q/k norm
reduce+epilogue pair with one cooperative fp32 body per norm, mirroring the
already-proven 4096-dim C6 route (`reduce_output_rmsnorm_1_4096`) and
llama.cpp's single `rms_norm_f32` kernel shape.

Branch `nvidia-bringup-20260731`.  Campaign harness
`extra/llm_research/decode/nv_reduce_output_primitive_ab.py` (extend with the
fp32 route arms).  Probes: `scratchpad/nv_fp32_qk_route_probe.py`,
`scratchpad/nv_fp32_qk_association_probe.py`.

## Population being attacked

The norms row is +495.330 us (ceiling +19.27 tok/s, 207.09 tok/s at full
booking; control at this HEAD is 5.398 ms/token = 185.26 tok/s).  The q/k
norms are the bulk of it: per decode token the census shows

| role | kernels | median us | total us |
| --- | ---: | ---: | ---: |
| q_norm_reduce (`r_2_8_4_4_16`) | 36 | 2.05 | 73.8 |
| k_norm_reduce (`r_8_16_8`) | 36 | 1.75 | 62.9 |
| q_norm_epilogue (`E_2_8_16_4`) | 72 | 1.79 | 128.9 |
| k_norm_epilogue (`E_8_2_16_4`) | 72 | 1.66 | 119.5 |

~385 us of the 495.33 us row sits in the q/k reduce+epilogue pair.  The C6
route (attn/ffn/output norms, fp16 consumer) cannot reach these norms: their
consumers are not C6-chain CALL arguments, and the marker itself never
exists because `_semantic_reduce_output_rmsnorm` requires `rows == 1`.

## What the production graph does today (probed 2026-08-10)

1. `_decode_reduce_output_rmsnorm(self.attn_q_norm, q, promoted)` is called
   with `q` at shape `(1, 32, 1, 128)` (post reshape+transpose) and
   `(1, 8, 1, 128)` for k.  The marker helper bails on `rows != 1` BEFORE a
   `REDUCE_OUTPUT` uop exists.  A marker-call probe counted 145 calls
   (73 x rows=1 attn/ffn/output + 36 x rows=32 q + 36 x rows=8 k) and the
   selector trace saw only the `16x32x8` association (18 accepted, 76
   `marker_not_eligible`).  **Blocker A: the marker rows check.**
2. The ordinary q/k reduce kernels are single kernels over ALL rows:
   `r_2_8_4_4_16` (32 rows x 128) and `r_8_16_8` (8 rows x 128), followed by
   the elementwise epilogue kernels `E_2_8_16_4` / `E_8_2_16_4`.  The reduce
   computes the per-row sumsq -> scale (`CONST 0.0078125` = 1/128, `1e-6`,
   `RECIPROCAL` stores at output offsets `l(i)`, `l(i)+4`, `l(i)+8`,
   `l(i)+12`), i.e. the scale is computed IN the reduce kernel and the
   epilogue is `x * scale[row] * w[i]`.
3. The normed q/k is consumed by `apply_rope` (ordinary elementwise) then the
   attention custom kernels.  There is no C6 `CONTIGUOUS(RESHAPE(MS(...)))`
   CALL argument; the marker value feeds an ordinary elementwise through a
   `PERMUTE` view.  **Blocker B: the lowering must bind the fused output
   through the PERMUTE view instead of a C6 CALL arg.**
4. The marker input proof for q/k: `x.uop` is `PERMUTE(RESHAPE(...))` of the
   q4k GEMV (precompiled) output; the marker's identity walk stops at
   `PERMUTE`, and `_identity_buffer_view` rejects permutes.  **Blocker C:
   extend the identity proof through pure PERMUTE views (offset-0,
   permutation-only) with a fail-closed validator.**

## Build plan (fail-closed, bitwise-exact)

1. Marker (`tinygrad/tensor.py` `_semantic_reduce_output_rmsnorm`):
   accept `rows in (1, 8, 32)` with `dim in (128, 4096)`; derive the spec
   association from the ordinary reduce shape the way the 4096 route already
   does (`warps = max(1, min(16, dim // 256))`, etc.) and carry `rows` in the
   `ReduceOutputSpec`; extend the identity walk to pure PERMUTE views over
   precompiled outputs.
2. Emitter (`tinygrad/codegen/late/reduce_output.py`): multi-row bodies.
   One kernel, `rows` independent per-row reductions + the existing
   serial-chain epilogue, mirroring the ordinary reduce's per-thread serial
   chains and cross-thread combine order (same proof pattern as the
   16x32x8 body: every lane redundantly computes the same serial chain, lane
   0 publishes, serial partial chain, one barrier).  The q reduce's chain is
   the nested 8x16 serial structure; the body must reproduce it bitwise.
3. Lowering (`tinygrad/schedule/rangeify.py`): admit the fp32 marker through
   the PERMUTE carrier (new carrier form, typed offset-0 view validator) and
   bind the fused body's output buffer as the marker value so the rope
   elementwise and the attention chain read it unchanged.
4. Weights: the fp32 route binds `norm.weight` (fp32 parameter, plain buffer)
   directly; no fp16 materialization, no cast, no fp16 round points
   (llama `rms_norm_f32` contract, pinned by
   `test_fp32_end_to_end_marker_owns_no_cast_and_body_has_no_fp16_round_points`).
5. Tests: extend `test_reduce_output_rmsnorm.py` and
   `test_generic_reduce_output.py` with the multi-row bitwise tripwires
   (rows 8/32 x 128, fp32/fp32 and fp32/fp16 weight mixes, body == ordinary
   bitwise on CPU), the marker ownership contract (no owned cast), and the
   PERMUTE-carrier admission tests.

## Campaign extension and measurement plan

Extend `extra/llm_research/decode/nv_reduce_output_primitive_ab.py` (or a
sibling fp32 harness) with:

1. Phase 0 smoke: candidate renders on sm_120 with the fused q/k bodies
   (`reduce_output_rmsnorm_32_128`, `reduce_output_rmsnorm_8_128`) in the
   compiled set.
2. Phase 1 exact full-logit gate: fp32 SHA-256 identical to control, token
   stream identical, per-row argmax == sampled token.
3. Phase 2 census: fused bodies 91 (36 q + 36 k + 19 C6); q/k reduce ledger
   roles 36 -> 0 each (the 17 q + 17 k warp-coop materialized reduces persist
   as real kernels under their own `r_32_32_4_4_*` / `r_8_32_4_4_*` names and
   are reported as exact side effects); q/k epilogue roles 72 -> 0 each (two
   epilogue kernels per ordinary norm, so the drop is 2 x the body count); C6
   fuses 18 block norms plus the final norm (rmsnorm reduce 56 -> 37,
   epilogue 55 -> 37, final epilogue role 1 -> 0); net norms kernels
   -254 (328 -> 74); honest net program delta reported with exact families.
4. Phase 3 reverse control/candidate/control wall bracket under the shared
   GPU bench lock, +50 us/token bar vs BOTH controls, token-stream hash
   identical across arms.  Promotion books the q/k share of the norms row;
   the +495.330 us row books only when the full row (C6 + q/k + output
   norm) promotes.

Expected census shape (production-observed, logits-verified at
`420c4afc2` + the reduce-collapse fix at `99d6e0fda`): per decode token every
q/k marker admits (36 q + 36 k fused bodies).  The 17 q + 17 k warp-coop
carriers keep their exact partials REDUCE as a materializing kernel, now
rendered as a real reduce under its own `r_32_32_4_4_*` / `r_8_32_4_4_*`
name (the pre-fix silent `E_*` copies are gone; values are bitwise-identical
to control, full-logit SHA-256 gate PASS at `70838f52...`).  The ledger's
q/k reduce roles count only the ordinary `r_2_8_4_4_16` / `r_8_16_8` names,
so they drop 36 -> 0 with the materialized reduces reported as side effects.
The C6 route fuses 19 bodies (18 block norms + the final norm).  Net norms
kernels -254 (328 -> 74); the +495.330 us row books only when the full row
(C6 + q/k + output norm) promotes under the reverse wall bracket.

## Risks and open questions

- Bitwise association: the ordinary q reduce is a nested 8x16 serial chain
  with a cross-thread combine order that must be mirrored exactly.  The
  08-10 4096-dim proof (commit `25370bbad`) is the template; the tripwire
  tests fail closed before any GPU arm if the body drifts by 1 ulp.
- PERMUTE-view proof: must reject offset/stride permutations and stay
  fail-closed (mirror `_identity_buffer_view`'s tests).
- Wall win is uncertain until the bracket runs: the fused body replaces two
  kernels with one, and the phase-6 evidence says small-kernel launch
  overhead is partially hidden by queue depth.  The bracket is the
  authority; the scope does not pre-book the row.
- The fp16 route's load-time weight materialization
  (`_decode_reduce_output_weight`) stays fp16 for the C6 route; the fp32
  route must NOT use it (it would round the weight and break bitwise
  equality with the ordinary fp32 epilogue).

## Out of scope

Flash row (+163.029 us), residual/cast/contiguous row, rope fusion, the
view-preserving opaque boundary (M3 reopen condition), and any change to
the ordinary q/k path defaults.  No policy promotion without a promoted
bracket; `decode-reduce-output-rmsnorm-route-policy.json` stays
`promoted_targets: []`.
