# NV residual-family epilogue absorption scope (M2a fp16 store, then M2b/M2c, then M1)

Date: 2026-08-10
Branch: `nvidia-bringup-20260731`. M2a BOOKED by the full AB: smoke PASS,
exact-logits SHA PASS (bitwise), census PASS (E_128_32_3 36 -> 0, fused16
x36, net -36, no unrelated shift), reverse wall bracket PROMOTED at
+57.2 us/token vs the control bracket median (56.2 vs A, 58.2 vs B).
Status: **IN PROGRESS.** M2a books +57.2 us of the +240.106 us
residual/cast/contiguous row (candidate 5.2723 ms/token = 189.67 tok/s vs
control bracket 5.3295 ms = 187.64 tok/s). ~183 us of the row remain
(M2b ffn_down fp16 store, M2c residual add in-kernel, then M1 norm
epilogue). The M2b/M2c/M1 gate sequence stays the booking authority.

## 1. Ledger position (per token)

| item | value |
| --- | --- |
| BOOKED candidate (fp32 q/k route, `68669d348`) | 5.3235 ms/token = 187.85 tok/s (+83.5 us vs control 5.4071) |
| **M2a booked (this scope)** | **candidate 5.2723 ms/token = 189.67 tok/s, +57.2 us/token vs control bracket 5.3295 ms (census: 36 casts -> 0, net -36 kernels)** |
| Target | ~194 tok/s = 5.1546 ms/token -> need ~-169 us wall |
| norms row | +495.330 us attribution, mostly tapped: q/k bodies booked; remaining 37 chains (r_16_256 3.84 us + E_32_32_4_f14a5cc0 2.27 us each ~= 226 us census) |
| **residual/cast/contiguous row** | **+240.106 us attribution, +8.87 tok/s ceiling; M2a books -57.2 us; ~183 us left for M2b/M2c/M1** |
| quant GEMV | ~4050 us census (70%): q4k/q6k; 10% win ~= 203 tok/s (after 194) |

The residual family is the clean win: per-row epilogue ops, no redundant
compute, existing `Q4KGEMVEpilogue` machinery for the add. The census-to-wall
mapping observed in the fp32 q/k booking was 0.61 for body-adding changes;
for pure kernel REMOVAL expect ~1.0+ (kernel time + launch gap).

## 2. Residual-family anatomy (structural probe, execution order, block 0)

```
attn path -> q4k_g3_lanemap_gemv_epi_resadd_4096_4096 (11.8 us)
r_16_256 + E_32_32_4_f14a5cc0 (ffn norm) -> w1w3fused_12288_4096 (39-46 us)
E_128_32_3 (ffn_activation_cast, 36 x 1.6 = 57.6 us)   <- w1w3fused output fp32->fp16
ffn_down: q6k_gen_coop_4096_12288_inkernel (shared, 18 x 34.8 = 626 us) /
          q4k_g3_lanemap_gemv_4096_4096 (non-shared, 19 x 9.66)
E_32_32_4_02a9738c (ffn_down_cast, 36 x 1.7 = 61.7 us)  <- ffn_down output fp32->fp16
E_32_32_4_fab82d40 (ffn_residual_add, ~49 x 1.79 = 87.7 us) <- h + ffn_out
```

Observed `E_32_32_4` hashes: `f14a5cc0` x37 (norm epilogue), `02a9738c` x36
(ffn_down cast), `fab82d40` x49 (ffn residual add), `5a5673a4` x1. NOTE:
`E_32_32_4_0a5eb0ac` (attention cast x36) appeared in the fp32 q/k record but
was ABSENT in the latest probe run (prompt-depth/session variation); recheck
it in the census gate before booking any attention-side claim. `E_128_32_3`
= ffn_activation_cast x36.

## 3. Absorption plan (M2 = residual family; M1 = norm epilogue after)

**M2a (BOOKED +57.2 us/token): w1w3fused stores fp16 -> kills E_128_32_3
(-57.6).** New kernel variant `q4k_g3_lanemap_gemv_w1w3fused16_{rows}_{k}`
(scalar style only; the fp32 name/hash is unchanged). The store is the same
fp32 expression wrapped in one half cast, so the bytes are bitwise-identical
to the separate cast kernel (cvt.rn.f16.f32). Wired under an explicit
research lease (`_q4k_w1w3_fp16_store_lease` on model + blocks; no loader
policy creates it). Route: `q4k_gate_up_primitive_linear_call(...,
store_fp16=True)` picks the variant and an fp16 `OutputSpec`.

**Root cause discovered during NV verification: the cast folds, the COPY
does not.** The consumer prelude
`x[:, 0, :].reshape(K).cast(fp16).contiguous()` cannot fold `contiguous()`
over the opaque program CALL/AFTER boundary: the compiled candidate still
contained one `E_128_32_3` per layer as a pure fp16->fp16 COPY
(`half* data0, half* data1`) after the real fp32->fp16 cast folded. The
absorption therefore needs a typed boundary, not just a store-dtype change.

**M2a mechanism (landed, M5 boundary extension).** Producer side:
`DeclaredTypedOutput` gains `epilogue_absorption_admitted`; the fused16
w1+w3 program declares its fp16 layout when `store_fp16=True`
(`combine_fusion_admitted=False`, producer-side lease only). Consumer side:
`TypedViewRequest` gains `requires_epilogue_absorption` (mutually exclusive
with `requires_combine_fusion`); the Q4K and Q6K ffn_down GEMV programs
(`decode_q4k*`/`decode_q6k*` + `.gemv`) issue the request for their fp16
K-vector slot when `route_role == "ffn_down"`. The M5 validator
(`_validated_typed_view`) is fail-closed: no producer declaration, closed
gate, wrong route role, or non-GEMV program identity -> the generic
flat-buffer ABI (with its materializing copy) is used unchanged, so control
is byte-identical. Hermetic tests cover the fold, the fail-closed rejections,
and the Q6K request wiring; the NV probe confirmed zero `E_128_32_3` bodies
under the lease.

**M2b: ffn_down GEMVs store fp16 -> kills E_32_32_4_02a9738c (-61.7).** The
q4k `q4k_g3_lanemap_gemv_kernel` already has an `fp16_cast` epilogue kind
(`Q4KGEMVEpilogue("fp16_cast")`); the q6k
`q6k_gen_coop_4096_12288_inkernel` needs its own variant. Both under the same
lease family; census reference updated for the new kernel names.

**M2c: ffn_down residual add in-kernel (`total + h[row]`, like attention's
`residual_add`) -> kills E_32_32_4_fab82d40 (-65+).** Must match the ordinary
add's dtype exactly (recheck `E_fab82d40` source: likely fp16 add of fp16
inputs; the attention resadd epilogue takes an fp32 residual, so the ffn
epilogue is a distinct variant or an fp16 residual slot).

**M1 (after M2): ffn-norm epilogue into w1w3fused via the existing research
route `q4k_gate_up_rms_affine_qualification_call`.** CAUTION: its scale uses
`.sum()` (association mismatch vs `r_16_256`) and its warp reduce differs
from the landed scalar kernel (`_warp_reduce_sum_staged` vs
`_lane_partition_reduce_sum`) -> NOT bitwise. Must mirror the landed scalar
kernel and use the bitwise-exact scale
`(h.float().square().mean(-1,keepdim=True)+eps).rsqrt()` (phase6
`candidate_topology_probe` verified this expression bitwise-equal to
`nn.RMSNorm`, `fused_epilogue_bitwise_equal=True`). Kills E_f14a5cc0 x36
(-82) but beware GEMV redundancy cost (quad-style staging regressed in-loop
before). M1 is deferred until M2 books.

## 4. A/B gate order (harness: sibling of
`extra/llm_research/decode/nv_reduce_output_fp32_qk_ab.py`)

1. Phase 0 smoke: candidate renders on sm_120, `q4k_g3_lanemap_gemv_w1w3fused16_*`
   in the compiled set; control fails closed if any lease route appears.
2. Phase 1 exact full-logit gate: fp32 SHA-256 identical to control over the
   stacked rows, token stream identical, per-row argmax == sampled token.
3. Phase 2 census: `E_128_32_3` 36 -> 0; `w1w3fused16_*` x36 present; all
   other program counts identical to the control arm; honest net program
   delta -36 reported with exact names. FAIL CLOSED if the cast remains or
   the variant is absent.
4. Phase 3 reverse control/candidate/control wall bracket under the shared
   GPU bench lock (`flock -w 60 /tmp/gpu-bench.lock`), +50 us/token bar vs
   BOTH controls, token-stream hash identical across arms.

The control arm is the booked fp32 q/k candidate conditions (callify flags +
`_decode_reduce_output_rmsnorm_promoted` on model + blocks +
`_decode_direct_greedy_promoted`), WITHOUT the M2 lease. The candidate arm
adds `_q4k_w1w3_fp16_store_lease` on the model and every block. Both arms
run as fresh processes so JIT capture and allocator state cannot leak.

Harness construction notes from the booking run: (1) BOTH arms must run with
the callify Context flags live (`_m2_arm_context`), because the M2 control
IS the booked fp32 q/k candidate -- reusing the fp32 q/k campaign's
closed-graph control context made the control arm render the generic final
norm chain (r_16_256 + per-layer copies) and the census failed closed with
unrelated deltas; (2) the M2 logits gate wraps the shared C6 validator with
the M2 schema name, so it must import `validate_logits_gate` from
`nv_reduce_output_primitive_ab` (schema-agnostic), not the fp32 q/k wrapper;
(3) children must run with `PYTHONPATH=<repo root>` (the harness spawns
script-path children, which do not see the repo root on `sys.path`).
Observed booking evidence: control bracket 5.3295 ms/token (median of
5.3285 / 5.3305), candidate 5.2723 ms/token, +57.2 us/token; census
cast_control 36 -> cast_candidate 0, fused16 0 -> 36, net -36, all other
program counts identical; logits SHA-256 identical
(`70838f52...4434819`) with identical token streams.

## 5. Risks and open questions

- The census reference in the fp32 q/k harness counts
  `E_32_32_4_fab82d40f922cf5fz` as 9; the structural probe observed 49
  instances. The M2 census gate must use the freshly measured control arm as
  its reference, never a stale constant.
- `E_32_32_4_0a5eb0ac` (attention cast) presence varies by run; do not book
  attention-side claims from the probe.
- Wall win is uncertain until the bracket runs; kernel removal saves kernel
  time + launch gap, but queue depth can hide small-kernel launches. The
  bracket is the authority; this scope does not pre-book the row.
- M2c's residual slot dtype (fp16 vs fp32) must match the ordinary add
  bitwise; the attention resadd epilogue's fp32 slot is NOT the same
  contract.
