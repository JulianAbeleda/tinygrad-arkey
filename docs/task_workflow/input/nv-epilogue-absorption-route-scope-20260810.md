# NV residual-family epilogue absorption scope (M2a fp16 store, then M2b/M2c, then M1)

Date: 2026-08-10
Branch: `nvidia-bringup-20260731`. M2a BOOKED by the full AB: smoke PASS,
exact-logits SHA PASS (bitwise), census PASS (E_128_32_3 36 -> 0, fused16
x36, net -36, no unrelated shift), reverse wall bracket PROMOTED at
+57.2 us/token vs the control bracket median (56.2 vs A, 58.2 vs B).
M2b LANDED but NOT BOOKED: the ffn_down in-kernel residual add runs and
passes the exact-logits and census gates, but the reverse wall bracket
measured candidate 5.2516 ms/token = ~190.4 tok/s at +27.5/+32.3 us
(median +29.9 us) vs the control bracket, BELOW the +50 us promotion bar
(verdict NO-GO). The absorbed GEMV bodies cost ~+14 us census over the
plain GEMVs and the ffnresadd add is in-kernel, so the real wall win is
~30 us, not the ~61.7 us census drop. M2c LANDED and BOOKED the combined
M2b+M2c package: the declared-AFTER output-slot rebind folds the fp32
block-output copy family `fab82d40` x49 (pure fp32 identity copies, so
byte-identical), and the reverse wall bracket measured candidate
5.2031 ms/token = ~192.2 tok/s at +66.2/+78.9 us (median +72.6 us) vs the
control bracket, ABOVE the +50 us promotion bar. Status: **M2a BOOKED;
M2b complete (absorbed into the M2c booking); M2c BOOKED; next: the
attention cast `0a5eb0ac` x36, then M1 norm epilogue.** M2a books +57.2 us
of the +240.106 us residual/cast/contiguous row (candidate 5.2723 ms/token
= 189.67 tok/s vs control bracket 5.3295 ms = 187.64 tok/s). M2b eliminates
the `E_32_32_4_02a9738c` residual-add family (36 x ~1.7 us = ~61.7 us
census) AND the transport-copy family `E_32_32_4_86a23e1a` x36 (an M2c-row
item) via the attn_qo typed-output declaration; M2c then folds `fab82d40`
x49. What remains in the row: the attention cast `0a5eb0ac` x36 (~60 us),
then M1 norm epilogue. The M2b/M2c/M1 gate sequence stays the booking
authority.

## 1. Ledger position (per token)

| item | value |
| --- | --- |
| BOOKED candidate (fp32 q/k route, `68669d348`) | 5.3235 ms/token = 187.85 tok/s (+83.5 us vs control 5.4071) |
| **M2a booked (this scope)** | **candidate 5.2723 ms/token = 189.67 tok/s, +57.2 us/token vs control bracket 5.3295 ms (census: 36 casts -> 0, net -36 kernels)** |
| **M2b (landed, NOT booked)** | **candidate 5.2516 ms/token = ~190.4 tok/s; wall +27.5 us (vs control A 5.2791) / +32.3 us (vs control B 5.2840), median +29.9 us < +50 us promotion bar (NO-GO); census PASS with swap-twin rule (36 `*_epi_ffnresadd` bodies 1:1, 0 residual adds, 0 `86a23e1a` transport copies); net -36 kernels** |
| **M2c booked (M2b+M2c package, this scope)** | **candidate 5.2031 ms/token = ~192.2 tok/s; wall +66.2 us (vs control A 5.2693) / +78.9 us (vs control B 5.2820), median +72.6 us > +50 us promotion bar (PROMOTED); census PASS: `fab82d40` copies 49 -> 0, `02a9738c` adds 36 -> 0, `*_epi_ffnresadd` 0 -> 36, `0a5eb0ac` attention cast 36 == 36 (M5-closed), net -85 (715 -> 630 kernels); logits SHA-256 identical** |
| Target | ~194 tok/s = 5.1546 ms/token -> need ~-169 us wall |
| norms row | +495.330 us attribution, mostly tapped: q/k bodies booked; remaining 37 chains (r_16_256 3.84 us + E_32_32_4_f14a5cc0 2.27 us each ~= 226 us census) |
| **residual/cast/contiguous row** | **+240.106 us attribution, +8.87 tok/s ceiling; M2a books -57.2 us; M2b+M2c books the combined package at -72.6 us wall (candidate 5.2031 ms = ~192.2 tok/s, census net -85: `02a9738c` adds + `fab82d40` copies + `86a23e1a` transport copies gone); remaining in-row: attention cast `0a5eb0ac` x36 (~60 us census) as the next item; M1 norm epilogue is its own row** |
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
E_32_32_4_02a9738c (fp32 h + ffn_out residual add, 36 x ~1.7 = 61.7 us)
E_32_32_4_fab82d40 (fp32 block-output copies, ~49 x 1.79 = 87.7 us)
E_32_32_4_0a5eb0ac (attention residual cast fp32->fp16, x36)
```

Observed `E_32_32_4` hashes: `f14a5cc0` x37 (norm epilogue), `02a9738c` x36
(fp32 residual add `val0.x + val1.x` float4, source-verified), `fab82d40`
x49 (fp32 copies, source-verified), `0a5eb0ac` x36 (attention residual
cast), `5a5673a4` x1. The old "ffn_down cast" and "ffn residual add" labels
were wrong: the 02a9738c family is the standalone `h + ffn_out` add and
fab82d40 is a copy family. `E_128_32_3` = ffn_activation_cast x36 (M2a
absorbed it).

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

**M2b (LANDED as the in-kernel residual add): ffn_down Q4K/Q6K GEMVs add the
hidden-state residual h in-kernel (`total + h[row].cast(fp32)`, fp32 store)
under the harness-installed `_ffn_down_resadd_lease` (model + blocks +
ffn_down linears) -> kills E_32_32_4_02a9738c x36 (-61.7).** The GEMVs render
their own `q4k_g3_lanemap_gemv_epi_ffnresadd_*` / `q6k_gen_coop_*_epi_ffnresadd`
variants; the standalone add folds away. The in-kernel add is the same fp32
expression the separate add kernel lowers and fp32 addition is commutative,
so the stored bytes are bitwise-identical to control.

**M2b premise correction (critical): the fp16-store spelling is NOT
bitwise-safe and was abandoned before landing.** The original plan was for
the ffn_down GEMVs to store fp16 (killing 02a9738c as a cast). The census
and source probes showed 02a9738c is not a cast at all: it is the fp32
`h + ffn_out` add, and the next block's attention residual epilogue
(`residual_for_output=x`, decode_routes attn_qo resadd branch) consumes the
fp32 block output. Storing the block output fp16 would round the residual
and break the exact-logits gate. The residual-slot bitwise discovery: the
attention resadd epilogue casts the fp32 residual to fp32 on read, so the
fp32 block output is the correctness contract; M2b therefore keeps the fp32
store and absorbs the add instead of the cast.

**M2c: kill the fp32 copy family `fab82d40` x49 (~87.7 us) and the attention
cast `0a5eb0ac` x36.** The copies are produced by the typed-boundary
flat-buffer ABI path (the same M5 mechanism that left the fp16 COPY in M2a);
the residual slot must stay fp32 for the attention resadd epilogue, so M2c
is about folding the contiguous boundary, not changing dtype. M2b already
closed one M2c-row sub-item: the `E_32_32_4_86a23e1a` x36 transport-copy
family, which only rendered because the attn_qo GEMV had no declared typed
output (the M2b attn_qo declaration makes the ffn_down residual fold accept).
~121 us of the row remain after M2b.

**M2c LANDED and BOOKED (the copy fold; the attention cast stays M5-closed
for now).** `_declared_epilogue_absorption_after` /
`_declared_after_output_slot_rebind` in `tinygrad/callify.py` prove, fail-closed,
that the absorbed block's result bottoms at a producer-declared
`epilogue_absorption_admitted` AFTER (`MEMORY_SEMANTIC(RESHAPE(AFTER(PARAM,
CALL)))`), then rebind the nested CALL's output PARAM to a view of the caller
output slot; `_precompiled_output_redirect` returns the bare AFTER (no STORE),
so the CALL is the sole writer and no boundary copy can render. The fold is
value-transparent (pure fp32 movement), so stored bytes stay bitwise-identical.
The `0a5eb0ac` attention cast is deliberately left out of the M2c contract:
the census gate asserts it stays 36 == 36 between arms (M5-closed) and it
remains the next in-row item (~60 us census).

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

1. Phase 0 smoke: candidate renders on sm_120, `*_epi_ffnresadd` bodies in
   the compiled set and no `E_32_32_4_02a9738c`; control fails closed if any
   M2b lease route appears.
2. Phase 1 exact full-logit gate: fp32 SHA-256 identical to control over the
   stacked rows, token stream identical, per-row argmax == sampled token.
3. Phase 2 census: `E_32_32_4_02a9738c` 36 -> 0; `*_epi_ffnresadd` x36
   present 1:1 with the control add count; the M2a fused16/cast families
   byte-identical between arms (both arms run the booked M2a candidate);
   honest net program delta -36 reported with exact names. FAIL CLOSED if
   the add remains, the bodies are absent, or any unrelated count shifts.
   The M2b swap-twin rule: `q4k_g3_lanemap_gemv_4096_12288` x18 and
   `q6k_gen_coop_4096_12288_inkernel` x18 may swap 1:1 with their
   `*_epi_ffnresadd` twins as an allowed side effect, and the gate fails
   closed if an ffnresadd body has no plain twin.
4. Phase 2b (M2c) census extension: `E_32_32_4_fab82d40` 49 -> 0 with the
   drop equal to the control copy count; `E_32_32_4_0a5eb0ac` stays
   36 == 36 (M5-closed); net program delta -85 = -(add drop 36 + copy drop
   49). FAIL CLOSED if any copy remains or the attention cast shifts.
5. Phase 3 reverse control/candidate/control wall bracket under the shared
   GPU bench lock (`flock -w 60 /tmp/gpu-bench.lock`), +50 us/token bar vs
   BOTH controls, token-stream hash identical across arms.

The M2b control arm is the BOOKED M2a candidate (callify flags +
`_decode_reduce_output_rmsnorm_promoted` on model + blocks +
`_decode_direct_greedy_promoted` + the M2a `_q4k_w1w3_fp16_store_lease`),
WITHOUT the M2b `_ffn_down_resadd_lease`. The candidate arm adds the M2b
lease on the model, every block, and every ffn_down linear. Both arms run
as fresh processes so JIT capture and allocator state cannot leak.

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
(`70838f52...4434819`) with identical token streams. M2b bracket evidence:
candidate 5.2516 ms/token = ~190.4 tok/s, +27.5 us vs control A 5.2791 /
+32.3 us vs control B 5.2840 (median +29.9 us), verdict NO-GO at the
+50 us bar; census gate_pass true (adds 36 -> 0, ffnresadd 0 -> 36,
`86a23e1a` 36 -> 0); logits SHA-256 identical. M2c booking evidence
(`docs/task_workflow/output/nv-epilogue-absorption-m2c-ab-20260811.json`):
candidate 5.2031 ms/token = ~192.2 tok/s, +66.2 us vs control A 5.2693 /
+78.9 us vs control B 5.2820 (median +72.6 us), verdict PROMOTED at the
+50 us bar; census gate_pass true (adds 36 -> 0, ffnresadd 0 -> 36,
copies 49 -> 0, attention cast 36 == 36, net -85, 715 -> 630 kernels);
logits SHA-256 identical; all three bracket token-stream hashes equal.

## 5. Risks and open questions

- The census reference in the fp32 q/k harness counts
  `E_32_32_4_fab82d40f922cf5fz` as 9; the structural probe observed 49
  instances. The M2 census gate must use the freshly measured control arm as
  its reference, never a stale constant.
- `E_32_32_4_0a5eb0ac` (attention cast) presence varies by run; it is NOT
  part of the M2b contract (only 02a9738c is absorbed) and attention-side
  claims are not booked from the probe.
- Wall win is uncertain until the bracket runs; kernel removal saves kernel
  time + launch gap, but queue depth can hide small-kernel launches. The
  bracket is the authority; this scope does not pre-book the row.
- The fp16-store spelling of M2b is NOT bitwise-safe: the next block's
  attention residual consumes the fp32 block output (residual-slot
  discovery, section 3 M2b). M2c must keep the fp32 store and fold the
  contiguous boundary instead of changing dtype.
