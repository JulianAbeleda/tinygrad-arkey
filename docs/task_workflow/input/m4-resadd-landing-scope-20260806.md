# M4 residual_add landing scope - fold the residual slot, promote one variant, book the recovery

Date: 2026-08-06
Branch boundary: tinygrad `nvidia-bringup-20260731`, HEAD `3f0784e7b`
Status: **implementation scope. The two P0 probes PASSED; this scope authorizes the
production fold, the per-variant promotion record, and the section-6 full gate. The
ffn_down prelude stays rejected; the combined M4 record stays closed; M3/M5/Path 3 stay
closed. No promotion of anything else.**

## 1. Why this lands now

`m4-variant-reopen-boundary-p0-scope-20260806.md` authorized two test-first probes; both
cleared their gates on 2026-08-06:

- **Probe 1 (hermetic PASS)**: the o-proj residual slot
  `epi_inputs["residual"][:, 0, :].reshape(N).cast(fp32)` (`decode_routes.py:97-98`) is a
  provably pure, offset-0 view chain over the precompiled block-output boundary
  (`MS(CONTIGUOUS(GETTUPLE(FUNCTION)))`). The extended typed-view contract folds it with
  ZERO materialization (1 boundary copy without the ABI, 0 with the fold). Layer-0 embedding
  fails closed. M5's validator untouched (27/27 green).
- **Probe 2 (GPU PASS)**: isolated one-variant-open census, same-session control+variant:
  fused `q4k_g3_lanemap_gemv_epi_resadd_4096_4096` 9.60us vs legacy 9.38us (+2.3%, under
  the 10% material bound; same-session legacy 9.645us, parity). Copy class
  `E_32_32_4_86a2` adds 72 launches/token @ 1.5us = **108us/token mass**. Copy-free ceiling
  **+100.1us/token measured** (book +45.0us). Control and variant token streams identical
  3/3, so the fused kernel is bitwise-exact. Wall: control 180.36, variant 178.77 (-0.88%
  with the copies present), matching the M4 isolated row shape.

Records: `m4-residual-boundary-fold-probe-record-20260806.md`,
`m4-resadd-kernel-microgate-record-20260806.md`.

The parity ledger (`nv-decode-parity-campaign-reconciled-ledger-20260805.md`) closed
"Attention-O custom epilogue and FFN-down residual composition" as TOPOLOGY NO-GO with 0
credit, attributing the failure to the 70+ `E_86a2` boundary copies and the FFN-down
recompute. The P0 probes isolate the residual_add half as clean: this landing books that
half as a measured recovery instead of leaving it inside a rejected composition.

## 2. Production shape (four deltas, one record)

### 2.1 New per-variant promotion record

The combined `decode-q4k-epilogue-fusion-route-policy.json` gates ALL three M4 variants
(residual_add, fp16_cast, ffn_down_fused) from ONE target set, and stays CLOSED. Add a
SEPARATE record `decode-q4k-epilogue-resadd-route-policy.json` (schema
`boltbeam.route_policy.v1`, same family as w1w3/kv-store), `promoted_targets: []`
initially, gaining `NV sm_120` ONLY after the section-6 gate passes. This is the same
pattern the w1w3 record uses to promote one route without disturbing the combined record.

- `tinygrad/llm/model_route_plan.py`: `load_decode_q4k_epilogue_resadd_promotion` +
  `_DECODE_Q4K_EPILOGUE_RESADD_PROMOTED_TARGETS` +
  `decode_q4k_epilogue_resadd_promoted(target)`.
- `tinygrad/llm/qk_primitives.py`: `QKPrimitiveRouteAdmission` gains
  `q4k_epilogue_resadd_promoted: bool = False` and property
  `q4k_epilogue_resadd_admitted` (`admitted and q4k_epilogue_resadd_promoted`); the install
  site resolves the new field from the new record (5th positional arg, same resolve-once
  pattern as the other four).
- `tinygrad/llm/model.py`: per-block flag `_decode_q4k_epilogue_resadd_promoted` (set at
  load beside `_decode_q4k_epilogue_fusion_promoted`, model.py:1605-1607); `_epi_residual`
  (model.py:620) keys off the new flag + `q4k_epilogue_resadd_admitted`. The combined
  `_epi_fused` flag stays for ffn_down/attn_kv ONLY, so those variants cannot fire.

### 2.2 Residual-slot typed view in the ABI family

`tinygrad/llm/kernel_program.py` gains a SECOND, distinct request type so M5 is untouched:

- `ResidualViewRequest` dataclass: `slot`, `dtype`, `flat_shape`, `route_role`, `kind`
  (default `"residual_add"`). Deliberately NOT a `TypedViewRequest`; `KernelProgram` gains
  `residual_input_views: tuple[ResidualViewRequest, ...] = ()` with unique-slot validation.
- `_validated_residual_view(uop, request, program)`: the probe-1 extended contract ported
  verbatim - residual-slot opt-in (slot 2, attn_qo, residual_add, q4k gemv consumer),
  movement-only chain (RESHAPE identity, SLICE offset-0, PERMUTE/EXPAND identity,
  CONTIGUOUS/MS transparent), dtype/numel preserved, base identity
  (`has_buffer_identity` or `has_precompiled_output_identity` or CONTIGUOUS over either,
  or AFTER with a declared typed output - the M5 path kept intact). Fail-closed: every
  mismatch rejects to the generic flat-buffer ABI.
- `_fold_residual_input_views(inputs, program)`: substitute `Tensor(view)` (the validated
  base) for the materialized request, run inside `_execute_outputs` beside
  `_fold_typed_input_views`. Rejection leaves the input untouched (byte-identical
  flat-buffer ABI).

`tinygrad/llm/decode_routes.py`:

- The attn_qo residual_add branch keys off `q4k_epilogue_resadd_admitted` (NOT the combined
  flag) and issues `ResidualViewRequest(slot=2, dtype=fp32, flat_shape=(binding.N,),
  route_role="attn_qo", kind="residual_add")` on the `KernelProgram`.
- ffn_down and attn_kv branches stay on the combined `q4k_epilogue_fusion_admitted`
  (closed default -> cannot fire).

### 2.3 What does NOT change

- `decode_kernels.py`: no emitter change; the epi_resadd kernel body and name are already
  measured and bitwise-exact.
- `UOp.custom_kernel` defaults, `ops.py` defaults: untouched.
- M5 (`TypedViewRequest`, `_validated_typed_view`, `_fold_typed_input_views`): untouched,
  all 27 tests must stay green.
- The combined M4 record, M3, M5, Path 3, w1w3, kv-store records: untouched.
- Legacy `q4k_g3_lanemap_gemv_4096_4096` rendered HIP sha `27857cb8ca03` must stay
  byte-identical (the epi variant is a NEW kernel name; the legacy emitter is unchanged).

## 3. Section-6 full gate (before the record gains NV sm_120)

Same-session, lock-held (`flock -w 600 /tmp/gpu-bench.lock`), Qwen3-8B-Q4_K_M, nmeas 20,
reps 3, median tok/s, fused prefill attention disabled. Open mode = gate forced open via
the module override (`mrp._DECODE_Q4K_EPILOGUE_RESADD_PROMOTED_TARGETS =
frozenset({("NV","sm_120")})`) + the production fold ACTIVE. Closed mode = default
records (fold dormant).

1. d512/d2048/d4096 wall, open mode: must not regress the M2-on baseline
   (172.80 / 161.50 / 149.00 tok/s, `nv-decode-parity-final-20260802.md`) and must show a
   positive delta attributable to the copy-free residual_add fusion.
2. Census assertions (d512, open mode): `E_32_32_4_86a2` count 0 for the residual x slot,
   `E_32_32_4_02a` count 0 for attn_qo, epi_resadd count 36, legacy attn_qo gemv count 0.
   The residual-slot copy count is the load-bearing assertion; the residual-add elementwise
   absorption (72 -> 36) is the second.
3. Pins 3/3 at every depth, both modes. NOTE: the scope's original pins
   (`9d6b3787...`, first `151936`) do NOT reproduce at this HEAD because the 08-05
   decode-parity campaign changed the kernel mix. Re-derived pins at this HEAD (control arm,
   probe-2 record): token sha `227ad3ce9621f2c382cc722a3c2f1677637d3e3f2bfbf37d6ca652f98880eb4e`,
   first token `271`. Control and variant streams must stay identical (bitwise-exactness).
4. `test/unit/test_m5_typed_boundary.py` green on the same tree; new unit tests green.
5. pg3 legacy hash `27857cb8ca03` for `q4k_g3_lanemap_gemv_4096_4096` unmoved.

The record gains `NV sm_120` only after 1-5 pass. After promotion, re-run the closed-vs-open
token-stream equality (pins 3/3 both modes) to prove the record-open state equals the
forced-open state.

## 4. Expected recovery and interaction contract

- Expected census: kernels/token 984 -> ~912 (72 copies removed, -36 absorbed residual
  adds net the probe-2 variant row at 984). Kernel us/token 5975.3 -> ~5867 (copy mass
  108us removed; fused penalty ~+8us). Wall at d512: ~+0.5-1.0% over the 180.36 control
  (projection, measured in section 6).
- Interaction: the fold lives entirely in the attn_qo o-proj slot; it does not touch
  w1w3 (gate/up), Q6K, kv-store, or flash-combine routes. Do NOT add this recovery to any
  composed row that already books the old rejected composition; book it as a fresh row
  with the measured section-6 delta.
- The ledger's "claims that remain unsupported" rules apply: no synthetic microgate
  extrapolation; the booked number is the same-session section-6 delta, not the probe-2
  ceiling.

## 5. Beyond parity - ranked continuation (NOT authorized here)

The residual_add fold is worth ~100us/token of kernel time, roughly 2% of the current
5.324ms native token; it does not close the 1.319ms llama gap by itself. The reconciled
ledger's ranked next steps remain the parity path, in order:

| Rank | Work | Why | Next decisive gate |
| ---: | --- | --- | --- |
| 1 | Distinct exact native Q4/Q6 DP4A substrate | Live d512 llama causality is MMVQ/Q8_1+DP4A, four warps; every exact-MMA construction is closed or wall-neutral. | A third physical representation with an independent oracle, PTX/resources, and a material included-cost win before one real family. |
| 2 | Native independently scheduled RM/HCQ work | Native light-kernel overlap construction PASSes, but queue/wait economics are negative. | A wait-adjusted decode forecast with a positive margin before another token-schedule arm. |
| 3 | Residual sampler/vocab/RoPE/KV tails | Small remaining ownership; P1/P5 consume descriptor and feedback parts. | A distinct topology/body mechanism with explicit P1/P5 interaction exclusion before GPU time. |

Each row is a separate scope with its own HARD STOP; this scope does not authorize any of
them.

## 6. HARD STOP

This scope authorizes exactly: the per-variant resadd record (section 2.1), the
residual-slot typed view (section 2.2), the section-6 gate run, and the promotion decision
of `decode-q4k-epilogue-resadd-route-policy.json` to `NV sm_120` after the gate passes.
It does NOT authorize: changing the combined M4 record, the ffn_down prelude, fp16_cast,
M3/M5/Path 3, w1w3, kv-store, any emitter change, or any of the section-5 continuation
rows. No GPU probe outside `flock -w 600 /tmp/gpu-bench.lock`.

## 7. References

- `m4-variant-reopen-boundary-p0-scope-20260806.md` (probe authority, sections 4-7)
- `m4-residual-boundary-fold-probe-record-20260806.md` (probe 1 PASS)
- `m4-resadd-kernel-microgate-record-20260806.md` (probe 2 PASS)
- `m5-typed-boundary-p0-implementation-record-20260803.md` (landed ABI precedent)
- `nv-decode-parity-campaign-reconciled-ledger-20260805.md` (recovery booking rules)
- `nv-decode-parity-final-20260802.md` (M2-on wall authority)
- `tinygrad/llm/kernel_program.py`, `tinygrad/llm/decode_routes.py`,
  `tinygrad/llm/decode_kernels.py`, `tinygrad/llm/model_route_plan.py`,
  `tinygrad/llm/qk_primitives.py`, `tinygrad/llm/model.py`
