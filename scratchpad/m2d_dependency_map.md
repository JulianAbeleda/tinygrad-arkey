# M2d dependency map: absorbing `E_32_32_4_0a5eb0ac` (attention fp32->fp16 cast, x36) via the M5 fp16 flash-combine store under a harness-installed lease

Date: 2026-08-11. Repo: `/home/ubuntu/tinygrad-arkey`, branch
`nvidia-bringup-20260731` (base `0dfe0ecac`, the M2c BOOKED commit). This
leg is a read-only census walk; `scratchpad/m2d_dependency_map.md` is the only
file it creates. During the walk the M2d implementation leg landed
concurrently: `model.py`, `decode_routes.py`, and the
`nv_epilogue_absorption_ab.py` harness now carry the
`_flash_combine_fp16_lease` wiring (verified below, not authored here), so all
inline citations reflect the post-landing line numbers. Source of truth:
`scratchpad/m2c_dependency_map.md` (M2c contract), the M2d arithmetic
validation `scratchpad/m2d_arithmetic_validation.md` (booked M2c AB baseline),
the M5 measurement record
`docs/task_workflow/input/m5-flash-combine-normalization-measurement-record-20260802.md`
(why the variant was non-landing: the opaque boundary replaced the cast 1:1
with `E_32_32_4_3b0fcfbc` fp16->fp16 copies x36), the M5 typed-boundary P0
scope `docs/task_workflow/input/m5-variant-reopen-boundary-p0-scope-20260803.md`
(commit `d46cee681`, the typed ABI that prevents exactly that), and the fresh
code walk of the producer/consumer/validator/gate sites cited inline below.

The M2d thesis, restated: the closed-default M5 flash-combine fp16 store
(`flash_fused_gmax_combine_f16_*`) already exists, emits the same RNE
fp32->fp16 `(half)` cast the standalone `0a5eb0ac` kernel performs
(`flash_decode_attention.py:245`), and already declares its typed fp16 output
layout; what was missing is a *measured reason to open it*. The M2d lease
(`_flash_combine_fp16_lease`, harness-installed on the model and every block)
opens the M5 variant by override at the model call site, and the M5 typed
boundary folds the consumer's prelude copy, so the cast disappears with no
`3b0fcfbc` replacement. No policy record changes; everything else stays
byte-identical.

## 1. Producer side

The fp16 combine is emitted by `flash_decode_live_split_block_tile`
(`tinygrad/llm/flash_decode_attention.py:613-653`). `combine_fp16` is a
parameter (default False, line 616); when True, `describe_flash_decode_attention`
receives it (lines 631-633) and builds `FlashCombineSpec(output_fp16=True)`,
whose `emit()` renders `flash_fused_gmax_combine_kernel(..., output_fp16=True)`
(lines 451-455). The emitter stores the RNE cast in-kernel and switches the
kernel name (`flash_decode_attention.py:245-248`): `value.cast(dtypes.float16)`
and `flash_fused_gmax_combine_f16_{Hq}_{Hd}` vs
`flash_fused_gmax_combine_{Hq}_{Hd}` (for the Qwen3 census shape:
`flash_fused_gmax_combine_f16_32_128` vs `flash_fused_gmax_combine_32_128`).

| producer kernel (x36 blocks) | concrete output | typed declaration | declaration site |
| --- | --- | --- | --- |
| `flash_fused_gmax_combine_f16_32_128` (M2d, lease open) | fp16 `(4096,)` AFTER | `DeclaredTypedOutput(TypedLayout(fp16, (Hq*Hd,), (Hq, Hd)), combine_fusion_admitted=True)` | `flash_decode_attention.py:644-647`, wired into the combine `OutputSpec` at 648-651 (dtype flips fp16 at 650) |
| `flash_fused_gmax_combine_32_128` (legacy, default) | fp32 `(4096,)` AFTER | **no declaration** (`combine_typed = None`, 644-645) | same site |

Exact gate chain that must open (or, under M2d, be *overridden*, never
re-recorded):

1. `binding.combine_fusion` is set at bind time from
   `admission.combine_fusion_admitted` (`decode_routes.py:491-493`).
2. `FlashDecodeAdmission.combine_fusion_admitted = admitted AND combine_fusion_promoted`
   (`flash_decode_attention.py:572-573`); `admitted` is shape+capability+target
   (566-567), `combine_fusion_promoted` is an `evaluate` parameter
   (601-604) fed by `decode_flash_combine_fusion_promoted(target)`
   (`decode_routes.py:484-486`).
3. The policy authority is `decode_flash_combine_fusion_promoted`
   (`model_route_plan.py:299-304`) over the record
   `tinygrad/llm/generated/decode-flash-combine-route-policy.json`, which is
   **closed**: `"promoted_targets": []` (verified file contents; loader
   semantics at `model_route_plan.py:281-297`). M2d does NOT touch this record.

M2d lease override point (landed during this walk): the flash call site in
`TransformerBlock._attention` (`model.py:887-890`) passes
`combine_fp16=bool(getattr(self, "_flash_combine_fp16_lease", False))`
(`model.py:890`) to `flash_decode_attention_route` (new parameter
`combine_fp16:bool|None=None`, `decode_routes.py:500-503`), which threads it
into `flash_decode_live_split_block_tile` as
`combine_fp16=bool(binding.combine_fusion or combine_fp16)`
(`decode_routes.py:534`). The lease is read off the *block* (`self` in
`_attention` is the `TransformerBlock`); the harness installs it on the model
and every block, mirroring the LEASE/LEASE2 pattern in
`extra/llm_research/decode/nv_epilogue_absorption_ab.py:65-66,116-140`. The
declaration records the gate state at production time
(`flash_decode_attention.py:641-643` comment: recording `combine_fp16` IS the
producer-side gate state the consumer validator requires).

## 2. Consumer side

The consumer is the attn_qo o-proj Q4K GEMV. `decode_routes.py:143-148` issues
`TypedViewRequest(slot=1, dtype=fp16, flat_shape=(binding.K,), route_role="attn_qo")`
with `requires_combine_fusion` defaulting True (`kernel_program.py:117`), only
for `route_role == "attn_qo"` (the `ffn_down` spelling at 146-147 explicitly
clears it and requires epilogue absorption instead). The fold is executed by
`_fold_typed_input_views` (`kernel_program.py:292-310`) inside
`_execute_outputs` (`kernel_program.py:421`), before `uop_program`.

Validator path for the combine case (`_validated_typed_view`,
`kernel_program.py:241-289`): request is a `contiguous()` (246); the lossless
roundtrip cancel runs first (253); view base must be an opaque AFTER (255);
dtype/shape must match the request (257-258) and be preserved through the
chain (259-260); `_is_pure_contiguous_view` must prove offset-0 row-major
(261, proof at 210-219 via `graph_rewrite` with `pm_mops + symbolic`); the
producer declaration must exist and exactly match (263-267); then for the
combine case `declared.combine_fusion_admitted` (282), `request.requires_combine_fusion`
(284), `route_role == "attn_qo"` (286), and the program must be a q4k `.gemv`
(287-288). Any rejection keeps the generic flat-buffer ABI (byte-identical).

Roundtrip cancel exactness (`_cancel_lossless_fp16_roundtrip`,
`kernel_program.py:222-238`): the real chain is fp16 AFTER ->
`reshape(B,Hq,T,Hd)` -> `cast(fp32)` (the model's lossless upcast,
`model.py:887` `out.reshape(B,Hq,T,Hd).cast(q.dtype)`, decode q is fp32) ->
`transpose(1,2)` + `reshape(B,T,-1)` (`model.py:949`) -> the prelude
`x[:, 0, :].reshape(K).cast(fp16).contiguous()` (`decode_routes.py:116`).
fp16->fp32->fp16 is exact for every fp16 value, so the cancel composes the
movement legs (236-237 requires a pure movement chain between the casts, 229-234
requires the inner leg to be an fp16 view of an fp16 AFTER) and returns the
pure fp16 view of the AFTER. The composed chain includes a `transpose(1,2)`
permute of size-1 dims (T=1); `_is_pure_contiguous_view` proves it identity
via the `pm_mops` rewrite, so the real chain **is accepted after the cancel**.
This is exactly the chain pinned by
`test_real_model_fp32_pipeline_chain_folds_to_view_with_zero_materialization`
and `test_decode_routes_o_proj_folds_real_model_fp32_pipeline_chain`
(`test/unit/test_m5_typed_boundary.py:132-137,358`).

Lossy/non-identity variants reject (all pinned): bf16 roundtrip (181), fp32
arithmetic between the casts (196), data-moving permute (212), missing
declaration (230), combine gate closed (234), wrong route_role (238), dtype
mismatch (242), flat-shape mismatch (246), typed-ABI gate closed (250), wrong
program identity (254).

## 3. Callify boundary

The fold is **purely execution-time** (`_fold_typed_input_views` at
`kernel_program.py:421` during `execute_promoted_program`, which is graph
construction inside the Tensor Function dispatch). No transform-time redirect
participates in the combine fold.

The callify precompiled-output redirect (`CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT`,
`tinygrad/callify.py:11`) is scoped to precompiled FUNCTIONs on the
reduce-output route: participation requires a REDUCE_OUTPUT marker body or
route membership (`callify.py:130-140,233-239`), and the M2c declared-AFTER
rebind branch (`callify.py:334-346,582-585`) applies only to a precompiled
function whose RESULT chain is a declared epilogue-absorbing AFTER (the block
output, `fab82d40` fold). The flash combine is an intermediate opaque program
output inside the block `_run` body, not a function RESULT, so the redirect
never sees it.

Interaction check: `transform_precompiled_call` passes inputs that are
`AFTER`/`BIND` through unchanged (`callify.py:592`), so the folded view of the
combine AFTER survives callification of the enclosing `_run` without a
transport `contiguous()` being reinserted. The declaration double-record
against the CALL SINK (`kernel_program.py:429-438`) exists for the M2c
block-output rebind proof after `@function` input substitution; the combine
declaration lookup happens at fold time, before substitution, so M2d does not
depend on it, but the SINK key also keeps the combine declaration findable if
an enclosing substitution rebuilds the AFTER (pattern pinned by
`test_m2c_declared_after_sink_key_survives_after_substitution`,
`test/unit/test_nv_epilogue_absorption_ffn_resadd.py:274`).

## 4. Census families that must stay byte-identical vs may swap

M2d runs on top of the booked M2c AB candidate (630 kernels/token per
`scratchpad/m2d_arithmetic_validation.md`, `fab82d40` 49->0, `0a5eb0ac` still
36; control 715). Projected M2d deltas vs the M2c candidate baseline:

Must move (M2d, exactly):

- `E_32_32_4_0a5eb0ac` 36 -> 0 (attention fp32->fp16 cast, absorbed by the
  fp16 combine store; digest
  `0a5eb0ac56c097a089f39541962d5d73b9bc613251a6320685824338d26b38c4` per the
  M5 measurement record)
- `flash_fused_gmax_combine_32_128` 36 -> 0 (legacy fp32 combine replaced by
  its f16 twin)
- `flash_fused_gmax_combine_f16_32_128` 0 -> 36 (new fp16 combine; the store's
  `(half)` cast is the same RNE expression, bitwise-identical bytes)
- net program delta -36 (630 -> 594); combined M2c+M2d vs the M2c AB control:
  715 -> 594 (-121)

Must stay byte-identical between arms (the M2c contract, unchanged):

- ffn_down resadd twins x36: `q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288`
  x18 + `q6k_gen_coop_4096_12288_inkernel_epi_ffnresadd` x18, and their plain
  GEMV twins folded 1:1 (M2b swap rule; `fab82d40` stays 0)
- `q4k_g3_lanemap_gemv_w1w3fused16_12288_4096` x36 (M2a booked fp16 store)
- FFN norm chain: `r_16_256` x37 + `E_32_32_4_f14a5cc0` x37
- flash score/pv x36 (`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128`)
  and its preludes (`E_16_32_4_2`, `r_8_8_16_2_4`, `E_8_8_16_2`,
  `E_16_32_4_2_164c0def`)
- q/k partials and providers: `q4k_warp_coop_q8_dp4a_partial_*`,
  `q4k_g3_lanemap_gemv_4096_4096`/`_1024_4096`, `q6k_gen_partial_*`,
  `q6k_q8_dp4a_1024_4096`
- norms: `rmsnorm_q8_1_llama_provider_4096` x17,
  `reduce_output_rmsnorm_1_4096` x19, `reduce_output_rmsnorm_32_128`/`_8_128`
- kv-store / rope / scatter: `r_8_4_(start_pos+1)_*`, `r_8_8_16_2_*`,
  `E_8_8_16_*`, `E_(start_pos+1)_8_4_*`
- lm_head path: final `reduce_output_rmsnorm_1_4096`,
  `E_32_32_4_5a5673a4` x1, `q6k_gen_coop_151936_4096_inkernel` x1,
  `E_1187_32_4` x1
- the `E_32_32_4_3b0fcfbc` fp16->fp16 copy class stays 0 (the M5 typed
  boundary exists to prevent it; any appearance fails closed)

No other consumer of the combine output: verified by walking the flash branch
of `_attention` (`model.py:882-961`). The combine AFTER `out` is read only by
`model.py:887` (the lossless fp32 upcast the roundtrip cancel walks), then
`model.py:949` (transpose/reshape, movement only), then `model.py:950`
(`attn if not attn_output_gate else attn * gate.sigmoid()`; Qwen3-8B sets
`attn_output_gate=False` -- `model.py:1655` gates it on arch `qwen35`/`qwen35moe`),
then the o-proj linear (`model.py:956-961`), whose GEMV prelude is the
`decode_routes.py:116` chain. Nothing else reads the combine AFTER, so the
fp16 swap cannot change any other consumer's bytes.

## 5. Gates (all fail-closed, default closed)

| gate / lease | default | where declared / checked |
| --- | --- | --- |
| `CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT` | ContextVar 0 | `tinygrad/callify.py:11`; reduce-output redirect machinery (`callify.py:236,320-351,582-585`) |
| `CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER` | ContextVar 0 | `tinygrad/callify.py:16`; typed semantic input producer (`callify.py:521-562`) |
| `_q4k_w1w3_fp16_store_lease` (M2a) | absent/False unless harness-installed | model + block + gate/up linears; `nv_epilogue_absorption_ab.py:65,107-108` |
| `_ffn_down_resadd_lease` (M2b) | absent/False unless harness-installed | model + every block + every ffn_down linear (`nv_epilogue_absorption_ab.py:66,109-114`); re-checked on the linear (`decode_routes.py:104-107`) |
| `_flash_combine_fp16_lease` (M2d) | absent/False unless harness-installed | NEW (`LEASE3`, `nv_epilogue_absorption_ab.py:78`): model + every block, candidate arm only (`nv_epilogue_absorption_ab.py:135-138`); read at the flash call site `model.py:890`; threaded as override OR at `decode_routes.py:534`; control arm must assert it absent |
| `decode_flash_combine_fusion` promotion record | closed (`promoted_targets: []`) | `model_route_plan.py:281-304`; `generated/decode-flash-combine-route-policy.json` (UNCHANGED by M2d) |
| `FLASH_DECODE_ADMISSION_DEBUG` | getenv 0 | `decode_routes.py:487-489` |
| `M5_TYPED_BOUNDARY_DEBUG` | getenv 0 | `kernel_program.py:300-308` typed-view fold debug print |
| `M4_RESADD_BOUNDARY_DEBUG` | getenv 0 | `kernel_program.py:404-405` residual-fold debug print |

Validator fail-closed paths: `_validated_typed_view` (`kernel_program.py:241-289`)
rejects unless the producer declared an exact-matching typed layout AND the
combine-fusion + typed-ABI gates are open; `_validated_residual_view`
(`kernel_program.py:349-389`) is unaffected (M2c residual contract). Any
rejection leaves the generic flat-buffer ABI byte-identical.

Harness gate changes for M2d (all in
`extra/llm_research/decode/nv_epilogue_absorption_ab.py`; the implementation
leg landed `_configure`/`_gates`/census during this walk, smoke is still
outstanding):

- `_configure` (116-140): LEASE (M2a) and LEASE2 (M2b) are now present in
  BOTH arms (the M2d control IS the booked M2c candidate); the candidate arm
  additionally installs `LEASE3 = "_flash_combine_fp16_lease"` (line 78) on
  the model and every block (135-138). `_gates` (143-165) and
  `_assert_candidate_configured`/`_assert_control_closed` (168-199) carry the
  lease keys; control fails closed if LEASE3 is observed or LEASE2 is missing.
- Smoke gate (236-...): NOT yet updated by the implementation leg. M2d needs
  the candidate smoke row to additionally assert
  `flash_fused_gmax_combine_f16_32_128` present and
  `flash_fused_gmax_combine_32_128` absent, and `E_32_32_4_0a5eb0ac` absent
  (no `attention_cast_present`); existing keys (fused16 body, ffnresadd body,
  no residual add, no block-output copy) unchanged.
- Census contract (landed): census rows gain `flash_combine_f16_count` /
  `flash_combine_f32_count` / `combine_copy_count` (344-347, 361-370);
  `validate_census` gains the combine swap rule (476-489: f16 count must
  equal the control fp32 count, fp32 candidate 0, f16 control 0) and the
  conditions `m2d_attention_cast_folded` (cast candidate 0, control > 0),
  `m2d_combine_swap_backed`, `m2d_no_opaque_copy` (`3b0fcfbc` 0/0)
  (502-504); `net_delta_matches_drop` (508) becomes
  `-(resadd_control + copy_control + attn_cast_control)` (net -36 on the
  booked baseline: 630 -> 594).
- Wall bracket unchanged: `validate_timing_bracket` (481-500), +50 us/token vs
  both bracketing controls, identical token-stream hashes; HARD_STOP_NOTES
  (503-510) line 509 ("no policy promotion; lease attribute harness-installed
  only") stays true.

## 6. Tests that pin the contracts

- `test/unit/test_m5_typed_boundary.py`: pins the M5 typed ABI end to end --
  the fp16 combine declaration (`test_fp16_combine_declares_typed_output_layout`),
  the legacy fp32 combine declaring nothing (`test_legacy_fp32_combine_declares_no_typed_output`),
  the contiguous-request fold with zero materialization (`test_typed_abi_folds_contiguous_request_to_view_with_zero_materialization`),
  the real model fp32-pipeline roundtrip chain folding (`test_real_model_fp32_pipeline_chain_folds_to_view_with_zero_materialization`),
  the copy materializing without the ABI (`test_without_typed_abi_the_copy_is_materialized`,
  `test_real_model_chain_without_abi_still_materializes_the_copy` -- the
  `3b0fcfbc` class), GEMV-AFTER buffer binding (`test_typed_abi_binds_the_gemv_to_the_combine_after_buffer`),
  the lossless-cancel exactness with lossy rejections (bf16/arithmetic/permute,
  lines 181-225), and every fail-closed validator path (230-286); the
  decode_routes opt-in and the attn_kv generic-ABI contrast (348-373).
- `test/unit/test_nv_epilogue_absorption_fp16_store.py`: pins the M2a fp16
  store, the cast fold, and the epilogue-absorption validator
  (`test_epilogue_absorption_folds_only_against_a_declared_producer`,
  `..._is_fail_closed_without_a_declaration`), plus the lease-driven gate-up
  name/dtype flip (212-232) -- the same fail-closed lease shape M2d reuses.
- `test/unit/test_nv_epilogue_absorption_ab.py`: pins the A/B census gate,
  swap-twin rule, lease install/close (`test_candidate_arm_installs_m2a_and_m2b_leases`,
  `test_control_arm_keeps_m2a_lease_and_closes_m2b`), and fail-closed family
  checks including `test_census_gate_fails_closed_when_m5_attention_cast_shifts`.
- `test/unit/test_nv_epilogue_absorption_ffn_resadd.py`: pins the ffnresadd
  kernels and the M2c declared-AFTER output-slot rebind
  (`test_m2c_declared_after_rebind_proves_the_output_slot_contract`,
  `test_m2c_declared_after_sink_key_survives_after_substitution`,
  `test_m2c_declared_after_rebind_is_fail_closed`).

M2d needs (new):

1. Lease wiring: `_configure` installs `_flash_combine_fp16_lease` on the model
   and every block for the candidate arm only, with control-absent fail-closed
   assertions (mirror `test_candidate_arm_installs_m2a_and_m2b_leases` /
   `test_control_arm_keeps_m2a_lease_and_closes_m2b`).
2. Lease fail-closed control: with the lease absent, the `model.py:887-890` call
   site threads `combine_fp16=False`, the legacy `flash_fused_gmax_combine_32_128`
   fp32 combine + `E_32_32_4_0a5eb0ac` x36 census renders, and no typed
   declaration exists.
3. Override-OR plumbing: `flash_decode_attention_route(combine_fp16=...)` ORs
   with `binding.combine_fusion` at `decode_routes.py:534` (both True paths
   and the False path), no policy-record change.
4. Census swap contract: `test_census_gate_fails_closed_when_m5_attention_cast_shifts`
   rewritten to the new contract (cast 36->0, legacy combine 36->0, f16
   combine 0->36, net -36); new fail-closed cases for f16 combine absent,
   legacy combine remaining, cast remaining, `3b0fcfbc` appearing, and any
   unrelated shift.
5. End-to-end validator acceptance: full block flash call under the lease
   produces no `0a5eb0ac` and no fp16 copy; the o-proj GEMV binds the fp16
   combine AFTER buffer (extend `test_typed_abi_binds_the_gemv_to_the_combine_after_buffer`
   to the leased model path).
6. Smoke gate: candidate smoke row asserts `flash_fused_gmax_combine_f16_32_128`
   body present and `flash_fused_gmax_combine_32_128`/`0a5eb0ac` absent.

## Summary

The M2d change opens the closed-default M5 flash-combine fp16 store
(`flash_fused_gmax_combine_f16_32_128`, in-kernel RNE `(half)` cast) via a
harness-installed `_flash_combine_fp16_lease` read at the model flash call
site (`model.py:890`) and OR'd into `binding.combine_fusion` at
`decode_routes.py:534`, with no policy-record change; the producer declares
its fp16 typed output (`flash_decode_attention.py:644-651`) and the M5 typed
boundary (`_validated_typed_view` + `_cancel_lossless_fp16_roundtrip`) folds
the attn_qo prelude `x[:,0,:].reshape(K).cast(fp16).contiguous()` over the
fp32 upcast to a zero-copy view of the combine AFTER, so the 
`E_32_32_4_0a5eb0ac` cast x36 and the legacy fp32 combine x36 disappear with
no `3b0fcfbc` copy class (net -36 kernels on the booked M2c candidate,
630 -> 594).
The fold is execution-time only; the callify precompiled-output redirect
stays scoped to the reduce-output/declared-AFTER block-output route and never
touches the combine. Census: only the three combine/cast families move;
everything else in the M2c contract (ffnresadd twins x36, w1w3fused16 x36,
norm chain x37, flash score/pv x36, partials, norms, kv/rope, lm_head) stays
byte-identical, and the only other combine consumer (the model's lossless
upcast into the o-proj GEMV) is exactly the chain the roundtrip cancel walks.
Gates: the new lease is fail-closed absent, the smoke gate adds the f16-body
present / legacy-absent / cast-absent checks, the census contract changes the
attention cast from "identical" to "must drop 36->0", and the wall bracket
(+50 us/token, identical token hashes) is unchanged.
