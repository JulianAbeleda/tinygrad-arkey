# M2c dependency map: killing `E_32_32_4_fab82d40` (fp32 block-output copies) and `E_32_32_4_0a5eb0ac` (attention fp32->fp16 cast)

Date: 2026-08-11. Repo: `/home/ubuntu/tinygrad-arkey`, branch
`nvidia-bringup-20260731` (clean worktree at `310cd4631`). Read-only census
walk; no repo code or tests were modified. Source of truth for the family
anatomy: ledger `docs/task_workflow/input/nv-epilogue-absorption-route-scope-20260810.md`
(M2c section), `scratchpad/m2c_arithmetic_validation.md` (final AB baseline),
`docs/task_workflow/input/m5-variant-reopen-boundary-p0-scope-20260803.md`
(legacy-chain description of `0a5eb0ac`), and the fresh structural probe run
(`/tmp/m2c_probe_raw.txt`, `/tmp/m2c_order.txt`, `/tmp/m2c_edges.txt`).

The ledger's M2c thesis, restated: the residual slot must stay **fp32** for the
attention resadd epilogue and the next block's attention norm, so M2c folds the
contiguous boundary (view-of-AFTER binding) instead of changing dtype. M2b
(committed) already closed the `E_32_32_4_86a23e1a` transport-copy sub-item by
declaring the attn_qo typed output so the ffn_down residual fold accepts; the
remaining rows are `fab82d40` x49 and `0a5eb0ac` x36.

## 1. Producer side

The fp32 block output is the fp32 `AFTER` of the **ffn_down residual-add GEMVs**;
the fp32 attention-intermediate `h` is the fp32 `AFTER` of the **attn_qo
residual-add GEMV**. Both declare typed fp32 outputs
(`epilogue_absorption_admitted=True`). The w1w3fused16 GEMV declares the M2a
fp16 typed output. Every declaration is recorded per-AFTER in
`kernel_program.py` `_DECLARED_TYPED_OUTPUTS` (weak table) at
`_execute_outputs` when the program runs.

| producer kernel (x36 blocks) | concrete output | typed declaration | declaration site |
| --- | --- | --- | --- |
| `q4k_g3_lanemap_gemv_epi_resadd_4096_4096` (attn_qo o-proj, `Q4KGEMVEpilogue("residual_add")`) | fp32 `h = x + attn` (block intermediate) | `DeclaredTypedOutput(TypedLayout(fp32, (4096,), (1,1,4096)), combine_fusion_admitted=False, epilogue_absorption_admitted=True)` | `tinygrad/llm/decode_routes.py:168-171` |
| `q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288` x18 / `q6k_gen_coop_4096_12288_inkernel_epi_ffnresadd` x18 (ffn_down, `ffn_down_resadd` epilogue) | fp32 **block output** `h + ffn_out` | same fp32 `(1,1,4096)` declaration, `epilogue_absorption_admitted=True` | Q4K `decode_routes.py:168-171`, Q6K `decode_routes.py:368-371` |
| `q4k_g3_lanemap_gemv_w1w3fused16_12288_4096` (gate*up fused) | fp16 `(12288,)` | `combine_fusion_admitted=False, epilogue_absorption_admitted=True` (M2a fp16 store) | `decode_routes.py:221-224` (store_fp16 lease path) |
| `flash_fused_gmax_combine_32_128` (flash op point) | fp32 `(4096,)` AFTER | **no declaration at HEAD** (the M5 combine-fusion variant is closed-default) | `tinygrad/llm/flash_decode_attention.py:650-652` |

Producer role-marker plumbing: `model.py` threads `residual_for_output=x` into
`_attention` only under the M4 resadd admission (`model.py:644`), and threads
`normed_h=h` into the ffn_down call under the M2b lease (`model.py:690-695`);
`decode_routes.py` turns those into the epilogue slots (`decode_routes.py:103-110`
Q4K, `decode_routes.py:350-356` Q6K). The residual slot is consumed as **fp32**:
`epi_inputs["residual"][:, 0, :].reshape(binding.N).cast(dtypes.float32)`
(`decode_routes.py:118`); the ffn_down residual slot is the same spelling on
`normed_h` (`decode_routes.py:110`).

## 2. Consumer side

Per-block execution order at the flash operating point (from
`/tmp/m2c_order.txt`, one decode step):

`flash score/pv -> flash_fused_gmax_combine (fp32 AFTER) -> E_32_32_4_0a5eb0ac -> q4k_g3_lanemap_gemv_epi_resadd_4096_4096 -> r_16_256 -> E_32_32_4_f14a5cc0 -> q4k_g3_lanemap_gemv_w1w3fused16_12288_4096 -> ffn_down resadd GEMV (*_epi_ffnresadd) -> E_32_32_4_fab82d40 (x1 or x2) -> next-block attention norm (rmsnorm_q8 / reduce_output_rmsnorm) -> q/k/v partials`

### (a) Next-block attention residual input (attn_qo resadd branch)

View expression: `epi_inputs["residual"][:, 0, :].reshape(binding.N).cast(dtypes.float32)`
(`decode_routes.py:118`, slot 2 of the next block's
`q4k_g3_lanemap_gemv_epi_resadd_4096_4096`). This is an **fp32->fp32 no-op
cast**, kept fp32 on purpose (M2b premise correction: the resadd epilogue reads
the residual as fp32; storing fp16 would round it and break the exact-logits
gate). The cast folds; the `contiguous()` over the opaque CALL/AFTER boundary
does not, so the flat-buffer ABI leaves a **materializing fp32 copy** here at
HEAD (the probe edges file shows slot 2 bound as a flat `PARAM`). This is one
`fab82d40` site. M2c folds it via `_validated_residual_view` (the producer
declaration already exists from M2b; the same mechanism that killed
`86a23e1a`).

### (b) Next-block attention norm (the `r_16_256`/`f14a5cc0` norm chain; note on the task's label)

The census pair `r_16_256` x37 + `E_32_32_4_f14a5cc0` x37 is the **FFN norm
chain of the same block**: `r_16_256` norms the attn_qo resadd `AFTER` `h`
(`model.py:649`), and `f14a5cc0` is the norm-output materialization feeding
`w1w3fused16` (M1 row, not an M2c target; it consumes the attn_qo `AFTER`, not
the block output). The fp32 **block output** feeds the *next* block's
**attention** norm, which at HEAD renders as `rmsnorm_q8_1_llama_provider_4096`
(shared-q8 provider, early blocks) or `reduce_output_rmsnorm_1_4096` (fused
reduce-output, later blocks); both read a materialized contiguous fp32 block
output, i.e. the second `fab82d40` site. The fused norm's fallback is
`out.cast(dtypes.float16)` (`model.py:424`), so the fp32 block output is
also the upstream of the norm's own fp16 cast. M2c's norm-side fold is the same
boundary fold, still fp32 at the copy (the norm's fp16 cast is downstream and
unchanged).

### (c) lm_head path (last block only)

Tail from `/tmp/m2c_order.txt`:
`fab82d40 (671) -> reduce_output_rmsnorm_1_4096 (672) -> E_32_32_4_5a5673a4 (673, fp32->fp16 final-norm cast) -> q6k_gen_coop_151936_4096_inkernel (674, vocab GEMV) -> E_1187_32_4 (675)`.
The last block's block-output copy feeds the final norm, whose fp16 output
feeds the vocab GEMV. `5a5673a4` x1 is the final-norm epilogue, a separate
one-off from the two M2c target families.

### The `0a5eb0ac` cast

Producer: `flash_fused_gmax_combine_32_128` fp32 `AFTER`. Consumer: the attn_qo
resadd GEMV's activation prelude
`x[:, 0, :].reshape(binding.K).cast(dtypes.float16).contiguous()`
(`decode_routes.py:116`). At HEAD the M5 combine-fusion variant is closed, so
the combine stores fp32 and the prelude's `cast(fp16)` is real lossy RNE work:
this is `E_32_32_4_0a5eb0ac` x36, one per block, in the flash op point only (the
plain-SDPA run has 0 of them). It is a cast (lossy), not a copy, so M2c cannot
drop it as an identity; it folds only by making the fp16 bytes in-kernel
bitwise-identical to the standalone cast (the M5 combine-fusion or a
consumer-epilogue spell) while the fp32 residual slot stays untouched.

## 3. Probe results (graph position of the two families)

Probe run: `PYTHONPATH=/home/ubuntu/tinygrad-arkey python3 scratchpad/nv_residual_family_structural_probe.py`
with the Qwen3 model at `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`. Exit 0, no
fallback needed (`/tmp/m2b-ab.children/*.json` are stale pre-fix artifacts and
were not used).

- Non-flash probe (64-token prompt, `executed programs: 751, residual family:
  87`): `fab82d40` x49, `f14a5cc0` x37, `5a5673a4` x1, `0a5eb0ac` x0 (plain
  SDPA path), `86a23e1a` x0 (M2b transport fix confirmed at HEAD).
- Flash operating point (`FLASH_DECODE=1`, 640-token prompt, 679 executed):
  `fab82d40` x49, `0a5eb0ac` x36, `f14a5cc0` x37, flash families x36 each
  (`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128`,
  `flash_fused_gmax_combine_32_128`), `86a23e1a` x0.

Graph position (authoritative `/tmp/m2c_order.txt`, per-block window):

- `0a5eb0ac`: sits between `flash_fused_gmax_combine_32_128` (fp32 AFTER) and
  `q4k_g3_lanemap_gemv_epi_resadd_4096_4096`; the combine AFTER feeds it, the
  attn_qo resadd GEMV reads its fp16 output.
- `fab82d40`: sits immediately after the ffn_down resadd GEMV
  (`q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288` x18 /
  `q6k_gen_coop_4096_12288_inkernel_epi_ffnresadd` x18) and before the
  next-block attention norm. **13 blocks render 2 copies, 23 blocks render 1**
  (49 = 13 x 2 + 23 x 1): the two sites are (1) the block's own
  `.contiguous()` transport at the `@function` boundary (`model.py:698`), and
  (2) the next-block norm/residual input materialization. The 2-copy vs
  1-copy split does not track the attention-norm provider flavor (the
  `rmsnorm_q8` / `reduce_output_rmsnorm` boundary sits at a different block
  index); the observable is that one of the two copies stops rendering from
  block 13 onward, consistent with the M2b typed-output declaration letting
  one of the two boundary folds bind there.

Producer/consumer edges at the GEMV level (`/tmp/m2c_edges.txt`):
`flash_fused_gmax_combine_32_128` reads `flash_block_tiled_xlane_score_pv`;
`q4k_g3_lanemap_gemv_epi_resadd_4096_4096` reads the combine output plus a flat
`PARAM` residual (the materialized block output at HEAD);
`*_epi_ffnresadd` reads `w1w3fused16` (slot 1) and the attn_qo resadd AFTER
(slot 2, the M2b fold). `reduce_output_rmsnorm_1_4096` reads the `fab82d40`
copy.

## 4. Gates (all fail-closed, default closed)

| gate / lease | default | where declared / checked |
| --- | --- | --- |
| `CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT` | ContextVar 0 | `tinygrad/callify.py:11`; precompiled-output redirect machinery (`callify.py:236,362,397,509,570`) |
| `CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER` | ContextVar 0 | `tinygrad/callify.py:16`; typed semantic input producer (`callify.py:483,789-791`) |
| `_q4k_w1w3_fp16_store_lease` (M2a) | absent/False unless harness-installed | model + block + gate/up linears; `decode_routes.py` `store_fp16` path |
| `_ffn_down_resadd_lease` (M2b) | absent/False unless harness-installed | model + every block + every ffn_down linear (`nv_epilogue_absorption_ab.py`); re-checked on the linear in `decode_routes.py:103-110,350-356` |
| `M4_RESADD_BOUNDARY_DEBUG` | getenv 0 (off) | `kernel_program.py:404-405` residual-fold debug print |
| `M5_TYPED_BOUNDARY_DEBUG` | getenv 0 (off) | `kernel_program.py:300-308` typed-view debug print |

Validator fail-closed paths: `_validated_typed_view` (`kernel_program.py:241-295`,
rejects unless the producer declared an exact-matching typed layout and the M5
gates are open) and `_validated_residual_view` (`kernel_program.py:349-395`,
rejects unless the producer has buffer/precompiled-output identity, or is an
AFTER with a declared `epilogue_absorption_admitted=True` typed output for the
M2b ffn_down consumer). Any rejection leaves the generic flat-buffer ABI
byte-identical.

## 5. Census families that must stay byte-identical vs may swap

Must stay byte-identical between arms when M2c lands (M2c touches none of
these; the census gate fails closed on any shift):

- flash attention families x36 each (`flash_block_tiled_xlane_score_pv...`,
  `flash_fused_gmax_combine_32_128`) and their preludes (`E_16_32_4_2`,
  `r_8_8_16_2_4`, `E_8_8_16_2`, `E_16_32_4_2_164c0def`)
- `q4k_g3_lanemap_gemv_w1w3fused16_12288_4096` x36 (M2a booked fp16 store)
- q4k/q6k partial and provider kernels: `q4k_warp_coop_q8_dp4a_partial_*`,
  `q4k_g3_lanemap_gemv_4096_4096`/`_1024_4096`, `q6k_gen_partial_*`,
  `q6k_q8_dp4a_1024_4096`
- norm chain: `r_16_256` x37 + `E_32_32_4_f14a5cc0` x37 (M1 row),
  `rmsnorm_q8_1_llama_provider_4096` x17, `reduce_output_rmsnorm_1_4096` x19,
  `reduce_output_rmsnorm_32_128`/`_8_128` (flash reductions)
- kv-store / rope / scatter families: `r_8_4_(start_pos+1)_*`, `r_8_8_16_2_*`,
  `E_8_8_16_*`, `E_(start_pos+1)_8_4_*`
- lm_head path: `reduce_output_rmsnorm_1_4096` (final), `E_32_32_4_5a5673a4` x1,
  `q6k_gen_coop_151936_4096_inkernel` x1, `E_1187_32_4` x1

Allowed to swap (M2b swap-twin rule, already booked): the plain ffn_down GEMVs
`q4k_g3_lanemap_gemv_4096_12288` x18 and `q6k_gen_coop_4096_12288_inkernel` x18
swap 1:1 with their `*_epi_ffnresadd` twins; the gate fails closed if an
ffnresadd body lacks its plain twin.

M2c itself is a pure kernel removal row (no GEMV body changes): `fab82d40`
49 -> 0 and `0a5eb0ac` 36 -> 0, net -85 kernels, -146.0 us census
(`scratchpad/m2c_arithmetic_validation.md` section 3), projected
~5.106 ms/token. The two target families are the only deltas allowed.

## 6. Tests that pin the contracts

- `test/unit/test_nv_epilogue_absorption_ffn_resadd.py`: pins the
  `*_epi_ffnresadd` kernel names/rendering, the in-kernel fp32 add, the
  lease-only admission, the fp32 typed-output declaration, and the
  next-block residual view acceptance (`test_q4k_ffn_down_resadd_declares_typed_fp32_output`,
  `test_m2b_ffn_down_after_accepts_next_block_residual_view`). M2c needs: a
  test that the next-block attention-norm input and the attn_qo residual slot
  bind the ffn_down AFTER as zero-copy views (no `fab82d40`), keeping fp32;
  and a fail-closed twin for a producer without the declaration.
- `test/unit/test_nv_epilogue_absorption_fp16_store.py`: pins the w1w3fused16
  fp16 store, the cast fold, and the epilogue-absorption validator
  (`test_epilogue_absorption_folds_only_against_a_declared_producer`,
  `..._is_fail_closed_without_a_declaration`). M2c needs: an analogous fold
  test for the fp32 block-output boundary (declared producer + contiguous
  request folds to a view, zero materialization) and a lossy-cast guard that
  `0a5eb0ac`-style fp16 bytes are only produced in-kernel bitwise-identically.
- `test/unit/test_m4_resadd_landing.py`: pins `ResidualViewRequest` +
  `_validated_residual_view` on the real block-output chain
  (`test_validator_fires_on_real_block_output_chain`,
  `test_fold_removes_the_boundary_copy_with_zero_materialization`). M2c needs:
  extend the real-chain case to the current ffn_down-resadd producer spelling
  (declared AFTER) and assert the census copy family disappears from the folded
  schedule.
- `test/unit/test_m5_typed_boundary.py`: pins the M5 typed ABI, the combine
  fp16 declaration, the lossless round-trip cancel, and the attn_qo o-proj fold
  (`test_typed_abi_folds_contiguous_request_to_view_with_zero_materialization`,
  `test_decode_routes_o_proj_opts_in_and_folds`). M2c needs: the fp32 boundary
  analog (or the explicit decision that the residual/norm fold goes through the
  residual validator, not the typed-view validator) and a guard that the legacy
  fp32 combine chain stays byte-identical when the M5 variant is closed.
- `test/unit/test_llm_kernel_program.py`: pins `KernelProgram`/
  `OutputSpec`/typed-output validation and the promoted-program boundary. M2c
  needs: no new shape/dtype contract per se; at most a validation case for the
  fp32 `(1,1,4096)` typed layout already covered by the ffn_resadd test.
- `test/unit/test_nv_epilogue_absorption_ab.py`: pins the A/B census gate,
  swap-twin rule, and fail-closed family checks
  (`test_census_gate_fails_closed_when_m2a_families_shift`). M2c needs: census
  expectations updated to 0 x `fab82d40` and 0 x `0a5eb0ac` (kernel count
  679 -> 594), with the fail-closed rule that any remaining copy/cast family
  or any shift in the byte-identical family list fails the gate.

## Summary

The fp32 block output is the ffn_down resadd GEMV `AFTER` (declared typed fp32,
`epilogue_absorption_admitted=True`); its consumers are the next block's
attn_qo residual slot (fp32, `reshape(N).cast(fp32)`), the next block's
attention norm (rmsnorm_q8/reduce_output), and the lm_head path, and the flat-
buffer ABI's `contiguous()` over that boundary is the `fab82d40` x49 copy
family. The attn_qo activation `AFTER` is the flash combine output, whose
fp32->fp16 prelude cast (`reshape(K).cast(fp16).contiguous()`) is the
`0a5eb0ac` x36 family; both are boundary materializations of the M5 typed-
boundary path, not dtype changes. M2b already made the residual fold accept by
declaring the attn_qo typed output (killing `86a23e1a`), so M2c extends the
same fail-closed view binding to the fp32 block-output boundary and keeps the
residual slot fp32. The change lives in the M5/residual typed-boundary fold
path: `tinygrad/llm/kernel_program.py` (`_validated_residual_view`/
`_validated_typed_view` + `_DECLARED_TYPED_OUTPUTS` consumers) plus the
declarations in `tinygrad/llm/decode_routes.py`, with no dtype or kernel-body
changes, gated by the existing leases and validated by the updated unit tests
and the exact-logits SHA gate.
