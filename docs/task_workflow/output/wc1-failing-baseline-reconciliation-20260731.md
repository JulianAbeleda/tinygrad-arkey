# WC1 — why the failing baseline fails (reconciliation)

Follow-on to `wc1-failing-baseline-audit-20260731.md`. Answers the question the audit
deliberately did not: which failures are the WMMA carrier migration, which predate it, which
are environmental, and which are flaky. Every claim below was produced by running the failing
files at three commits:

| commit | identity |
| --- | --- |
| `0e41c260d^` (`814cb2475`) | parent — `[codegen] separate scalar dtype from value lanes` |
| `0e41c260d` | migration — `[uop] move AMD fragment lanes into descriptors` (the bisect boundary from the task file) |
| HEAD (`e3f2173ad` + two docs commits) | current |

All runs: same 28 failing files, `DEV=NV`, this Linux box (no AMD GPU, no Metal), logs in
`/tmp/wc1_at_parent.log`, `/tmp/wc1_at_0e41c260d.log`, per-file signature logs `/tmp/parent_sig_*.log`.

## Set arithmetic (measured)

| commit | failing ids | notes |
| --- | ---: | --- |
| parent | 55 | 54 survive to HEAD; 1 fixed by HEAD |
| `0e41c260d` | 101 | = parent 55 + 46 newly introduced |
| HEAD | 122 | = 54 pre-existing + 46 migration-introduced + 22 added after the migration |

`HEAD ∩ 0e41c260d` = 100; `HEAD − 0e41c260d` = 22; `0e41c260d − HEAD` = 1. No set overlaps
ambiguously; the arithmetic is exact.

## Bucket 1 — introduced by the migration `0e41c260d` (46 ids)

Pass at parent, fail at `0e41c260d` with the **same** UOp-verification signature as HEAD
(verified per-test: same uop index, e.g. `852` kv-tile chunk, `824` static-loop body, `8`
packed-weight GEP). All are the WC0 root cause: `X.alu(Ops.MUL, alpha)` where `alpha` is an
`AMD_ROW_SOFTMAX_SLOT` typed `float` with shape `(8,)`; scalar-typed MUL survives until
`expand_native_row_softmax_repack` substitutes a `STACK float.vec(8)`, then `spec.py:164` /
`spec.py:69` reject it.

| HEAD signature | count | representative ids |
| --- | ---: | --- |
| `MUL` of two `STACK float.vec(8)` | 28 | `test_full_kernel_compiles_with_unaligned_kv_tokens_and_partial_final_chunk[1..15]`, `test_gfx1100_model_grid_static_loop_body_is_invariant[...]`, `acc_slice_v2_drain[...]`, `lowering_baseline` check x2, `model_grid_final_wmma_role_ledger`, `grid_causal_mask_is_fused`, `q16_kv64_loop_reaches_bounded_final_isa`, `shared_attention_compiler_capture` |
| `IndexError` in INDEX shape (`ops.py:265`) | 12 | `test_tensor_vector_load_gep_spec.py` (all 12) |
| `GEP` of vector `LOAD` | 3 | `test_packed_weight.py` tile-producer x3 |
| exact `float.vec(8)` repack `ValueError` (`softmax.py:40`) | 2 | `q16_kv32_builder_is_one_online_chain`, `q16_kv32_hd128_has_exact_shared_p_and_output_ownership` |
| stale assertion (`assert CONTRACT is AMD_ROW_SOFTMAX_SLOT`) | 1 | `native_qk_consumer_exposes_raw_c_and_reaches_two_wmmas` |

These 46 are genuinely new failures: the tests passed at parent. They are compile-only tests,
so nothing env-dependent was masking them.

## Bucket 2 — pre-existing at parent, signature unchanged at HEAD (28 ids + 1 subtest parent)

These fail identically at parent, `0e41c260d`, and HEAD. **Not caused by the migration.**

| HEAD signature | count | why they fail |
| --- | ---: | --- |
| `cycle detected while indexing` (`rangeify.py:859`) | 10 | Pre-existing defect in the semantic-attention lowering: `split_store` → `find_bufs` rejects a `PARAM` buffer that is indexed by two different producer ops in one graph (`ParamArg(4, device='NV')` etc.). Same failure in `test_attention_semantic` (8), `test_composite_scalar_loop` (1), `test_nested_composite_reduce` (1). |
| `AssertionError` (heterogeneous) | 6 | `online_softmax_l_isolation`, `process_isolated_guarded_dispatch_passes`, `hcq_interface_allocator` (`assert 4096 is None`), `host_safety_canary` (`'device_lost' == 'timed_out'`, timing-sensitive), `lowering_fingerprint` (trace mutates fingerprint set), `isolated_timeout_retains_result_published_before_teardown_hang`. Each pre-existing; several are timing/env-sensitive on this box. |
| no AMD device | 9 | Environmental: `AMD:0 does not exist (0 devices available)`; 3 of the 9 are compound `required final_isa_manifest/resource_summary unavailable` wrapping the AMD sub-exception. |
| `unmasked row-softmax repack` (`ops.py:1691`) | 1 | `test_native_repack_all_valid_mode_does_not_enter_causal_specialization` — pre-existing validation-identity error, same at all three commits. |
| `FullKernelAdmissionError: capability_geometry` | 2 | `packed_attn_qo_compile_only[...]` x2 — at parent these failed env (`manifest unavailable` because compile needs AMD); at HEAD the fp16 overlay admission work (S3/S4) changed them to fail the capability gate (`no declared tensor-core family`). Both are "cannot admit on this box". |
| `AMD:ISA unsupported WMMA operand carrier dtypes.half` | 3 | `q16_causal_tail_reaches_final_isa_without_intermediate_buffers`, `q16_live_owner_builder...`, `split_score_state_pv_slice_direct_diagnostic`. At parent these failed with `AssertionError vgpr 175 vs 180` or other; at HEAD the carrier is a scalar `half` and `_wmma_operand_regs` (`amd_wmma_residency.py:147`) rejects it. Signature changed, but the tests were already red at parent. |
| exact `float.vec(8)` repack `ValueError` (`softmax.py:40`) | 4 | `q16_kv32_hd128_numeric`, `reaches_spill_free_final_isa`, `numeric_two_tile_transition`, `reaches_final_isa_program` — red at parent (env/other); now fail the build with the dtype check. |
| emitted-code fixture SHA mismatch | 1 test (3 subtests) | `test_wmma_emitted_code_fixtures_are_unchanged`: recorded `binary_sha256`/`mnemonic_sha256` goldens do not match current emitter output **at all three commits** (SUBFAILED at parent too). The goldens are stale relative to the emitter independently of the migration; the delta is un-reviewed. This is exactly the silent-movement class the campaign warns about. |

## Bucket 3 — pre-existing at parent, but now masked by the carrier defect (19 ids)

At parent these failed for non-verification reasons (mostly no-AMD-device at device open, plus
two `UnboundLocalError: wave_id` at `amd.py:1114`). At HEAD they fail earlier, at build, with
carrier `UOp verification`:

- `MUL` of two `STACK float.vec(8)`: 6 — `attention_residency_contract` x2 (parent: `assert 0 == 1`),
  `lowering_baseline` build artifact, `acc_slice_v2_two_launch`, `lds_rotating_pv_pressure`,
  `q32_hq4_hkv2_kv64_grid_loop_final_isa`.
- `MUL` with `AMD_ROW_SOFTMAX_SLOT` operand: 13 — all the numeric attention tests
  (`q16_kv64_loop_numeric`, causal-tail numerics, model profile/grid numerics,
  `numeric_parity_unaligned_kv_tokens`, `q32_grid_loop_numeric` (parent: `UnboundLocalError
  wave_id`), etc.). At parent the graphs built and died at `AMD:0`; at HEAD they never build.

Net effect: the migration converted ~19 env/other reds into build-time carrier reds, and added
46 new reds on top.

## Bucket 4 — added after the migration `0e41c260d` (22 ids)

All pass at `0e41c260d` (the tests or their assertions did not exist there) and fail at HEAD.

| HEAD signature | count | why they fail |
| --- | ---: | --- |
| Metal env (`libSystem.dylib`) | 20 | New/updated tests in files touched by 1–3 post-migration commits (`test_metal_graph` synthetic suite, `test_flash_decode_intrinsics_renderer_lowering`, `test_generic_tc_split_range_axis`, `test_kernel_lds_mapping`, `test_search_provider`, `test_warp_shfl_xor_renderer_lowering`). Environmental on this Linux box. Includes all 9 ERROR ids. |
| `AssertionError` GEP-lane store | 1 | `test_distinct_gep_store_keeps_ordinary_lane_store_semantics` — standalone-reproducible; introduced by the hot-path refactor sync (`90e93875c`) which changed GEP/vector store shape while the test asserts the old lane-store identity. |
| order-dependent | 1 | `test_runtime_tracemeta_context_controls_explicit_metadata_without_import_time_mode` — fails in-suite, passes standalone. |

## Bucket 5 — fixed between parent and HEAD (not in HEAD's failing set)

- `test_verify_full_output_correctness_against_immutable_artifact` — `NVRTC_ERROR_COMPILATION`
  at parent, passes at HEAD.
- The `UnboundLocalError: wave_id` defect at `amd.py:1114` itself is gone at HEAD (no test
  fails with it anymore), even though two of its former victims now fail on the carrier defect.

## Bottom line (numbers only)

- 46 of 122 ids are new failures introduced by the migration commit; all share the WC0
  carrier root cause and are the actionable set for WC2.
- 54 of 122 predate the migration; of those, 19 changed signature to carrier-verification at
  HEAD, 28+1 are unchanged reds (10 pre-existing rangeify defect, 6 assertions, 9 no-AMD env,
  1 validation, 2 admission-gate, 3 residency, 4 softmax-dtype, 1 fixture-golden parent),
  and 1 was fixed.
- 22 of 122 are post-migration additions: 20 Metal-env, 1 real (devectorizer lane-store
  assertion from the hot-path refactor sync), 1 flaky.
- Environmental at HEAD: 29 (20 Metal + 9 no-AMD); flaky: 1. These should be skips, not
  baseline noise (see the cross-platform gating discussion).
- The campaign's "~114" and the task's "72" are not reproducible as stated on this box/env:
  measured HEAD = 116 failed (NV), of which 47 are carrier-verification; DEV=AMD = 171 failed
  (55 extra ids, all hardware-absent or GGUF-path). Reconciliation deltas are recorded, not
  guessed.
