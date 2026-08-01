# WC1 — failing-baseline audit (`test/unit`, HEAD `e3f2173ad`)

Task: `docs/dtype-carrier-census-task-deepseek-20260731.md` §3. Every failing id below comes
from commands run on this box (Linux, no AMD GPU, no Metal). Logs: `/tmp/wc1_full_run.log`
(DEV=NV), `/tmp/wc1_full_run_amd.log` (DEV=AMD), signature map `/tmp/wc1_map_fixed.json`.

## Runs

Both runs: `pytest test/unit -q --tb=no --ignore=test/unit/test_target_capability_facts.py`

| env | failed | passed | skipped | errors | subtests passed | wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DEV=NV | 116 | 1813 | 29 | 9 | 8 | 151.50s |
| DEV=AMD | 171 | 1758 | 29 | 9 | 8 | 123.03s |

Repro file (`test_online_softmax_tile.py`): **38 failed / 49 passed** at HEAD, matching the
task's stated expectation.

Id accounting (DEV=NV): 113 `FAILED` lines + 9 `ERROR` lines = 122 unique ids; 116 failed =
113 test-level failures + 3 subtest failures, all three inside one parent test
(`test_amd_isa_extraction_fixtures.py::...test_wmma_emitted_code_fixtures_are_unchanged`).

**The failing set is env-dependent.** DEV=AMD is a strict superset by id: 0 NV-only ids,
55 AMD-only ids (168 FAILED + 9 ERROR = 177 lines there). The campaign's "~114" figure is
therefore a property of a specific env; per-id AMD signatures were collected for the 55
AMD-only ids only.

## Groups (DEV=NV), by actual error signature

### 1. `UOp verification failed` — scalar `Ops.MUL` of two `STACK float.vec(8)` (34)

`spec.py:69: RuntimeError: UOp verification failed at N on Ops.MUL dtypes.float 2
[(Ops.STACK, dtypes.float.vec(8), None), (Ops.STACK, dtypes.float.vec(8), None)]`

- `test_amd_attention_kv_tile_oob_guard.py`: `test_full_kernel_aligned_hot_path_isa_is_spill_free_and_unchanged_shape`; `test_full_kernel_compiles_with_unaligned_kv_tokens_and_partial_final_chunk[1..15]`
- `test_attention_residency_contract.py`: `test_native_gqa_prefill_semantic_owner_reaches_one_grid_wmma_body[32]`, `[40]`
- `test_lowering_baseline.py`: `test_build_artifact_is_deterministic_in_process`, `test_check_detects_a_real_mutated_source_hash`, `test_check_passes_against_a_freshly_written_baseline`
- `test_online_softmax_tile.py`: `test_gfx1100_acc_slice_v2_drain_preserves_output_block_base_to_amd_stores[0-expected_blocks0]`, `[4-expected_blocks1]`, `test_gfx1100_acc_slice_v2_two_launch_causal_diagnostic`, `test_gfx1100_grid_causal_mask_is_fused_without_bool_or_infinity_vgprs`, `test_gfx1100_lds_rotating_pv_pressure_compile_microgate`, `test_gfx1100_model_grid_final_wmma_role_ledger`, `test_gfx1100_model_grid_static_loop_body_is_invariant[32-8-512]`, `[32-8-4096]`, `[40-8-512]`, `[40-8-4096]`, `test_gfx1100_q16_kv64_hd128_loop_reaches_bounded_final_isa`, `test_gfx1100_q32_hq4_hkv2_kv64_hd128_grid_loop_final_isa`
- `test_shared_attention_compiler_capture.py`: `test_constructor_uses_actual_scheduled_call_and_final_hip_amdisa_programs`

Same carrier defect as WC0 § Verified mechanism, observed after slot expansion (see group 2).

### 2. `UOp verification failed` — scalar `Ops.MUL` with an `AMD_ROW_SOFTMAX_SLOT` operand (13)

Same `spec.py:69` error text but the second src is `(Ops.AMD_ROW_SOFTMAX_SLOT, dtypes.float,
None)` — the failure caught before expansion.

- `test_amd_attention_kv_tile_oob_guard.py`: `test_numeric_parity_unaligned_kv_tokens_against_reference_softmax_attention`
- `test_online_softmax_tile.py`: `test_gfx1100_grid_fused_causal_mask_numeric_tail_and_fully_masked[0--32]`, `[61-29]`, `[64-32]`, `test_gfx1100_model_grid_causal_mask_uses_runtime_q_tile[8-2]`, `[10-2]`, `test_gfx1100_model_profile_grid_numeric_first_and_prefix[32-8-512-0]`, `[40-8-1024-512]`, `test_gfx1100_q16_kv64_hd128_loop_causal_tail_numeric[61-None]`, `[64--16]`, `[64-None]`, `test_gfx1100_q16_kv64_hd128_loop_numeric`, `test_gfx1100_q32_hq4_hkv2_kv64_hd128_grid_loop_numeric`

Groups 1+2 are the carrier-defect failures: **47 total**, not 72. The task's "72 occurrences at
HEAD" is not reproduced on this box/env; the 72-vs-47 delta is a reconciliation question, not
guessed away here.

### 3. `UOp verification failed` — `Ops.GEP` of a vector `LOAD` (3)

`Ops.GEP dtypes.uint 1 [(Ops.LOAD, dtypes.uint.vec(4|2), None)] (0,)` /
`Ops.GEP dtypes.ushort 1 [(Ops.LOAD, dtypes.ushort.vec(4), None)] (0,)`

- `test_packed_weight.py`: `test_tile_fp16_producer_matches_scalar_across_group_and_block_boundaries[Q4_K-248-16]`, `[Q4_K-28-8]`, `[Q6_K-248-16]`

Note: this signature is run-mode-sensitive; the standalone per-file run produces it while the
in-suite run cannot be attributed by id without `--tb`. The map records the standalone signature.

### 4. Emitted-code fixture SHA mismatch (1 test, 3 subfailures)

`test_amd_isa_extraction_fixtures.py::TestAMDISAExtractionFixtures::test_wmma_emitted_code_fixtures_are_unchanged`
subtests `tc_16x16x16_unrolled`, `tc_16x16x64_unrolled`, `tc_16x16x64_rolled`: recorded
`binary_sha256`/`mnemonic_sha256` fixture hashes do not match current emitter output. Standalone
reproduces (3 failed, 6 passed).

### 5. Metal environment — `OSError: /usr/lib/libSystem.dylib` (20)

Missing Metal runtime on Linux. Includes all 9 ERROR ids (setup errors) plus 11 FAILED ids:

- `test_flash_decode_intrinsics_renderer_lowering.py` (3): capability-properties, fdot2 semantics, no-amd-builtin-compiles-on-real-metal
- `test_generic_tc_split_range_axis.py` (2): `[METAL]` x2
- `test_kernel_lds_mapping.py` (2): cooperative-store rotation, renderers-declare-lds-bank-facts
- `test_metal_graph.py` (11): TestMetalDirectEncoderSynthetic (2), TestMetalGraphAdmissionSynthetic (3), TestMetalHybridReplaySynthetic (6)
- `test_search_provider.py` (1): metal target facts
- `test_warp_shfl_xor_renderer_lowering.py` (1): simd shuffle xor source

### 6. No AMD device (9)

`RuntimeError: AMD:0 does not exist (0 devices available)` /
`ExceptionGroup: No interface for AMD:0 is available (3 sub-exceptions)` (with
`FileNotFoundError: /dev/kfd` + `IndexError` in the chain). Three of the nine are compound:
`ValueError: required final_isa_manifest/resource_summary unavailable: final compile failed
(ExceptionGroup: No interface for AMD:0 ...)` at `extra/llm_research/prefill/current_prefill_execution_adapter.py:111`.

- `test_attention_semantic.py`: `test_exact_q16_gfx1100_native_runtime_matches_reference`
- `test_attn_qo_register_compile.py`: `test_attn_qo_register_prefill_compile_is_cpu_only_and_zero_lds`
- `test_current_prefill_execution_adapter.py` (3): packed-shipping gates `[Q4_K-uint32]`, `[Q6_K-uint16]`, promoted-attn-qo
- `test_online_softmax_tile.py` (4): pv-c-lane lowers, q16 causal tail `[13--16]`, `[13-0]`, `[16-0]`

### 7. `RuntimeError: cycle detected while indexing` (10)

`tinygrad/schedule/rangeify.py:859`, raised in `find_bufs`.

- `test_attention_semantic.py` (8): bounded-semantic admission (causal-mask source, gqa+additive, inlines-qk, keeps-hd, scoped-composite, fp16-exact, scheduler-one-fused-call, wmma-shape)
- `test_composite_scalar_loop.py`: `test_semantic_composite_scalar_loop_q16_kv16_hd64_cpu`
- `test_nested_composite_reduce.py`: `test_composite_reduce_state_adapter_q16_hd64_fp16_numeric_gate`

### 8. `ValueError: gfx1100 ... repack requires exact dtypes.float.vec(8) m/l row state` (6)

`tinygrad/schedule/wmma/softmax.py:40`.

- `test_online_softmax_tile.py` (6): q16-kv32 builder/ownership/numeric/spill-free, numeric-two-tile, reaches-final-isa

### 9. `ValueError: unmasked row-softmax repack must retain the canonical validity identity` (1)

`tinygrad/uop/ops.py:1691`.

- `test_amd_wave_lds_fence.py`: `test_native_repack_all_valid_mode_does_not_enter_causal_specialization`

### 10. `IndexError: tuple index out of range` in INDEX shape (12)

`tinygrad/uop/ops.py:265` (`Ops.INDEX` shape computation).

- `test_tensor_vector_load_gep_spec.py` (12): integer-load scalar-gep `[dtype0..dtype3]`, rejects-invalid-dtype, rejects-invalid-lane `[None,arg0..arg3]`, rejects-out-of-range-lane `[-1,4]`

### 11. `FullKernelAdmissionError: capability_geometry` (2)

`extra/llm_research/runtime_specs.py:483`: `capability_geometry: capability_device: no declared
tensor-core family ...`.

- `test_current_prefill_execution_adapter.py`: `test_packed_attn_qo_compile_only_is_one_fp16_wmma_program_with_packed_b_abi[Q4_K-...-144-36]`, `[Q6_K-...-210-105]`

### 12. `AssertionError` — heterogeneous (8)

- `test_composite_reduce.py`: `test_online_softmax_l_isolation`
- `test_current_decode_execution_adapter.py`: `test_process_isolated_guarded_dispatch_passes` (`{'schema': 'isolated-guarded-executor.v1', ...}`)
- `test_devectorizer_output_safety.py`: `test_distinct_gep_store_keeps_ordinary_lane_store_semantics` (GEP store assertion)
- `test_hcq_interface_allocator.py`: `test_amd_interface_allocator_publishes_its_large_allocation_granularity` (`assert 4096 is None`)
- `test_host_safety_canary_20260713.py`: `test_synthetic_child_hang_times_out_and_parent_survives_with_cleanup` (`'device_lost' == 'timed_out'`)
- `test_lowering_fingerprint.py`: `test_enabling_the_trace_does_not_change_the_fingerprint` (hash dict mismatch)
- `test_online_softmax_tile.py`: `test_native_qk_consumer_exposes_raw_c_and_reaches_two_wmmas` (`assert (Ops.CONTRACT is Ops.AMD_ROW_SOFTMAX_SLOT)`)
- `test_process_isolated.py`: `test_isolated_timeout_retains_result_published_before_teardown_hang`

### 13. `NotImplementedError: AMD:ISA unsupported WMMA operand carrier dtypes.half` (3)

`tinygrad/renderer/isa/amd_wmma_residency.py:147`.

- `test_online_softmax_tile.py`: `test_gfx1100_q16_causal_tail_reaches_final_isa_without_intermediate_buffers`, `test_gfx1100_q16_live_owner_builder_feeds_proven_dual_wmma_pipeline`, `test_gfx1100_split_score_state_pv_slice_direct_diagnostic`

### 14. Order-dependent / flaky (1)

`test_graph_admission.py::test_runtime_tracemeta_context_controls_explicit_metadata_without_import_time_mode`
failed in the full-suite run; passes standalone (verified: `1 passed`). Not attributable to a
signature; recorded as order-dependent.

## Collection-blocked file

`test/unit/test_target_capability_facts.py` (12 tests) errors at collection on this box:
`OSError: /usr/lib/libSystem.dylib` (imports `tinygrad.runtime.graph.metal`). Excluded from both
runs via `--ignore`; environmental.

## Env delta (DEV=AMD minus DEV=NV), 55 ids

All 55 fail under DEV=AMD but pass under DEV=NV:

- 53: `RuntimeError: AMD:0 does not exist (0 devices available)` (`/dev/kfd` absent) —
  `test_mmq_q4k_q8_reference` (16), `test_packed_weight` (5), `test_attention_semantic` (7),
  `test_mmq_q4k_q8_atom` (4), `test_model_route_plan` (6), `test_mmq_ds4_logical_emitter` (2),
  `test_composite_reduce` (2), `test_recompute_hostile_cost_gate` (2),
  `test_qk_capability_policy_gate` (2), `test_online_softmax_state_split` (2),
  `test_rangeify_multireduce` (1), `test_q4k_w_f16_decode` (1), `test_nested_composite_reduce` (1),
  `test_lowering_invariants` (1), `test_current_decode_execution_adapter` (1)
- 2: `RuntimeError: qwen3: selected-GGUF backing allocation is unknown from the selected path and
  scanned allocation granularity` at `tinygrad/llm/model.py:1110` — `test_prefill_overlay_roles.py`
  (`test_inventory_overlay_bytes_equal_model_walk_on_real_qwen3_8b`,
  `test_nv_no_promoted_candidate_census_in_admit_report`)

## Totals and reconciliation

- Unique failing/erroring ids at DEV=NV: 122 (113 FAILED + 9 ERROR); 121/122 mapped to a
  signature; the 1 unmapped is the flaky test (group 14).
- The parent of the 3 fixture subfailures (group 4) is a 123rd unique id, reported by pytest as
  3 failed subtests.
- Groups 1+2 (carrier defect) = 47; task's stated 72 is not reproduced here. The campaign's
  "~114" is env-dependent (116 vs 171 by DEV). Both deltas are recorded, not reconciled by
  guessing.
