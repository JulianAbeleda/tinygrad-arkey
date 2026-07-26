# Codebase organization audit

Audited commit: `mac-first-boot-20260610-3419-gbece3963e-dirty` (dirty: True)
Scope: `<repository>` | manifest coverage required for: extra/qk/
Verdict: **ORG_R1_BLOCKED_ORGANIZATION_DRIFT** (1 hard errors, 82 warnings)

> Machine-derived facts (discovery, LOC, imports, references, coverage, boundary checks) are produced by this
> script. Every purpose, role, status, disposition, action, and promotion/prune judgment below is human-authored
> in `extra/audit/codebase_organization_manifest.json` and is reproduced here, not computed.

## Coverage

- Authored: 435 files / 68387 token-bearing LOC (sz.py rules)
- Generated (reported, never manifested): 1 files / 57 LOC
- Manifest scope: 96 files / 13275 LOC (96 explicit records, 0 covered by group rule, 0 uncovered)

## LOC by domain

| domain | files | loc |
|---|---|---|
| amd_runtime | 3 | 420 |
| attention_decode | 10 | 788 |
| attention_prefill | 4 | 574 |
| codegen_lowering | 10 | 560 |
| evidence | 10 | 1608 |
| measurement | 17 | 1941 |
| quant_mmq | 18 | 2930 |
| route_authority | 16 | 3447 |
| search_promotion | 8 | 1007 |
| unclassified | 339 | 55112 |

## LOC by role

| role | files | loc |
|---|---|---|
| adapter | 11 | 1143 |
| authority | 51 | 7830 |
| benchmark | 2 | 99 |
| diagnostic | 11 | 1232 |
| evidence | 5 | 438 |
| execution | 9 | 1811 |
| integration | 2 | 199 |
| research | 4 | 387 |
| test | 1 | 136 |
| unclassified | 339 | 55112 |

## LOC by status

| status | files | loc |
|---|---|---|
| active_research | 13 | 2285 |
| fallback | 5 | 272 |
| historical_one_off | 1 | 43 |
| production | 67 | 9532 |
| promoted_default | 6 | 648 |
| refuted | 1 | 21 |
| retained_reference | 1 | 114 |
| unclassified | 339 | 55112 |
| unresolved_reproducer | 2 | 360 |

## LOC by disposition

| disposition | files | loc |
|---|---|---|
| consolidate | 2 | 864 |
| investigate | 2 | 256 |
| keep | 92 | 12155 |
| unclassified | 339 | 55112 |

## Default-path source footprint

54 declared default-path files / 8856 LOC.

- `extra/qk/amd_isa_proof.py`
- `extra/qk/amd_resource_artifact.py`
- `extra/qk/amd_warp_reduce.py`
- `extra/qk/attention_harness_common.py`
- `extra/qk/cooperative_stage_lanemap.py`
- `extra/qk/decode/current_decode_execution_adapter.py`
- `extra/qk/decode/flash_decode_attention_executor.py`
- `extra/qk/decode/flash_decode_attention_spec.py`
- `extra/qk/flash_common.py`
- `extra/qk/flash_kernels.py`
- `extra/qk/gemv_g2_lanemap.py`
- `extra/qk/gemv_g3_codegen_lowering.py`
- `extra/qk/kernel_vocabulary.py`
- `extra/qk/kv_load.py`
- `extra/qk/lane_partition_reduce.py`
- `extra/qk/layout.py`
- `extra/qk/live_split_geometry.py`
- `extra/qk/memory_adaptive_allocation_observer.py`
- `extra/qk/memory_adaptive_policy.py`
- `extra/qk/memory_adaptive_runtime_collector.py`
- `extra/qk/mmq_atom_boundary.py`
- `extra/qk/mmq_compile_evidence.py`
- `extra/qk/mmq_ds4_probe_contract.py`
- `extra/qk/mmq_epoch_manifest_export.py`
- `extra/qk/mmq_lifecycle.py`
- `extra/qk/mmq_logical_vocabulary.py`
- `extra/qk/mmq_q4k_q8_atom.py`
- `extra/qk/mmq_q4k_q8_reference.py`
- `extra/qk/model_profiles.py`
- `extra/qk/operand_attribution.py`
- `extra/qk/packed_wmma_compile_gate.py`
- `extra/qk/prefill/candidate_payloads.py`
- `extra/qk/prefill/current_prefill_execution_adapter.py`
- `extra/qk/prefill/executable_artifact_preparation.py`
- `extra/qk/prefill/execution_bridge_contracts.py`
- `extra/qk/prefill/flash_prefill_attention_spec.py`
- `extra/qk/prefill/guarded_execution.py`
- `extra/qk/prefill/host_safety_canary.py`
- `extra/qk/prefill/isolated_guarded_executor.py`
- `extra/qk/prefill/operand_path_execution_worker.py`
- `extra/qk/prefill/packed_wmma_correctness_canary.py`
- `extra/qk/prefill/packed_wmma_prefill_candidates.py`
- `extra/qk/prefill/prefill_graph_gemm_route.py`
- `extra/qk/prefill/pure_register_compile_capture.py`
- `extra/qk/prefill/pure_register_evaluation_gate.py`
- `extra/qk/prefill/q4k_prefill_route_spec.py`
- `extra/qk/prefill/q6k_prefill_route_spec.py`
- `extra/qk/q4k_tile_loader.py`
- `extra/qk/q6k_route_spec.py`
- `extra/qk/quant/q4_k_gemv_primitive.py`
- `extra/qk/quant/q6_k_gemv_primitive.py`
- `extra/qk/route_manifest.py`
- `extra/qk/runtime_specs.py`
- `extra/qk/warp_reduce_lowering.py`

## Largest authored files

| loc | path | domain |
|---|---|---|
| 2267 | `tinygrad/renderer/isa/amd.py` | None |
| 2124 | `tinygrad/uop/ops.py` | None |
| 1084 | `test/unit/test_online_softmax_tile.py` | None |
| 1067 | `tinygrad/llm/model.py` | None |
| 903 | `tinygrad/runtime/ops_amd.py` | None |
| 832 | `extra/qk/mmq_q4k_q8_atom.py` | quant_mmq |
| 794 | `test/amd/disasm.py` | None |
| 758 | `extra/qk/runtime_specs.py` | route_authority |
| 700 | `tinygrad/tensor.py` | None |
| 660 | `tinygrad/schedule/rangeify.py` | None |
| 641 | `test/unit/test_runtime_specs.py` | None |
| 634 | `tinygrad/runtime/ops_nv.py` | None |
| 619 | `tinygrad/renderer/isa/x86.py` | None |
| 598 | `tinygrad/codegen/opt/postrange.py` | None |
| 596 | `tinygrad/mixin/__init__.py` | None |
| 576 | `extra/audit/codebase_organization_audit.py` | None |
| 576 | `tinygrad/renderer/amd/sqtt.py` | None |
| 556 | `extra/llm/cli.py` | None |
| 552 | `tinygrad/viz/js/profiler.js` | None |
| 538 | `test/unit/test_amd_isa_wmma.py` | None |

## Duplicate authority

None: every declared authority key has one owner.

## Cross-domain dependency violations

None.

## Import cycles

- ['extra/hardware/sqtt/roc.py', 'tinygrad/renderer/amd/sqtt.py', 'tinygrad/viz/http.py', 'tinygrad/viz/profile.py', 'tinygrad/viz/render.py', 'tinygrad/viz/serve.py']
- ['tinygrad/__init__.py', 'tinygrad/callify.py', 'tinygrad/codegen/__init__.py', 'tinygrad/codegen/gpudims.py', 'tinygrad/codegen/late/composite_combines.py', 'tinygrad/codegen/late/devectorizer.py', 'tinygrad/codegen/late/expander.py', 'tinygrad/codegen/late/flash_attn.py', 'tinygrad/codegen/late/gater.py', 'tinygrad/codegen/late/linearizer.py', 'tinygrad/codegen/late/reduce_lowering.py', 'tinygrad/codegen/late/reg_store.py', 'tinygrad/codegen/late/regalloc.py', 'tinygrad/codegen/opt/heuristic.py', 'tinygrad/codegen/opt/kernel_lds.py', 'tinygrad/codegen/opt/kernel_pipeline.py', 'tinygrad/codegen/opt/packed_weight.py', 'tinygrad/codegen/opt/postrange.py', 'tinygrad/codegen/simplify.py', 'tinygrad/device.py', 'tinygrad/engine/jit.py', 'tinygrad/engine/realize.py', 'tinygrad/function.py', 'tinygrad/llm/fused_attention.py', 'tinygrad/mixin/__init__.py', 'tinygrad/mixin/movement.py', 'tinygrad/mixin/rand.py', 'tinygrad/mixin/reduce.py', 'tinygrad/nn/state.py', 'tinygrad/renderer/__init__.py', 'tinygrad/renderer/isa/__init__.py', 'tinygrad/schedule/__init__.py', 'tinygrad/schedule/allreduce.py', 'tinygrad/schedule/flash_fusion.py', 'tinygrad/schedule/indexing.py', 'tinygrad/schedule/memory.py', 'tinygrad/schedule/multi.py', 'tinygrad/schedule/rangeify.py', 'tinygrad/schedule/wmma/__init__.py', 'tinygrad/schedule/wmma/composite.py', 'tinygrad/schedule/wmma/fragments.py', 'tinygrad/schedule/wmma/kernels.py', 'tinygrad/schedule/wmma/loop_state.py', 'tinygrad/schedule/wmma/softmax.py', 'tinygrad/tensor.py', 'tinygrad/uop/decompositions.py', 'tinygrad/uop/divandmod.py', 'tinygrad/uop/ops.py', 'tinygrad/uop/render.py', 'tinygrad/uop/spec.py', 'tinygrad/uop/symbolic.py', 'tinygrad/uop/upat.py', 'tinygrad/uop/validate.py']
- ['tinygrad/renderer/cstyle.py', 'tinygrad/renderer/isa/amd.py', 'tinygrad/renderer/llvmir.py', 'tinygrad/runtime/graph/hcq.py', 'tinygrad/runtime/ops_amd.py', 'tinygrad/runtime/support/am/amdev.py', 'tinygrad/runtime/support/c.py', 'tinygrad/runtime/support/compiler_amd.py', 'tinygrad/runtime/support/hcq.py', 'tinygrad/runtime/support/system.py', 'tinygrad/runtime/support/usb.py']
- ['tinygrad/runtime/graph/cuda.py', 'tinygrad/runtime/ops_cuda.py']

## Files with no inbound import and no reference anywhere

An entry point, gate, probe, or operator tool may legitimately have no inbound import; this list is evidence
for human classification, not a death sentence.

- `extra/gpu_fault_analysis/bench_alloc_trace.py`
- `extra/hardware/amdpci/am_smi.py`
- `extra/hardware/amdpci/generate_deep_psp_trace.py`
- `extra/hardware/amdpci/linux_mmhub_gart_snapshot.py`
- `extra/hardware/amdpci/proclogs.py`
- `extra/hardware/amdpci/receive_capture_tar.py`
- `extra/hardware/sqtt/generate_examples.py`
- `extra/hardware/sqtt/rgptool.py`
- `extra/llm/llama_kv_ctx_slope_bench.py`
- `extra/llm/sft_smoke_train.py`
- `extra/qk/q4k_fused_mmq_contract.py`
- `extra/remote/bench.py`
- `extra/tools/amd_isa_generate.py`
- `extra/tools/check_doc_links.py`
- `scratch_attn_bench.py`
- `scratch_attn_bench2.py`
- `test/backend/test_softmax_fusion.py`
- `test/helpers.py`
- `test/test_guarded_execution.py`
- `test/test_runtime_bridge.py`
- `test/unit/test_alloc_trace.py`
- `test/unit/test_am_experiment_registry.py`
- `test/unit/test_am_psp_mem.py`
- `test/unit/test_amd_aql_packet_publication.py`
- `test/unit/test_amd_attention_kv_tile_oob_guard.py`
- `test/unit/test_amd_compile_capture_fail_closed_20260712.py`
- `test/unit/test_amd_elf_entry_offset.py`
- `test/unit/test_amd_epilogue_address_schedule_probe.py`
- `test/unit/test_amd_final_elf_capture_20260712.py`
- `test/unit/test_amd_isa_extraction_fixtures.py`
- `test/unit/test_amd_isa_integer_vector_loads.py`
- `test/unit/test_amd_isa_integer_wmma_hardware_correctness.py`
- `test/unit/test_amd_kfd_fault_event_reset.py`
- `test/unit/test_amd_resource_artifact.py`
- `test/unit/test_amd_resource_artifact_intervals_20260712.py`
- `test/unit/test_amd_wave_lds_fence.py`
- `test/unit/test_amdllvm_waitcnt.py`
- `test/unit/test_analyze_faults.py`
- `test/unit/test_attn_qo_register_compile.py`
- `test/unit/test_bench_entrypoint.py`
- `test/unit/test_candidate_context_propagation.py`
- `test/unit/test_clock_pin.py`
- `test/unit/test_compiler_amd_pure_disassembly_20260712.py`
- `test/unit/test_composite_axis_constraints.py`
- `test/unit/test_composite_scalar_loop.py`
- `test/unit/test_composite_tile_carrier.py`
- `test/unit/test_current_decode_execution_adapter.py`
- `test/unit/test_current_prefill_execution_adapter.py`
- `test/unit/test_decode_resource_capture.py`
- `test/unit/test_devectorizer_memory_widths.py`
- `test/unit/test_devectorizer_reconstruction.py`
- `test/unit/test_disk_staging_timeout.py`
- `test/unit/test_dispatch_trace.py`
- `test/unit/test_executable_artifact_preparation.py`
- `test/unit/test_execution_bridge_contracts.py`
- `test/unit/test_final_regalloc_proof_transport_20260712.py`
- `test/unit/test_flash_decode_attention_spec.py`
- `test/unit/test_gemm_consumer_adapters.py`
- `test/unit/test_generated_quant_binding_audit.py`
- `test/unit/test_gguf_memory_scan.py`
- `test/unit/test_hcq_graph_profile_export.py`
- `test/unit/test_hcq_interface_allocator.py`
- `test/unit/test_hcq_kernargs_contract.py`
- `test/unit/test_host_safety_canary_20260713.py`
- `test/unit/test_isolated_guarded_executor.py`
- `test/unit/test_kernel_candidate_context.py`
- `test/unit/test_kernel_lds_mapping.py`
- `test/unit/test_kernel_naming.py`
- `test/unit/test_kernel_pipeline_expansion.py`
- `test/unit/test_linearizer_stable_priority.py`
- `test/unit/test_llama_bench_artifacts.py`
- `test/unit/test_llm_context_admission.py`
- `test/unit/test_llm_decode_correctness.py`
- `test/unit/test_llm_decode_routes.py`
- `test/unit/test_llm_model_lm_head_prefill_route.py`
- `test/unit/test_memory_adaptive_allocation_observer.py`
- `test/unit/test_memory_adaptive_exact_ledger.py`
- `test/unit/test_memory_adaptive_model_integration.py`
- `test/unit/test_memory_adaptive_policy.py`
- `test/unit/test_memory_adaptive_route_manifest.py`
- `test/unit/test_memory_adaptive_runtime_collector.py`
- `test/unit/test_mmq_atom_boundary.py`
- `test/unit/test_mmq_ds4_logical_emitter.py`
- `test/unit/test_mmq_epoch_manifest_export.py`
- `test/unit/test_mmq_lifecycle.py`
- `test/unit/test_mmq_logical_vocabulary.py`
- `test/unit/test_model_facts.py`
- `test/unit/test_model_profiles.py`
- `test/unit/test_model_route_plan.py`
- `test/unit/test_online_softmax_state_split.py`
- `test/unit/test_operand_attribution_20260714.py`
- `test/unit/test_operand_path_execution_worker.py`
- `test/unit/test_packed_weight.py`
- `test/unit/test_packed_wmma_compile_gate.py`
- `test/unit/test_packed_wmma_correctness_canary.py`
- `test/unit/test_pm4_ib_and_nv_cmdq_wrap_drain.py`
- `test/unit/test_precontract_int8_lds_contract.py`
- `test/unit/test_prefill_graph_gemm_route.py`
- `test/unit/test_prefill_harness.py`
- `test/unit/test_prefill_memory_plan.py`
- `test/unit/test_prefill_memory_plan_integration.py`
- `test/unit/test_prefill_route_memory_semantics.py`
- `test/unit/test_prefill_whole_synced.py`
- `test/unit/test_process_isolated.py`
- `test/unit/test_pure_pipe_graph_abi.py`
- `test/unit/test_pure_register_compile_capture.py`
- `test/unit/test_pure_register_do_assemble_capture_boundary_20260712.py`
- `test/unit/test_pure_register_evaluation_gate.py`
- `test/unit/test_pure_register_final_program_adapter_20260712.py`
- `test/unit/test_q4_q4_owner_comparison.py`
- `test/unit/test_q4k_prefill_route_spec.py`
- `test/unit/test_q4k_q8_mmq_prefill_spec.py`
- `test/unit/test_q4k_w_f16_decode.py`
- `test/unit/test_qk_route_purity.py`
- `test/unit/test_rangeify_multireduce.py`
- `test/unit/test_recompute_hostile_cost_gate.py`
- `test/unit/test_regalloc_addr_lifetime.py`
- `test/unit/test_regalloc_candidate_scarcity.py`
- `test/unit/test_regalloc_rematerialization.py`
- `test/unit/test_regalloc_spans.py`
- `test/unit/test_register_contracts.py`
- `test/unit/test_shared_attention_evidence.py`
- `test/unit/test_shared_attention_promotion.py`
- `test/unit/test_shared_attention_replay_admission.py`
- `test/unit/test_shared_attention_synchronization_capture.py`
- `test/unit/test_size_accounting.py`
- `test/unit/test_stage1_wmma_compile_evidence.py`
- `test/unit/test_strided_lds_record_layout.py`
- `test/unit/test_tensor_vector_load_gep_spec.py`
- `test/unit/test_viz_application_boundary.py`
- `test/unit/test_wmma_gep_spec.py`
- `test/unit/test_wmma_value_semantics.py`
- `test/unit/test_wrapping_allocator_invariant.py`
- `test/unit/test_x86_encoding_fixtures.py`
- `tinygrad/llm/__main__.py`
- `tinygrad/runtime/ops_cpu.py`
- `tinygrad/runtime/ops_npy.py`
- `tinygrad/runtime/ops_null.py`
- `tinygrad/runtime/ops_nv.py`
- `tinygrad/runtime/support/autogen.py`
- `tinygrad/viz/js/graph.js`
- `tinygrad/viz/js/profiler.js`
- `tinygrad/viz/js/rewrite.js`
- `tinygrad/viz/js/ui.js`
- `tinygrad/viz/js/worker.js`

## Test and one-off script inventory

| test_role | count | files |
|---|---|---|
| active_regression | 9 | `extra/qk/clock_pin.py`, `extra/qk/decode/decode_codegen_identity_check.py`, `extra/qk/live_split_geometry.py`, `extra/qk/packed_wmma_compile_gate.py`, `extra/qk/prefill/pure_register_compile_capture.py`, `extra/qk/prefill/pure_register_evaluation_gate.py`, `extra/qk/prefill/q4k_prefill_route_spec.py`, `extra/qk/prefill/q4k_q8_mmq_prefill_spec.py`, `extra/qk/timing_harness.py` |
| active_validation | 25 | `extra/qk/amd_isa_proof.py`, `extra/qk/amd_resource_artifact.py`, `extra/qk/bench.py`, `extra/qk/benchmark_shared_attention.py`, `extra/qk/decode/decode_runtime_overhead.py`, `extra/qk/generated_candidates.py`, `extra/qk/memory_adaptive_allocation_observer.py`, `extra/qk/memory_adaptive_policy.py`, `extra/qk/memory_adaptive_runtime_collector.py`, `extra/qk/mmq_atom_boundary.py`, `extra/qk/mmq_ds4_logical_emitter.py`, `extra/qk/mmq_epoch_manifest_export.py`, `extra/qk/mmq_lifecycle.py`, `extra/qk/mmq_logical_vocabulary.py`, `extra/qk/mmq_q4k_q8_atom.py`, `extra/qk/mmq_q4k_q8_reference.py`, `extra/qk/prefill/packed_wmma_correctness_canary.py`, `extra/qk/prefill/prefill_graph_gemm_route.py`, `extra/qk/prefill/prefill_harness.py`, `extra/qk/prefill/prefill_primitive_spec.py`, `extra/qk/prefill/prefill_whole_synced.py`, `extra/qk/pure_search_guard.py`, `extra/qk/shared_attention_capture.py`, `extra/qk/shared_attention_evidence.py`, `extra/qk/shared_attention_promotion.py` |
| operational_tool | 14 | `extra/qk/decode/current_decode_execution_adapter.py`, `extra/qk/decode/decode_harness.py`, `extra/qk/decode/decode_tile_timing.py`, `extra/qk/generate_shared_attention_captures.py`, `extra/qk/mmq_compile_evidence.py`, `extra/qk/packed_wmma_canary_evidence.py`, `extra/qk/phase_abi_v1_resource_probe.py`, `extra/qk/prefill/host_safety_canary.py`, `extra/qk/prefill/packed_wmma_prefill_promotion_gate.py`, `extra/qk/prefill/prefill_boltbeam_trace.py`, `extra/qk/prefill/prefill_causal_tile_skip_promotion_gate.py`, `extra/qk/prefill/prefill_flash_e2e_parity.py`, `extra/qk/prefill/prefill_flash_perf.py`, `extra/qk/prefill/prefill_softmax_reduce_fuse_promotion_gate.py` |
| orphan_unknown | 1 | `extra/qk/bubblebeam_futuresight.py` |
| retained_reference | 2 | `extra/qk/mmq_ds4_probe_contract.py`, `extra/qk/q4k_tile_loader.py` |
| unresolved_reproducer | 4 | `extra/qk/benchmark_split_shared_attention.py`, `extra/qk/decode/decode_hd_sweep_numerics.py`, `extra/qk/prefill/prefill_hd_sweep_numerics.py`, `extra/qk/prefill/prefill_long_context_numerics.py` |

## Decouple candidates

- **`extra/qk/mmq_compile_evidence.py`** -- The shipped prefill and decode execution adapters pull the entire 835-LOC MMQ atom-search file onto the default path through one module-scope import, and with it a module whose status is refuted. One of the two imported names is a constant that layout.py already owns. (gross -0, +0, net -0) | evidence: extra/qk/mmq_compile_evidence.py:21 'from extra.qk.mmq_q4k_q8_atom import Q4K_WORDS_PER_BLOCK, _q4k_q8_1_bounded_ds4_coop_tile_kernel'; extra/qk/prefill/current_prefill_execution_adapter.py and extra/qk/decode/current_decode_execution_adapter.py import mmq_compile_evidence at module scope, so the whole chain loads on the default path; Q4K_WORDS_PER_BLOCK is owned by extra/qk/layout.py:12; mmq_q4k_q8_atom.py:23 merely re-exports it; extra/qk/mmq_q4k_q8_atom.py in turn imports extra/qk/mmq_ds4_probe_contract.py, whose status is refuted -- this is the audit's one hard error, and it is a true positive; mmq_q4k_q8_atom.py holds ~10 progressively elaborated kernel stages; production needs exactly one builder | tests: test/unit/test_mmq_q4k_q8_atom.py keeps covering the atom search; add one assertion that the execution adapters do not import mmq_q4k_q8_atom

## Centralize candidates

- **`extra/qk/amdgpu_metadata.py`** -- parse_amdgpu_metadata is implemented twice from the same llvm-readelf note fields; production adapters already use the mmq_compile_evidence copy, so the standalone module is a second source of truth for one rule. (gross -30, +0, net -30) | evidence: extra/qk/amdgpu_metadata.py:11-33 and extra/qk/mmq_compile_evidence.py:80-96 parse the same .vgpr_count/.sgpr_count/.spill/.group_segment_fixed_size/.wavefront_size/amdhsa.target fields; production consumers import the mmq_compile_evidence version: prefill/current_prefill_execution_adapter.py, decode/current_decode_execution_adapter.py, shared_attention_capture.py; amdgpu_metadata.py is imported only by test/unit/test_q4k_q8_mmq_uop.py, test/unit/test_online_softmax_tile.py, and extra/qk/phase_abi_v1_resource_probe.py | tests: repoint test/unit/test_q4k_q8_mmq_uop.py and test/unit/test_online_softmax_tile.py at the surviving owner; both already project a key subset, so the extra max_workgroup_threads field is inert
- **`extra/qk/attention_harness_common.py`** -- Two files appeared to own the shared_attention_proof schema keys; only one writes them. Authority recorded on the producer, and the reader documented as a consumer. (gross -0, +0, net -0) | evidence: writers: extra/qk/shared_attention_evidence.py:85 (v2), :156 (acc_slice_v3), :180 (phase_v4); extra/qk/attention_harness_common.py:16-22 load_shared_attention_proof only validates status/passed | tests: existing test/unit/test_shared_attention_evidence.py already covers the producer

## Modularize / reusable-asset candidates

None proposed at this evidence level.

## Reuse candidates

- **`extra/qk/prefill/packed_wmma_prefill_promotion_gate.py`** -- Three promotion gates repeat the same evidence-loading and verdict-emitting rule verbatim; a fourth gate that looks similar must NOT be folded in because its verdict semantics differ. (gross -39, +35, net -4) | evidence: packed_wmma_prefill_promotion_gate.py, prefill_softmax_reduce_fuse_promotion_gate.py and prefill_causal_tile_skip_promotion_gate.py each derive required shapes from route_manifest ROUTES shape_guards, read a fixed docs/*.json, check _schema/route_id/flag, fail closed, and print 'AUTHORITY_GATE: {verdict}' with an identical result dict; prefill_softmax_reduce_fuse_promotion_gate.py's docstring calls itself 'Sibling of ... deliberately the same shape'; the two newer gates share the MIN_PAIRS=3 / MIN_MEAN_DELTA_PCT=1.0 / MIN_SIGNAL_TO_NOISE=2.0 / MAX_NOISE_FLOOR_PCT=1.0 threshold block verbatim; EXCLUDED: extra/qk/prefill/pure_register_evaluation_gate.py has no main(), does not import route_manifest, and validates compile-artifact provenance rather than gating a route's evidence file -- different rule | tests: one shared test asserting fail-closed on missing/malformed evidence, plus the three existing per-gate tests unchanged (each still asserts its own thresholds and route id)
- **`extra/qk/mmq_q4k_q8_atom.py`** -- Nine structurally identical *_source_hash wrappers now express one hashing rule through one helper. (gross -3, +0, net -3) | evidence: all nine built a kernel, applied UOp.placeholder args positionally, repr'd the graph and took sha256[:16]; they differed only in builder and placeholder shapes; all ten produced hash values (nine functions plus the coop-tile alternate writeback_mode) verified byte-identical before and after; test/unit/test_mmq_q4k_q8_atom.py: 16 passed; the 14 failures under -k mmq reproduce with the change stashed and are pre-existing | tests: 

## Prune candidates

- **`extra/qk/p2_probe_1.py`** -- Six one-shot fusion-bisection probes whose conclusion is recorded in a doc and in production comments. (gross -44, +0, net -44) | evidence: zero inbound references of any kind; durable verdict in docs/flash-prefill-piece2-probe-20260721.md; conclusion echoed in tinygrad/schedule/flash_fusion.py, tinygrad/schedule/rangeify.py, tinygrad/codegen/opt/heuristic.py | tests: 
- **`extra/qk/shared_attention_evidence_gate.py`** -- A validator for a bundle schema that no code in the repository ever produces. (gross -131, +0, net -131) | evidence: the schema string 'tinygrad.shared_attention_evidence_bundle.v1' appears exactly once repo-wide -- its own definition at extra/qk/shared_attention_evidence_gate.py:12; imported only by test/unit/test_shared_attention_evidence_gate.py; contrast: extra/qk/shared_attention_promotion.py gates COMPOSITE_ADMISSION_SCHEMA, which is produced and consumed live | tests: delete test/unit/test_shared_attention_evidence_gate.py with it
- **`extra/qk/q4k_wmma_tile_lowering.py`** -- The q4k int8 WMMA-tiled campaign is closed: correct on all four Qwen3-14B role shapes, speed-refuted at 140 tok/s against a 364.5 tok/s direct-packed baseline, dispatch already removed in 45cfc399c. (gross -1138, +0, net -1138) | evidence: verdict recorded in route_manifest.py's note for prefill_q4k_int8_wmma_tiled_research; 45cfc399c (2026-07-21) removed the dispatch: 'superseded by the shipped scheduler-native packed-WMMA route'; last substantive change 05b67146a (2026-07-14); 383c71c72 (2026-07-25) was a path move only; its gate scanned prefill_routes.py for a q8_mode == 'wmma_tiled' branch that no longer exists, so it could never pass again; the strict-xfail wording 'WIP research; not yet PASS' was stale; 31 failures under -k 'wmma or q4k or mmq or prefill' before and after, byte-identical, all pre-existing | tests: 

## Promotion candidates

### ready

None.

### blocked

None.

### not justified

- `extra/qk/kernel_lds.py` -> `tinygrad/codegen/opt/kernel_lds.py`: There is no duplicated knowledge to delete. extra/qk/kernel_lds.py imports 20+ primitives from the core module and only adds new types on top; the sole literal duplicate is a 3-line private _window helper. | invariant: cooperative LDS ownership math (already owned by tinygrad/codegen/opt/kernel_lds.py) | moved 0, deleted 0 | missing evidence: none; the answer is negative

## Deletion candidates with recovery information

| path | class | former purpose | last campaign | replacement | commit | recovery | loc |
|---|---|---|---|---|---|---|---|
| `extra/qk/p2_probe_1.py` | delete_ready | Bisect where REDUCE-preserving fusion breaks in attention by walking max -> sum -> broadcast-subtract -> exp-sum -> softmax -> softmax@v under DEV=AMD TC_OPT=2. | flash-prefill Piece 2-A (2026-07-21) | docs/flash-prefill-piece2-probe-20260721.md | `bece3963e9a7` | git show ad65bd05e951f6e460d167c207fdf3e97faf5c76 -- extra/qk/p2_probe_1.py ... p2_probe_6.py | 44 |
| `extra/qk/shared_attention_evidence_gate.py` | delete_after_verdict_capture | Validate a shared-attention evidence bundle before admitting it as promotion evidence. | shared-attention evidence pipeline | extra/qk/shared_attention_promotion.py (the gate that acts on a schema with a real producer) | `bece3963e9a7` | git log --diff-filter=A -- extra/qk/shared_attention_evidence_gate.py | 131 |
| `extra/qk/q4k_wmma_tile_lowering.py` | delete_after_verdict_capture | Route Q4_K prefill matmuls through RDNA3 v_wmma_i32_16x16x16_iu8 int8 tensor-core tiles. | q4k int8 WMMA-tiled prefill (2026-07-05 to 2026-07-14) | docs/q4k-int8-wmma-tiled-campaign-retirement-20260726.md | `bece3963e9a7` | git show 05b67146a -- <path> | 1138 |

## Workflow inventory

| workflow_id | domain | default path | entry points | phases |
|---|---|---|---|---|
| model-load-admission | route_authority | True | `tinygrad/llm/model.py` | discover-device -> profile-model -> install-adapters -> select-policy |
| decode-route-selection | attention_decode | True | `tinygrad/llm/decode_routes.py` | select-route -> build-spec -> emit-kernel -> execute |
| prefill-route-selection | attention_prefill | True | `tinygrad/llm/prefill_routes.py` | select-route -> build-spec -> compile -> guard -> execute |
| quant-kernel-lowering | quant_mmq | True | `tinygrad/llm/qk_primitives.py` | parse-opt -> build-spec -> emit-kernel |
| mmq-atom-search | quant_mmq | False | `extra/qk/mmq_q4k_q8_atom.py` | build-candidate -> compile -> measure -> compare-reference -> record |
| packed-wmma-prefill-promotion | search_promotion | False | `extra/qk/prefill/packed_wmma_prefill_promotion_gate.py` | load-evidence -> check-shape-guards -> verdict -> emit-record |
| prefill-softmax-fuse-promotion | search_promotion | False | `extra/qk/prefill/prefill_softmax_reduce_fuse_promotion_gate.py` | load-evidence -> check-shape-guards -> verdict -> emit-record |
| prefill-causal-tile-skip-promotion | search_promotion | False | `extra/qk/prefill/prefill_causal_tile_skip_promotion_gate.py` | load-evidence -> check-shape-guards -> verdict -> emit-record |
| whole-model-throughput-measurement | measurement | False | `extra/qk/bench.py` | setup -> warmup -> measure -> report |
| shared-attention-evidence | evidence | False | `extra/qk/generate_shared_attention_captures.py`, `extra/qk/benchmark_shared_attention.py` | capture -> aggregate -> admit -> report |

## Repeated workflow phases

- `build-spec`: ['decode-route-selection', 'prefill-route-selection', 'quant-kernel-lowering']
- `check-shape-guards`: ['packed-wmma-prefill-promotion', 'prefill-causal-tile-skip-promotion', 'prefill-softmax-fuse-promotion']
- `compile`: ['mmq-atom-search', 'prefill-route-selection']
- `emit-kernel`: ['decode-route-selection', 'quant-kernel-lowering']
- `emit-record`: ['packed-wmma-prefill-promotion', 'prefill-causal-tile-skip-promotion', 'prefill-softmax-fuse-promotion']
- `execute`: ['decode-route-selection', 'prefill-route-selection']
- `load-evidence`: ['packed-wmma-prefill-promotion', 'prefill-causal-tile-skip-promotion', 'prefill-softmax-fuse-promotion']
- `measure`: ['mmq-atom-search', 'whole-model-throughput-measurement']
- `report`: ['shared-attention-evidence', 'whole-model-throughput-measurement']
- `select-route`: ['decode-route-selection', 'prefill-route-selection']
- `verdict`: ['packed-wmma-prefill-promotion', 'prefill-causal-tile-skip-promotion', 'prefill-softmax-fuse-promotion']

## Promotion budget

- Budgeted (`tinygrad`, `bench`, `structure`): **34087 / 35000** -- headroom 913
- Against the standing 30000 target: **4087 over**. sz.py:13 records 35000 as temporary headroom; the standing target is 30000
- Default-path LOC currently sitting unbudgeted in `extra/`: **8856**
- Net budget cost of pending promotions (moved minus deleted): 0
- Declarative LOC a data-file conversion would remove from the budget entirely: 0

Promotion converts unbudgeted `extra/` LOC into budgeted core LOC. A promotion that moves more than it deletes
spends headroom that nothing gives back, and `sz.py` counts only `.py`/`.js` -- a table moved to a data file
costs zero.

## LOC impact

- **Realized** (actions A1, A10, A14, A15, A2, A3, A4, A5, A6, A7, A8, A9): gross -1388, +35, net **-1353**
- Still proposed: gross -0, +0, net -0
- LOC merely moved between directories (NOT a reduction): 60

## Recommended sequence

1. A2: delete the six p2_probe_* files (verdict already recorded in docs/flash-prefill-piece2-probe-20260721.md).
2. A3: capture the shared_attention_evidence_gate verdict in one line, then delete the gate and its test.
3. A1: collapse parse_amdgpu_metadata onto extra/qk/mmq_compile_evidence.py and repoint three call sites.
4. A4: land the authority correction (manifest-only) so the duplicate-authority check stays meaningful.
5. A5: extract promotion_gate_common.py for the three sibling gates only, leaving pure_register_evaluation_gate alone.
6. Answer A8 (DEV=AMD:ISA retention) and A9 (WMMA-tiled route revival) before touching either block; together they govern about 4.3K of the 17.7K LOC in scope.
7. Answer A6 and A10, then re-run the audit and report net authored LOC reduction.

## Files requiring human classification

- extra/qk/mmq_q4k_q8_atom.py -- which kernel stages still teach something
- extra/qk/kernel_lds.py clusters C and E -- open question or superseded
- the DEV=AMD:ISA surface -- live research or completed experiment
- the q4k int8 WMMA-tiled gate family -- revive or retire
- extra/qk/quant_specs.py -- unwired API or superseded

## Investigation backlog

- `extra/qk/mmq_q4k_q8_atom.py`: amd_warp_batched_atom_source_hash, amd_dot4_batched_atom_source_hash and amd_dot4x4_batched_atom_source_hash have no importer anywhere -- dead public API in a default-path file. Deleting them only pays if the kernel builders behind them go too, which is the same question as which atom stages still teach something.
- `extra/qk/mmq_q4k_q8_atom.py`: 835 LOC hold ~10 progressively elaborated kernel stages (naive, warp, dot4, dot4x4, ds4 variants, coop-tile). Only the DS4 coop-tile stage is a confirmed evidence-capture target. Are the earlier stages still teaching anything, or is their verdict recorded elsewhere?
- `extra/qk/decode/decode_hd_sweep_numerics.py`: No recorded verdict found for Hd in {64,192,256}. Open sweep or completed one-off?
- `extra/qk/benchmark_split_shared_attention.py`: The split-versus-fused-recompute question has no recorded closing verdict.
- `extra/qk/pure_search_guard.py`: Its docstring claims model.py calls assert_pure_machine_search() at Transformer init, but model.py calls only install_memory_adaptive_model_adapters and automatic_promoted_prefill_graph_policy. Is the purity guard meant to be live-enforced, or is it a test-only contract?
- `extra/qk/bubblebeam_futuresight.py`: Shares a name with the BUBBLEBEAM_FUTURESIGHT env var but never reads os.environ. Confirm the naming collision is harmless or rename one of them.
- `extra/audit/pure_machine_search_default_path_census.py`: Its overlay is stale: route ids decode_flash_live_split_g4_kvboth and packed_wmma_prefill_generated exist in route_manifest but have no census row (the overlay still carries decode_flash_live_split_g4_8b_kvboth and prefill_wmma_lds_dbuf_generated), so the census reports PMS_R0_BLOCKED_ROUTE_ATTRIBUTION_MISSING. This is overlay drift, not an impure default kernel.

## Hard errors

- **default_path_dead_status**: extra/qk/mmq_ds4_probe_contract.py is default_path but status is 'refuted'

## Warnings

- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/amd_isa_proof.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/amd_resource_artifact.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/amd_warp_reduce.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/attention_harness_common.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/cooperative_stage_lanemap.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/decode/current_decode_execution_adapter.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/decode/flash_decode_attention_executor.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/decode/flash_decode_attention_spec.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/flash_common.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/flash_kernels.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/gemv_g2_lanemap.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/gemv_g3_codegen_lowering.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/kernel_vocabulary.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/kv_load.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/lane_partition_reduce.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/layout.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/live_split_geometry.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/memory_adaptive_allocation_observer.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/memory_adaptive_policy.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/memory_adaptive_runtime_collector.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/mmq_atom_boundary.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/mmq_compile_evidence.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/mmq_ds4_probe_contract.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/mmq_epoch_manifest_export.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/mmq_lifecycle.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/mmq_logical_vocabulary.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/mmq_q4k_q8_atom.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/mmq_q4k_q8_reference.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/model_profiles.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/operand_attribution.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/packed_wmma_compile_gate.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/candidate_payloads.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/current_prefill_execution_adapter.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/executable_artifact_preparation.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/execution_bridge_contracts.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/flash_prefill_attention_spec.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/guarded_execution.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/host_safety_canary.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/isolated_guarded_executor.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/operand_path_execution_worker.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/packed_wmma_correctness_canary.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/packed_wmma_prefill_candidates.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/prefill_graph_gemm_route.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/pure_register_compile_capture.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/pure_register_evaluation_gate.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/q4k_prefill_route_spec.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/prefill/q6k_prefill_route_spec.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/q4k_tile_loader.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/q6k_route_spec.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/quant/q4_k_gemv_primitive.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/quant/q6_k_gemv_primitive.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/route_manifest.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/runtime_specs.py is on the default production path from extra/ with no promotion or retention decision
- **extra_on_default_path** (production behavior should live with its domain owner): extra/qk/warp_reduce_lowering.py is on the default production path from extra/ with no promotion or retention decision
- **high_fan_in** (a widely-imported module is a de-facto authority): extra/qk/layout.py is imported by 21 modules
- **high_fan_in** (a widely-imported module is a de-facto authority): extra/qk/route_manifest.py is imported by 20 modules
- **high_fan_out** (prefer deep modules with small interfaces): extra/qk/decode/current_decode_execution_adapter.py imports 17 internal modules
- **high_fan_out** (prefer deep modules with small interfaces): extra/qk/prefill/current_prefill_execution_adapter.py imports 19 internal modules
- **large_file** (minimize what a reader must hold in their head): extra/qk/mmq_q4k_q8_atom.py is 832 LOC (threshold 400); responsibilities declared: 4
- **large_file** (minimize what a reader must hold in their head): extra/qk/prefill/prefill_whole_synced.py is 423 LOC (threshold 400); responsibilities declared: 5
- **large_file** (minimize what a reader must hold in their head): extra/qk/route_manifest.py is 415 LOC (threshold 400); responsibilities declared: 7
- **large_file** (minimize what a reader must hold in their head): extra/qk/runtime_specs.py is 758 LOC (threshold 400); responsibilities declared: 5
- **large_file** (minimize what a reader must hold in their head): extra/qk/shared_attention_capture.py is 463 LOC (threshold 400); responsibilities declared: 4
- **multiple_authority_keys** (centralize authoritative knowledge under one owner): extra/qk/clock_pin.py declares 4 authority keys: ['clock_pin.PIN_PEAK_CMD', 'clock_pin.SET_AUTO_CMD', 'clock_pin.ROCM_SMI_PIN_CMD', 'clock_pin.RESET_PERF_DETERMINISM']
- **multiple_authority_keys** (centralize authoritative knowledge under one owner): extra/qk/memory_adaptive_policy.py declares 4 authority keys: ['tinygrad.memory_adaptive_policy.v1', 'tinygrad.memory_adaptive_policy_cache.v1', 'tinygrad.accelerated_candidate.production_eligibility.v1', 'tinygrad.accelerated_candidate.production_eligibility_requirement.v1']
- **multiple_authority_keys** (centralize authoritative knowledge under one owner): extra/qk/mmq_atom_boundary.py declares 4 authority keys: ['PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID', 'Prefill14BHybridMMQAtomSpec', 'Prefill14BHybridMMQAtomDescriptor', 'milestone_evidence(M1..M7)']
- **multiple_authority_keys** (centralize authoritative knowledge under one owner): extra/qk/mmq_logical_vocabulary.py declares 4 authority keys: ['mmq-logical-vocabulary/1', 'LogicalMMQDescriptor', 'MMQCandidate', 'packed_ds4_geometry canonical grammar (256/32/4 Q4_K, 32/128/4 Q8_1)']
- **multiple_authority_keys** (centralize authoritative knowledge under one owner): extra/qk/mmq_q4k_q8_atom.py declares 12 authority keys: ['BACKEND_ATOM_ID', 'AMD_BACKEND_ATOM_ID', 'AMD_WARP_BACKEND_ATOM_ID', 'AMD_WARP_BATCHED_BACKEND_ATOM_ID', 'AMD_DOT4_BATCHED_BACKEND_ATOM_ID', 'AMD_DOT4X4_BATCHED_BACKEND_ATOM_ID', 'AMD_STAGED_DS4_BACKEND_ATOM_ID', 'AMD_DS4_WARP_BACKEND_ATOM_ID', 'AMD_DS4_DOT4X4_BACKEND_ATOM_ID', 'AMD_DS4_LDS_SKELETON_BACKEND_ATOM_ID', 'AMD_DS4_COOP_TILE_BACKEND_ATOM_ID', 'MMQ_WRITEBACK_MODES']
- **multiple_authority_keys** (centralize authoritative knowledge under one owner): extra/qk/mmq_q4k_q8_reference.py declares 5 authority keys: ['Q8_1_MMQ_DS4_LAYOUT', 'Q8_1_ROW_MAJOR_LAYOUT', 'Q4KQ81MMQTileSpec', 'Q81MMQDS4ActivationSpec', 'MMQOutputTileSpec']
- **multiple_authority_keys** (centralize authoritative knowledge under one owner): extra/qk/prefill/execution_bridge_contracts.py declares 14 authority keys: ['execution_bridge.request.v1', 'execution_bridge.result.v1', 'execution_bridge.transport_plan.v1', 'execution_bridge.guard_protocol.v1', 'execution_bridge.timing_protocol.v1', 'execution_bridge.correctness_protocol.v1', 'execution_bridge.dispatch_evidence.v1', 'execution_bridge.safety_admission.v1', 'execution_bridge.typed_error.v1', 'DISPATCH_STATES', 'RESEARCH_VERDICTS', 'SHIPPING_DECISIONS', 'OPERAND_STRATEGIES', 'RESULT_STATUSES']
- **multiple_authority_keys** (centralize authoritative knowledge under one owner): extra/qk/q4k_tile_loader.py declares 4 authority keys: ['Q4K_D_OFFSET', 'Q4K_DMIN_OFFSET', 'Q4K_SCALE_MIN_OFFSET', 'Q4K_QS_OFFSET byte-offset layout']
- **multiple_authority_keys** (centralize authoritative knowledge under one owner): extra/qk/route_manifest.py declares 6 authority keys: ['extra.qk.route_manifest.ROUTES', 'extra.qk.route_manifest.ROUTE_PROVENANCE', 'extra.qk.route_manifest.PURITY_STATUS_VOCAB', 'extra.qk.route_manifest.REFUTED', 'extra.qk.route_manifest.DEFERRED_CAPABILITIES', 'extra.qk.route_manifest.derive_purity_status']
- **multiple_authority_keys** (centralize authoritative knowledge under one owner): extra/qk/runtime_specs.py declares 5 authority keys: ['extra.qk.runtime_specs.FULL_KERNEL_CANDIDATE_SCHEMA', 'extra.qk.runtime_specs.FULL_KERNEL_CANDIDATE_SET_SCHEMA', 'extra.qk.runtime_specs.GFX1100_SINGLE_BUFFER_CAPABILITY (+ sibling *_CAPABILITY constants)', 'extra.qk.runtime_specs.PROVENANCE (candidate-level provenance vocabulary)', 'extra.qk.runtime_specs._validate_full_kernel_payload']
- **multiple_authority_keys** (centralize authoritative knowledge under one owner): extra/qk/shared_attention_capture.py declares 5 authority keys: ['tinygrad.shared_attention_compiler_capture.v2', 'tinygrad.shared_attention_compiler_capture.acc_slice_v3', 'tinygrad.shared_attention_compiler_capture.phase_v4', 'tinygrad.shared_attention_phase_plan.v1', 'tinygrad.shared_attention_acc_slice_pass.v1']
- **multiple_authority_keys** (centralize authoritative knowledge under one owner): extra/qk/shared_attention_evidence.py declares 4 authority keys: ['tinygrad.shared_attention_proof.v2', 'tinygrad.shared_attention_proof.acc_slice_v3', 'tinygrad.shared_attention_proof.phase_v4', 'tinygrad.shared_attention_evidence.v1']
- **production_seam_to_research** (production must not depend on research surfaces): lazy seam tinygrad/llm/route_ops.py declares a wrapper onto extra/qk/mmq_ds4_logical_emitter.py (role=research); the wrapper is not evidence that the default path calls it
- **repeated_workflow_phase** (modularize execution without scattering authority): phase 'build-spec' is re-implemented by 3 workflows: ['decode-route-selection', 'prefill-route-selection', 'quant-kernel-lowering']
- **repeated_workflow_phase** (modularize execution without scattering authority): phase 'check-shape-guards' is re-implemented by 3 workflows: ['packed-wmma-prefill-promotion', 'prefill-causal-tile-skip-promotion', 'prefill-softmax-fuse-promotion']
- **repeated_workflow_phase** (modularize execution without scattering authority): phase 'emit-record' is re-implemented by 3 workflows: ['packed-wmma-prefill-promotion', 'prefill-causal-tile-skip-promotion', 'prefill-softmax-fuse-promotion']
- **repeated_workflow_phase** (modularize execution without scattering authority): phase 'load-evidence' is re-implemented by 3 workflows: ['packed-wmma-prefill-promotion', 'prefill-causal-tile-skip-promotion', 'prefill-softmax-fuse-promotion']
- **repeated_workflow_phase** (modularize execution without scattering authority): phase 'verdict' is re-implemented by 3 workflows: ['packed-wmma-prefill-promotion', 'prefill-causal-tile-skip-promotion', 'prefill-softmax-fuse-promotion']
- **role_location_conflict** (organize around domain meaning, not accident of location): extra/qk/decode/decode_codegen_identity_check.py declares role=test but does not live in test/ or name itself a test

## Coverage limitations

- Phase 1 covers extra/qk only: 123 authored files, 17675 token-bearing LOC. Files outside extra/qk appear as dependency, consumer, or reference evidence and are deliberately not classified.
- Out of scope by instruction and not attempted: a full tinygrad/llm census, a scheduler/codegen census, an AMD renderer/runtime census, and a full bench/ and test/ census.
- allowed_dependency_domains is pinned to the dependency domains observed at the audited commit. It is a drift detector for future edits, not an independently designed layering.
- Default-path membership is traced through tinygrad/ call sites and the lazy _attr seam. A declared seam wrapper is recorded as a warning, never as proof that the default path calls it.
- Runtime behavior was not executed for this audit; no kernel was run and no default was changed.
