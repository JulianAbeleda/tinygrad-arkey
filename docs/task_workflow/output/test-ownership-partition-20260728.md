# Test Ownership Partition

This is the conservative branch-tier partition produced by the R7 evidence audit. Mixed-owner, TinyGPU, and GPU-lock
tests remain unresolved and are intentionally omitted.

## Master Retain (14)

Production implementation regressions:

- `test/unit/test_flash_prefill_spec.py`
- `test/unit/test_codegen_recurrence_unroll.py`
- `test/unit/test_reg_store_devec.py`
- `test/unit/test_fdot2_lowering.py`
- `test/unit/test_codegen_list_scheduler.py`
- `test/unit/test_warp_reduce_lowering.py`
- `test/unit/test_codegen_opt_parser.py`

Production policy and audit authority:

- `test/unit/test_measurement_authority.py`
- `test/unit/test_size_accounting.py`
- `test/unit/test_tinygrad_boundary.py`
- `test/unit/test_codebase_organization_audit.py`
- `test/unit/test_flash_variant_fingerprint.py`
- `test/unit/test_lowering_baseline.py`
- `test/unit/test_lowering_fingerprint.py`

## Dev Only

These 25 debug, qualification, extra-LLM, diagnostic, and eGPU tests remain on `dev` and `exp`:

```text
test/test_guarded_execution.py
test/unit/test_amd_isa_extraction_fixtures.py
test/unit/test_amd_resource_artifact.py
test/unit/test_amd_resource_artifact_intervals_20260712.py
test/unit/test_bench_entrypoint.py
test/unit/test_clock_pin.py
test/unit/test_decode_resource_capture.py
test/unit/test_executable_artifact_preparation.py
test/unit/test_execution_bridge_contracts.py
test/unit/test_host_safety_canary_20260713.py
test/unit/test_isolated_guarded_executor.py
test/unit/test_operand_attribution_20260714.py
test/unit/test_operand_path_execution_worker.py
test/unit/test_prefill_harness.py
test/unit/test_prefill_whole_synced.py
test/unit/test_pure_register_compile_capture.py
test/unit/test_pure_register_evaluation_gate.py
test/unit/test_pure_register_final_program_adapter_20260712.py
test/unit/test_qk_decode_duration.py
test/unit/test_shared_prefill_measurement.py
test/unit/test_llama_bench_artifacts.py
test/unit/test_llm_decode_correctness.py
test/unit/test_analyze_faults.py
test/unit/test_egpu_minimal_compute.py
test/unit/test_egpu_qualify.py
```

## Exp Only

These 26 MMQ, packed-WMMA, candidate-context, shared-attention, and research tests remain on `exp`:

```text
test/unit/test_candidate_context_propagation.py
test/unit/test_kernel_candidate_context.py
test/unit/test_kernel_lds_mapping.py
test/unit/test_kernel_naming.py
test/unit/test_mmq_atom_boundary.py
test/unit/test_mmq_ds4_logical_emitter.py
test/unit/test_mmq_epoch_manifest_export.py
test/unit/test_mmq_lifecycle.py
test/unit/test_mmq_logical_vocabulary.py
test/unit/test_mmq_q4k_q8_atom.py
test/unit/test_mmq_q4k_q8_reference.py
test/unit/test_packed_weight.py
test/unit/test_packed_wmma_compile_gate.py
test/unit/test_packed_wmma_correctness_canary.py
test/unit/test_precontract_int8_lds_contract.py
test/unit/test_pure_pipe_graph_abi.py
test/unit/test_q4_q4_owner_comparison.py
test/unit/test_q4k_q8_mmq_prefill_spec.py
test/unit/test_shared_attention_compiler_capture.py
test/unit/test_shared_attention_evidence.py
test/unit/test_shared_attention_promotion.py
test/unit/test_shared_attention_replay_admission.py
test/unit/test_shared_attention_synchronization_capture.py
test/unit/test_shared_prefill_policy.py
test/unit/test_stage1_wmma_compile_evidence.py
test/unit/test_strided_lds_record_layout.py
```

The exact paths and evidence packet are retained in the task-workflow audit mailbox; no mixed-owner test is moved by
this partition.
