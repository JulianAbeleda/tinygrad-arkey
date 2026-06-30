# Whole-Repo Principles Cleanup — Inventory

HEAD `df8433b0f` · 2828 tracked · 2404 project rows · 345 vendor files in 11 dirs

## By recommendation

- **ARCHIVE_PROVENANCE**: 1828
- **KEEP_DOC_AUTHORITY**: 276
- **KEEP_TEST**: 128
- **KEEP_LIVE_TOOLING**: 75
- **KEEP_LIBRARY_HELPER**: 39
- **KEEP_CORE**: 31
- **DELETE**: 27
- **IGNORE_EXTERNAL_VENDOR**: 11

## By subsystem

| subsystem | ARCHIVE_PROVENANCE | DELETE | IGNORE_EXTERNAL_VENDOR | KEEP_CORE | KEEP_DOC_AUTHORITY | KEEP_LIBRARY_HELPER | KEEP_LIVE_TOOLING | KEEP_TEST |
|---|---|---|---|---|---|---|---|---|
| audit_tooling |  |  |  |  |  |  | 5 |  |
| bench_artifact | 940 |  |  |  |  |  |  |  |
| core_runtime |  |  |  | 5 |  |  |  |  |
| docs | 746 |  |  |  | 243 |  |  |  |
| evaluator_search_ledger |  |  |  |  |  |  | 27 |  |
| extra_qk_tooling | 132 | 27 |  |  |  | 39 | 43 |  |
| root_config |  |  |  | 26 |  |  |  |  |
| structure | 10 |  |  |  | 33 |  |  |  |
| test |  |  |  |  |  |  |  | 128 |
| vendor |  |  | 11 |  |  |  |  |  |

## DELETE candidates (27) — proof: no importer/doc/test/ledger ref

- `extra/amd_isa_grid_gate.py`
- `extra/amd_isa_inc4_gate.py`
- `extra/amd_isa_phase_h_inmodel_gate.py`
- `extra/amd_isa_phase_j_gate.py`
- `extra/amd_isa_phase_k_gate.py`
- `extra/amd_isa_phase_l_gate.py`
- `extra/amd_isa_phase_m_gate.py`
- `extra/amd_isa_phase_n1a_gate.py`
- `extra/amd_isa_phase_n1b_gate.py`
- `extra/amd_isa_phase_n3f0_ctx_confirmation.py`
- `extra/amd_isa_q6k_lmhead_token_gate.py`
- `extra/qk_block_tile_one_case.py`
- `extra/qk_decode_attention_cross_lane_reduce_store_gate.py`
- `extra/qk_decode_attention_generated_pv_kernel_audit.py`
- `extra/qk_decode_attention_split_xlane_output.py`
- `extra/qk_decode_attention_xlane_recurrence_matrix.py`
- `extra/qk_decode_attention_xlane_reducer_matrix.py`
- `extra/qk_decode_physical_tile_all_primitives_gate.py`
- `extra/qk_decode_physical_tile_pall_lifecycle_scaling_probe.py`
- `extra/qk_decode_physical_tile_score_broadcast_chain_gate.py`
- `extra/qk_decode_physical_tile_score_broadcast_direct_gate.py`
- `extra/qk_decode_primitive_candidate_template.py`
- `extra/qk_decode_primitive_detector.py`
- `extra/qk_decode_primitive_gap_gate.py`
- `extra/qk_decode_score_broadcast_control_matrix_gate.py`
- `extra/qk_new_profile_search.py`
- `extra/qk_prefill_pipe_role_profile.py`
