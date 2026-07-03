# Tinygrad Boundary Deep Audit - 20260703

**Verdict:** `PASS_WITH_COMPAT_SHIMS`

## Counts

- **boltbeam_owned_compat_shim**: 4
- **boltbeam_owned_not_yet_ported**: 248
- **provenance_only**: 1804
- **stay_runner_adapter**: 25
- **stay_runtime**: 842
- **test_split_needed**: 3

## Physically Ported To BoltBeam

- `extra/qk_decode_role_profile.py` -> `boltbeam/profile/decode_roles.py` (present)
- `extra/qk_descriptor_policy.py` -> `boltbeam/policy/descriptor.py` (present)
- `extra/qk_artifact_cache_inventory.py` -> `boltbeam/artifacts/cache_inventory.py` (present)
- `extra/qk_decode_primitive_candidate_template.py` -> `boltbeam/search/primitive_template.py` (present)
- `extra/qk_policy_consistency_check.py` -> `boltbeam/report/policy_consistency.py` (present)
- `extra/qk_route_manifest.py` -> `boltbeam/policy/route_manifest.py` (present)
- `extra/qk_search_spec.py` -> `boltbeam/search/spec.py` (present)
- `extra/qk_search_util.py` -> `boltbeam/search/util.py` (present)
- `extra/qk_semantic_candidate.py` -> `boltbeam/search/semantic_candidate.py` (present)

## Contract-Decoupled Runner Adapters

| path | reason | decoupling_status |
|---|---|---|
| extra/amd_isa_g3_weight_promotion_gate.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/amd_isa_reg_accum_lds_reclaim_audit.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/amd_isa_weight_path_route_attribution.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/q8_ffn_fast_artifact_probe.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/q8_ffn_hcq_artifact.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_amdgpu_isa_primitive_audit.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_artifact_cache.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_candidate_template_gen.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_decode_attention_generated_pv_kernel_audit.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_decode_attention_generated_wall_audit.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_decode_search_gate.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_decode_search_runner.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_harness_contract.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_lanemap_template_audit.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_large_model_decode_route_gap_audit.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_large_shape_knob_reachability_audit.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_large_shape_topology_space_audit.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_lifecycle_search_loop.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_pathology_artifact_g5.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_prefill_pipe_role_profile.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_split_kv_economics_audit.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_system_fusion_sf0_audit.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_tg_p14_combine_reopen.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/qk_tg_p8_geometry_search.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |
| extra/tinygrad_runtime_boundary_audit.py | ownership resolved as tinygrad runner adapter; tinygrad keeps GPU/runtime execution and emits JSON evidence for BoltBeam policy/eval | contract_resolved |

## BoltBeam-Owned Still In Tinygrad

| path | reason | tiny_principle_action |
|---|---|---|
| bench/pure-machine-search-default-path-census/default_route_table.json | bench artifact is current model/search/eval/roofline evidence: census, search | port_or_summarize_then_remove |
| bench/pure-machine-search-default-path-census/fallback_table.json | bench artifact is current model/search/eval/roofline evidence: census, search | port_or_summarize_then_remove |
| bench/pure-machine-search-default-path-census/latest.json | bench artifact is current model/search/eval/roofline evidence: census, search | port_or_summarize_then_remove |
| bench/pure-machine-search-default-path-census/summary.md | bench artifact is current model/search/eval/roofline evidence: census, search | port_or_summarize_then_remove |
| bench/qk-bandwidth-roofline-20260613/README.md | bench artifact is current model/search/eval/roofline evidence: roofline | port_or_summarize_then_remove |
| bench/qk-decode-eval/candidates.json | bench artifact is current model/search/eval/roofline evidence: candidate | port_or_summarize_then_remove |
| bench/qk-lifecycle-search/candidates.json | bench artifact is current model/search/eval/roofline evidence: candidate, search | port_or_summarize_then_remove |
| bench/qk-lifecycle-search/evaluator_contract.json | bench artifact is current model/search/eval/roofline evidence: contract, evaluator, search | port_or_summarize_then_remove |
| bench/qk-lifecycle-search/generated_candidates.json | bench artifact is current model/search/eval/roofline evidence: candidate, search | port_or_summarize_then_remove |
| bench/qk-lifecycle-search/policy_exports.json | bench artifact is current model/search/eval/roofline evidence: policy, search | port_or_summarize_then_remove |
| bench/qk-lifecycle-search/refutations.json | bench artifact is current model/search/eval/roofline evidence: refut, search | port_or_summarize_then_remove |
| bench/qk-lifecycle-search/runner_bindings.json | bench artifact is current model/search/eval/roofline evidence: search | port_or_summarize_then_remove |
| bench/qk-lifecycle-search/search_candidates.json | bench artifact is current model/search/eval/roofline evidence: candidate, search | port_or_summarize_then_remove |
| bench/qk-lifecycle-search/search_policy.json | bench artifact is current model/search/eval/roofline evidence: policy, search | port_or_summarize_then_remove |
| bench/qk-lifecycle-search/search_schema.json | bench artifact is current model/search/eval/roofline evidence: search | port_or_summarize_then_remove |
| bench/qk-lifecycle-search/summary.md | bench artifact is current model/search/eval/roofline evidence: search | port_or_summarize_then_remove |
| bench/qk-lifecycle-search/template_schema.json | bench artifact is current model/search/eval/roofline evidence: search | port_or_summarize_then_remove |
| bench/qk-lifecycle-search/templates.json | bench artifact is current model/search/eval/roofline evidence: search | port_or_summarize_then_remove |
| bench/qk-prefill-theoretical-ceiling/roofline_floor.json | bench artifact is current model/search/eval/roofline evidence: roofline | port_or_summarize_then_remove |
| bench/qk-pure-machine-search-gap/latest.json | bench artifact is current model/search/eval/roofline evidence: search | port_or_summarize_then_remove |
| bench/tg-p14-amd-recovery-and-pure-attention-landing/practical_roofline_audit.json | bench artifact is current model/search/eval/roofline evidence: audit, roofline | port_or_summarize_then_remove |
| bench/tg-p14-amd-recovery-and-pure-attention-landing/practical_roofline_audit.md | bench artifact is current model/search/eval/roofline evidence: audit, roofline | port_or_summarize_then_remove |
| docs/amd-isa-active-surface-principles-audit-20260629.md | dated audit/search artifact matches BoltBeam concepts: audit | port_or_summarize_then_remove |
| docs/amd-isa-backend-e2e-roadmap-20260629.md | dated audit/search artifact matches BoltBeam concepts: roadmap | port_or_summarize_then_remove |
| docs/archive/8b-decode-remaining-gap-research-scope-20260618.md | dated audit/search artifact matches BoltBeam concepts: search | port_or_summarize_then_remove |
| docs/archive/8b-decode-research-banks-roadmap-20260618.md | dated audit/search artifact matches BoltBeam concepts: roadmap, search | port_or_summarize_then_remove |
| docs/archive/amd-broad-backend-bb5a10-p8-global-direct-candidate-decision-result-20260620.md | dated audit/search artifact matches BoltBeam concepts: candidate | port_or_summarize_then_remove |
| docs/archive/amd-broad-backend-bb5a10-p8-tta3-macro-candidate-result-20260620.md | dated audit/search artifact matches BoltBeam concepts: candidate | port_or_summarize_then_remove |
| docs/archive/amd-broad-backend-bb5a10-ptm3-native-candidate-scope-20260620.md | dated audit/search artifact matches BoltBeam concepts: candidate | port_or_summarize_then_remove |
| docs/archive/amd-broad-backend-bb5a10-tensile-layout-audit-20260619.md | dated audit/search artifact matches BoltBeam concepts: audit | port_or_summarize_then_remove |
| docs/archive/amd-broad-backend-roadmap-result-20260619.md | dated audit/search artifact matches BoltBeam concepts: roadmap | port_or_summarize_then_remove |
| docs/archive/amd-broad-backend-roadmap-scope-20260619.md | dated audit/search artifact matches BoltBeam concepts: roadmap | port_or_summarize_then_remove |
| docs/archive/amd-decode-bandwidth-roofline.md | dated audit/search artifact matches BoltBeam concepts: roofline | port_or_summarize_then_remove |
| docs/archive/amd-decode-beyond-llama-roadmap.md | dated audit/search artifact matches BoltBeam concepts: roadmap | port_or_summarize_then_remove |
| docs/archive/amd-decode-demotion-search-20260616.md | dated audit/search artifact matches BoltBeam concepts: search | port_or_summarize_then_remove |
| docs/archive/amd-decode-final-report.md | dated audit/search artifact matches BoltBeam concepts: report | port_or_summarize_then_remove |
| docs/archive/amd-decode-flash-threshold-20260616.md | dated audit/search artifact matches BoltBeam concepts: threshold | port_or_summarize_then_remove |
| docs/archive/amd-decode-lossy-quant-search.md | dated audit/search artifact matches BoltBeam concepts: search | port_or_summarize_then_remove |
| docs/archive/amd-decode-memory-access-audit.md | dated audit/search artifact matches BoltBeam concepts: audit | port_or_summarize_then_remove |
| docs/archive/amd-decode-methodology-and-roadmap.md | dated audit/search artifact matches BoltBeam concepts: roadmap | port_or_summarize_then_remove |
| docs/archive/amd-decode-sequential-tax-profile-20260616.md | dated audit/search artifact matches BoltBeam concepts: profile | port_or_summarize_then_remove |
| docs/archive/amd-isa-decode-attention-ceiling-audit-scope-20260629.md | dated audit/search artifact matches BoltBeam concepts: audit | port_or_summarize_then_remove |
| docs/archive/amd-lds-research-consolidation-20260619.md | dated audit/search artifact matches BoltBeam concepts: search | port_or_summarize_then_remove |
| docs/archive/amd-rocm-llamacpp-research.md | dated audit/search artifact matches BoltBeam concepts: search | port_or_summarize_then_remove |
| docs/archive/amd-rocprofiler-r1p1-aqlprofile-replay-result-20260619.md | dated audit/search artifact matches BoltBeam concepts: profile | port_or_summarize_then_remove |
| docs/archive/amd-rocprofiler-r1p2-hcq-replay-result-20260619.md | dated audit/search artifact matches BoltBeam concepts: profile | port_or_summarize_then_remove |
| docs/archive/amd-rocprofiler-r1p2-v2-exporter-result-20260619.md | dated audit/search artifact matches BoltBeam concepts: profile | port_or_summarize_then_remove |
| docs/archive/amd-rocprofiler-r1p2-v2-exporter-scope-20260619.md | dated audit/search artifact matches BoltBeam concepts: profile | port_or_summarize_then_remove |
| docs/archive/amd-rocprofiler-reopen-tracks-scope-result-20260619.md | dated audit/search artifact matches BoltBeam concepts: profile, reopen | port_or_summarize_then_remove |
| docs/archive/amd-rocprofiler-thread-trace-audit-result-20260619.md | dated audit/search artifact matches BoltBeam concepts: audit, profile | port_or_summarize_then_remove |
| docs/archive/amd-scheduler-tooling-t1b-att-aqlprofile-result-20260619.md | dated audit/search artifact matches BoltBeam concepts: profile | port_or_summarize_then_remove |
| docs/archive/attention-tail-after-b5-audit-result-20260622.md | dated audit/search artifact matches BoltBeam concepts: audit | port_or_summarize_then_remove |
| docs/archive/attention-tail-after-b5-audit-scope-20260622.md | dated audit/search artifact matches BoltBeam concepts: audit | port_or_summarize_then_remove |
| docs/archive/bank5-smoothquant-audit-20260618.md | dated audit/search artifact matches BoltBeam concepts: audit | port_or_summarize_then_remove |
| docs/archive/bank6-machine-search-infra-scope-20260618.md | dated audit/search artifact matches BoltBeam concepts: search | port_or_summarize_then_remove |
| docs/archive/beam-hang-premise-audit-20260619.md | dated audit/search artifact matches BoltBeam concepts: audit | port_or_summarize_then_remove |
| docs/archive/candidate-template-generation-v0-result-20260621.md | dated audit/search artifact matches BoltBeam concepts: candidate | port_or_summarize_then_remove |
| docs/archive/canonical-policy-handoff-audit-result-20260621.md | dated audit/search artifact matches BoltBeam concepts: audit, policy | port_or_summarize_then_remove |
| docs/archive/canonical-policy-handoff-audit-scope-20260621.md | dated audit/search artifact matches BoltBeam concepts: audit, policy | port_or_summarize_then_remove |
| docs/archive/cross-shape-generalization-search-targets-scope-20260623.md | dated audit/search artifact matches BoltBeam concepts: search | port_or_summarize_then_remove |
| docs/archive/cross-vendor-isa-primitive-audit-and-search-result-20260623.md | dated audit/search artifact matches BoltBeam concepts: audit, search | port_or_summarize_then_remove |
| docs/archive/cross-vendor-isa-primitive-audit-and-search-scope-20260623.md | dated audit/search artifact matches BoltBeam concepts: audit, search | port_or_summarize_then_remove |
| docs/archive/decode-att-unblock-audit-result-20260620.md | dated audit/search artifact matches BoltBeam concepts: audit | port_or_summarize_then_remove |
| docs/archive/decode-attention-candidate-ab-result-20260620.md | dated audit/search artifact matches BoltBeam concepts: candidate | port_or_summarize_then_remove |
| docs/archive/decode-ctx-slope-audit-result-20260623.md | dated audit/search artifact matches BoltBeam concepts: audit | port_or_summarize_then_remove |
| docs/archive/decode-ctx-slope-audit-scope-20260623.md | dated audit/search artifact matches BoltBeam concepts: audit | port_or_summarize_then_remove |
| docs/archive/decode-ctx-slope-lifecycle-primitive-audit-scope-20260624.md | dated audit/search artifact matches BoltBeam concepts: audit | port_or_summarize_then_remove |
| docs/archive/decode-dnr4-t3-candidate-grid-result-20260620.md | dated audit/search artifact matches BoltBeam concepts: candidate | port_or_summarize_then_remove |
| docs/archive/decode-fused-coop-primitive-roadmap-scope-20260621.md | dated audit/search artifact matches BoltBeam concepts: roadmap | port_or_summarize_then_remove |
| docs/archive/decode-gap-audit-consolidated-20260622.md | dated audit/search artifact matches BoltBeam concepts: audit | port_or_summarize_then_remove |
| docs/archive/decode-machine-search-execution-result-20260623.md | dated audit/search artifact matches BoltBeam concepts: search | port_or_summarize_then_remove |
| docs/archive/decode-machine-search-execution-scope-20260623.md | dated audit/search artifact matches BoltBeam concepts: search | port_or_summarize_then_remove |
| docs/archive/decode-machine-search-readiness-package-result-20260623.md | dated audit/search artifact matches BoltBeam concepts: search | port_or_summarize_then_remove |
| docs/archive/decode-machine-search-readiness-package-scope-20260623.md | dated audit/search artifact matches BoltBeam concepts: search | port_or_summarize_then_remove |
| docs/archive/decode-mmvq-artifact-import-discovery-result-20260619.md | dated audit/search artifact matches BoltBeam concepts: artifact | port_or_summarize_then_remove |
| docs/archive/decode-mmvq-large-project-p0-contract-inventory-result-20260619.md | dated audit/search artifact matches BoltBeam concepts: contract | port_or_summarize_then_remove |
| docs/archive/decode-mode-b-generated-tile-variant-search-scope-20260623.md | dated audit/search artifact matches BoltBeam concepts: search | port_or_summarize_then_remove |
| docs/archive/decode-mode-b-search-result-20260623.md | dated audit/search artifact matches BoltBeam concepts: search | port_or_summarize_then_remove |
| docs/archive/decode-native-renderer-dnr3c7a-resource-ledger-result-20260620.md | dated audit/search artifact matches BoltBeam concepts: ledger | port_or_summarize_then_remove |
| docs/archive/decode-native-renderer-dnr3c9-new-info-ledger-20260620.md | dated audit/search artifact matches BoltBeam concepts: ledger | port_or_summarize_then_remove |
| ... | 168 more rows in JSON | |

## Removal Candidates

| path | recommended_category | risk_if_removed | references_checked |
|---|---|---|---|

## Old Beam / BubbleBeam / FutureSight References

| path | beam_reference_kind | bubblebeam_futuresight_status | tiny_principle_action |
|---|---|---|---|
| bench/amd-decode-flywheel-proof-20260614/loop-live-L2/README.md | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/amd-isa-backend-g3-weight-promotion/latest.json | compat_env_alias | canonical_path | summarize_then_remove |
| bench/amd-isa-backend-g3-weight-promotion/summary.md | compat_env_alias | canonical_path | summarize_then_remove |
| bench/amd-isa-backend-phase-n5/native_tile_residual/latest.json | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/amd-scheduler-tooling-backend/execution.json | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/canonical-benchmarks.json | compat_env_alias | canonical_path | summarize_then_remove |
| bench/qk-14b-remeasure-20260612/README.md | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-active-surface-reduction/docs_index.json | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-ansor-transition-20260612/README.md | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-bandwidth-roofline-20260613/README.md | stale_doc | stale_name | port_or_summarize_then_remove |
| bench/qk-codegen-wmma/inmodel_matmul.json | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-decode-attention-a3-6-tile-score-max/decode-attention-a3-6-tile-score-max-20260625-212749.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-decode-attention-fused-score-state-pv-tile/latest.json | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-decode-attention-generated-pv-kernel-audit/latest.json | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-decode-eval/candidates.json | stale_doc | stale_name | port_or_summarize_then_remove |
| bench/qk-decode-pressure-search-ownership/latest.json | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-flash-prefill-phase5/result.json | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-163748.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-164426.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-165432.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-170200.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-170416.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-170614.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-171635.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-171700.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-172354.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-181922.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-182051.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-183835.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-183900.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-184739.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-190038.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-190103.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/gemv-purity-gate-20260625-190122.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-gemv-purity-gate/latest.json | compat_env_alias | canonical_path | summarize_then_remove |
| bench/qk-packed-tile-research-closeout-20260613/README.md | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-prefill-pipe-promotion/latest.json | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-prefill-pipe-tm2-tn2-hardening/promotion_package.json | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-prefill-pipe-tm2-tn2-hardening/summary.md | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-profile-opener/qwen3_8b_q4_k_m_gfx1100/latest.json | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-pure-machine-search-gap/latest.json | compat_env_alias | canonical_path | port_or_summarize_then_remove |
| bench/qk-pure-search-gap/latest.json | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-repo-principles-cleanup/inventory.json | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-171554.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-181841.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-183810.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-184718.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-190004.json | historical_provenance | blocked_review | summarize_then_remove |
| bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_latest.json | compat_env_alias | canonical_path | summarize_then_remove |
| bench/qk-search-spaces/README.md | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-search-spaces/decode_attention_loop_search_space.json | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-search-spaces/decode_ffn_gemv_gfx1100_v1.json | compat_env_alias | canonical_path | summarize_then_remove |
| bench/qk-semantic-20260612/README.md | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qk-tensile-extraction/codegen_oracle.json | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/qwen-adapter-20260613/training-data-v4_1-compiler/README.md | stale_doc | stale_name | rename_or_summarize_then_remove |
| bench/tg-p2-q4k-g3-policy-driven/latest.json | compat_env_alias | canonical_path | summarize_then_remove |
| bench/tg-p2-q4k-g3-policy-driven/summary.md | compat_env_alias | canonical_path | summarize_then_remove |
| docs/README.md | stale_doc | stale_name | rename_or_summarize_then_remove |
| docs/abstractions4.py | compat_env_alias | canonical_path | keep |
| docs/amd-isa-active-surface-principles-audit-20260629.md | historical_provenance | blocked_review | port_or_summarize_then_remove |
| docs/amd-isa-backend-e2e-roadmap-20260629.md | historical_provenance | blocked_review | port_or_summarize_then_remove |
| docs/amd-isa-backend-phase-h-o-claude-scope-20260629.md | historical_provenance | blocked_review | summarize_then_remove |
| docs/amd-isa-backend-scope-and-enablement-20260628.md | historical_provenance | blocked_review | summarize_then_remove |
| docs/amd-isa-g3-weight-promotion-hardening-scope-20260629.md | historical_provenance | blocked_review | summarize_then_remove |
| docs/amd-isa-q6k-direct-route-full-scope-20260629.md | historical_provenance | blocked_review | summarize_then_remove |
| docs/archive/8b-decode-remaining-gap-research-scope-20260618.md | historical_provenance | blocked_review | port_or_summarize_then_remove |
| docs/archive/amd-broad-backend-bb5a10-ptm2-prefill-primitive-decision-result-20260620.md | historical_provenance | blocked_review | summarize_then_remove |
| docs/archive/amd-broad-backend-bb5a10-ptm3-native-candidate-scope-20260620.md | historical_provenance | blocked_review | port_or_summarize_then_remove |
| docs/archive/amd-broad-backend-roadmap-scope-20260619.md | historical_provenance | blocked_review | port_or_summarize_then_remove |
| docs/archive/amd-decode-ansor-direction.md | historical_provenance | blocked_review | summarize_then_remove |
| docs/archive/amd-decode-batched-tc-result.md | historical_provenance | blocked_review | summarize_then_remove |
| docs/archive/amd-decode-beyond-llama-roadmap.md | historical_provenance | blocked_review | port_or_summarize_then_remove |
| docs/archive/amd-decode-current-verdicts.md | historical_provenance | blocked_review | summarize_then_remove |
| docs/archive/amd-decode-dp4a-vocabulary.md | historical_provenance | blocked_review | summarize_then_remove |
| docs/archive/amd-decode-final-report.md | historical_provenance | blocked_review | port_or_summarize_then_remove |
| docs/archive/amd-decode-fix-plan.md | historical_provenance | blocked_review | summarize_then_remove |
| docs/archive/amd-decode-flywheel-postmortem.md | historical_provenance | blocked_review | summarize_then_remove |
| docs/archive/amd-decode-flywheel-proof-plan.md | historical_provenance | blocked_review | summarize_then_remove |
| docs/archive/amd-decode-hypothesis-statement.md | historical_provenance | blocked_review | summarize_then_remove |
| docs/archive/amd-decode-latency-vocabulary.md | historical_provenance | blocked_review | summarize_then_remove |
| ... | 179 more rows in JSON | | |

## Hardcoded Constants

| path | hardcoded_constants | recommended_category | owner |
|---|---|---|---|
| .claude/commands/pure-search-loop.md | ['8B', 'gfx1100'] | stay_runtime | tinygrad |
| README.md | ['14B', '32B', '7900', '8B', 'gfx1100', 'qwen'] | stay_runtime | tinygrad |
| bench/README.md | ['14B', '32B', '4096', '7900', '8B', 'Q4_K', 'Q6_K', 'gfx1100'] | provenance_only | neither |
| bench/amd-cross-ring-dependency-probe/result.json | ['gfx1100'] | provenance_only | neither |
| bench/amd-decode-flywheel-proof-20260614/README.md | ['8B', 'Q4_K', 'Qwen'] | provenance_only | neither |
| bench/amd-decode-flywheel-proof-20260614/loop-live-L0/README.md | ['4096'] | provenance_only | neither |
| bench/amd-decode-flywheel-proof-20260614/loop-live-L1/README.md | ['4096'] | provenance_only | neither |
| bench/amd-decode-flywheel-proof-20260614/loop-live-L2/README.md | ['4096'] | provenance_only | neither |
| bench/amd-decode-flywheel-proof-20260614/shadow-v0/README.md | ['Q4_K'] | provenance_only | neither |
| bench/amd-decode-flywheel-proof-20260614/shadow-v0/runs/threeway-blk-0-attn-output-weight/README.md | ['Q4_K'] | provenance_only | neither |
| bench/amd-decode-flywheel-proof-20260614/triage-adapter-smoke-v0-rollout/summary.json | ['8B'] | provenance_only | neither |
| bench/amd-decode-flywheel-proof-20260614/triage-adapter-smoke-v0/README.md | ['8B'] | provenance_only | neither |
| bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v0/README.md | ['Q6_K'] | provenance_only | neither |
| bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1-featured/README.md | ['Q6_K'] | provenance_only | neither |
| bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1/README.md | ['Q6_K'] | provenance_only | neither |
| bench/amd-decode-flywheel-proof-20260614/triage-qwen3-8b-base-v0/summary.json | ['8B'] | provenance_only | neither |
| bench/amd-isa-backend-decode-attention-ceiling/latest.json | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-decode-attention-ceiling/loss_stack.json | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-decode-attention-ceiling/math_floor.json | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-decode-attention-ceiling/summary.md | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-g3-vs-owned-weight-parity/latest.json | ['4096', 'Q4_K'] | provenance_only | neither |
| bench/amd-isa-backend-g3-vs-owned-weight-parity/summary.md | ['Q4_K'] | provenance_only | neither |
| bench/amd-isa-backend-g3-weight-promotion/latest.json | ['12288', '4096', 'Q4_K', 'Q6_K'] | provenance_only | neither |
| bench/amd-isa-backend-g3-weight-promotion/route_counts.json | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-g3-weight-promotion/summary.md | ['4096', 'Q4_K'] | provenance_only | neither |
| bench/amd-isa-backend-grid/latest.json | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-inc0/latest.json | ['gfx1100'] | provenance_only | neither |
| bench/amd-isa-backend-inc1/latest.json | ['gfx1100'] | provenance_only | neither |
| bench/amd-isa-backend-inc2/latest.json | ['4096', 'gfx1100'] | provenance_only | neither |
| bench/amd-isa-backend-pc-source-trace/owned_disasm.json | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-phase-c/latest.json | ['Q4_K'] | provenance_only | neither |
| bench/amd-isa-backend-phase-i/latest.json | ['12288', '4096'] | provenance_only | neither |
| bench/amd-isa-backend-phase-m/latest.json | ['12288'] | provenance_only | neither |
| bench/amd-isa-backend-phase-n0/disasm_owned.txt | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-phase-n0/latest.json | ['12288', '4096'] | provenance_only | neither |
| bench/amd-isa-backend-phase-n1a/latest.json | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-phase-n2/latest.json | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-phase-n2/native_trace.json | ['gfx1100'] | provenance_only | neither |
| bench/amd-isa-backend-phase-n2/owned_trace.json | ['gfx1100'] | provenance_only | neither |
| bench/amd-isa-backend-phase-n3/n3f_latest.json | ['151936', '4096'] | provenance_only | neither |
| bench/amd-isa-backend-phase-n4/latest.json | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-phase-n6/latest.json | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-phase-n7/latest.json | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-q6k-direct-correctness/latest.json | ['12288', '4096', 'Q6_K'] | provenance_only | neither |
| bench/amd-isa-backend-q6k-direct-correctness/summary.md | ['12288', '4096'] | provenance_only | neither |
| bench/amd-isa-backend-q6k-direct-route-design/candidate_routes.json | ['Q4_K', 'Q6_K'] | provenance_only | neither |
| bench/amd-isa-backend-q6k-direct-route-design/current_route.json | ['151936', '4096', 'Q4_K'] | provenance_only | neither |
| bench/amd-isa-backend-q6k-direct-route-design/implementation_plan.json | ['12288', '4096', 'Q6_K'] | provenance_only | neither |
| bench/amd-isa-backend-q6k-direct-route-design/latest.json | ['12288', '151936', '4096', 'Q4_K', 'Q6_K'] | provenance_only | neither |
| bench/amd-isa-backend-q6k-direct-route-design/merge_plan.json | ['Q4_K', 'Q6_K'] | provenance_only | neither |
| bench/amd-isa-backend-q6k-direct-route-design/risk_register.json | ['4096', 'Q4_K'] | provenance_only | neither |
| bench/amd-isa-backend-q6k-direct-route-design/summary.md | ['4096', 'Q4_K', 'Q6_K'] | provenance_only | neither |
| bench/amd-isa-backend-q6k-direct-speed/amdahl_vs_measured.json | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-q6k-direct-speed/latest.json | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-q6k-direct-speed/route_counts.json | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-q6k-direct-speed/summary.md | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-q6k-direct-speed/wd_table.json | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-regalloc-accum-lds-reclaim/rl0_latest.json | ['12288'] | provenance_only | neither |
| bench/amd-isa-backend-regalloc-accum-lds-reclaim/rl0_summary.md | ['12288'] | provenance_only | neither |
| bench/amd-isa-backend-regalloc-accum-lds-reclaim/rl1_latest.json | ['12288'] | provenance_only | neither |
| bench/amd-isa-backend-regalloc-accum-lds-reclaim/rl2_latest.json | ['12288'] | provenance_only | neither |
| bench/amd-isa-backend-regalloc-accum/ra3_latest.json | ['12288', '4096'] | provenance_only | neither |
| bench/amd-isa-backend-regalloc-accum/ra3_summary.md | ['4096'] | provenance_only | neither |
| bench/amd-isa-backend-regalloc-accum/ra4_latest.json | ['12288'] | provenance_only | neither |
| bench/amd-llvm-backend-model/latest.json | ['gfx1100'] | provenance_only | neither |
| bench/amd-multiprocess-decode-throughput/result.json | ['8B', 'gfx1100'] | provenance_only | neither |
| bench/amd-ring-overlap-characterize/result.json | ['gfx1100'] | provenance_only | neither |
| bench/amd-scheduler-tooling-backend/att_decoder_binary_probe.json | ['gfx1100'] | provenance_only | neither |
| bench/amd-scheduler-tooling-backend/execution.json | ['12288', '4096', 'Q4_K'] | provenance_only | neither |
| bench/amd-scheduler-tooling-backend/r1p1_aqlprofile_replay_proof.json | ['4096', 'gfx1100'] | provenance_only | neither |
| bench/amd-scheduler-tooling-backend/r1p2_hcq_replay.json | ['4096', 'gfx1100'] | provenance_only | neither |
| bench/amd-scheduler-tooling-backend/r1p2_hcq_replay_predispatch.json | ['gfx1100'] | provenance_only | neither |
| bench/amd-scheduler-tooling-backend/t0_capture_q8_gateup_full.json | ['12288', '4096'] | provenance_only | neither |
| bench/amd-scheduler-tooling-backend/t1b_att_aqlprofile.json | ['4096', 'gfx1100'] | provenance_only | neither |
| bench/amd-scheduler-tooling-backend/t1c_att_decoder_repair.json | ['gfx1100'] | provenance_only | neither |
| bench/amd-two-compute-ring-probe/result.json | ['gfx1100'] | provenance_only | neither |
| bench/amd-two-ring-dag-probe/result.json | ['gfx1100'] | provenance_only | neither |
| bench/amd-two-ring-dependency-probe/result.json | ['gfx1100'] | provenance_only | neither |
| bench/amd-two-stream-decode-probe/result.json | ['gfx1100'] | provenance_only | neither |
| bench/canonical-benchmarks.json | ['4096', '7900', '8B', 'gfx1100'] | provenance_only | neither |
| ... | 1625 more rows in JSON | | |

## Principle Findings

| severity | path | line | principle | finding | recommended_fix |
|---|---|---|---|---|---|
| medium | bench/amd-decode-flywheel-proof-20260614/loop-live-L2/README.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/amd-isa-backend-phase-n5/native_tile_residual/latest.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/amd-scheduler-tooling-backend/execution.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-14b-remeasure-20260612/README.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-active-surface-reduction/docs_index.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-ansor-transition-20260612/README.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-bandwidth-roofline-20260613/README.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-codegen-wmma/inmodel_matmul.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-decode-attention-fused-score-state-pv-tile/latest.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-decode-attention-generated-pv-kernel-audit/latest.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-decode-eval/candidates.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-decode-pressure-search-ownership/latest.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-flash-prefill-phase5/result.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-packed-tile-research-closeout-20260613/README.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-prefill-pipe-promotion/latest.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-prefill-pipe-tm2-tn2-hardening/promotion_package.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-prefill-pipe-tm2-tn2-hardening/summary.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-profile-opener/qwen3_8b_q4_k_m_gfx1100/latest.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-pure-search-gap/latest.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | bench/qk-repo-principles-cleanup/build_repo_inventory.py | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | bench/qk-repo-principles-cleanup/inventory.json | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | bench/qk-repo-principles-cleanup/inventory.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | bench/qk-repo-principles-cleanup/inventory.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | bench/qk-search-spaces/README.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-search-spaces/decode_attention_loop_search_space.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-semantic-20260612/README.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qk-tensile-extraction/codegen_oracle.json | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | bench/qwen-adapter-20260613/training-data-v4_1-compiler/README.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/README.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/README.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/abstractions4.py | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-a1-generated-skeleton-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/decode-attention-a1-generated-skeleton-result.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/decode-attention-a1-generated-skeleton-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/decode-attention-a1-generated-skeleton-scope.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/decode-attention-a2-wholecache-skeleton-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-a3-1-vdot2-score-lowering-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-a3-1-vdot2-score-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-a3-10-tile-prob-partial-pv-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-a3-2-cross-lane-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-a3-2b-scoped-lane-map-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-a3-2b-xlane-score-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-a3-3-lds-tile-lifecycle-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-a3-4-tile-combine-lifecycle-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-a3-5-minimal-tile-placeholder-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-a3-6-tile-score-max-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-a3-7-tile-prob-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-a3-8-stage-attribution-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-a3-9-tile-partial-pv-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-a3-performance-primitive-lowering-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/decode-attention-a3-performance-primitive-lowering-scope.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/decode-attention-control-plane-closure-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/decode-attention-control-plane-closure-result.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/decode-attention-online-pv-tile-p2-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-online-pv-tile-p3-lanemap-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-online-pv-tile-p4-codegen-decision-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-online-pv-tile-p4-codegen-decision-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-online-state-pv-tile-p10-xlane-output-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-online-state-pv-tile-p5-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-online-state-pv-tile-p6-lowering-bind-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-online-state-pv-tile-p7-xlane-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-attention-online-state-pv-tile-p9-scalar-numeric-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/decode-attention-online-state-pv-tile-p9-scalar-numeric-scope.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/decode-attention-primitive-complete-online-softmax-pv-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/decode-attention-primitive-complete-online-softmax-pv-scope.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/decode-attention-pure-search-gap-audit-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/decode-attention-pure-search-gap-audit-result.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/decode-attention-pure-search-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/decode-attention-pure-search-scope.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/decode-block-tile-and-isa-gate-codex-prompt.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-block-tile-and-isa-gate-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-block-tile-generated-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-block-tile-scheduling-claude-prompt.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-block-tile-scheduling-codex-prompt.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-block-tile-scheduling-execution-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-block-tile-scheduling-resolution-plan.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-codegen-list-scheduler-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-codegen-recurrence-rewire-claude-prompt.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/decode-codegen-recurrence-rewire-claude-prompt.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/decode-codegen-recurrence-rewire-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/decode-codegen-recurrence-rewire-result.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/decode-codegen-recurrence-unroll-primitive-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-codegen-recurrence-unroll-primitive-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-codegen-scheduler-capability-codex-prompt.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/decode-codegen-scheduler-capability-codex-prompt.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/decode-codegen-scheduler-capability-continuation-codex-prompt.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/decode-codegen-scheduler-capability-continuation-codex-prompt.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/decode-codegen-scheduler-capability-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/decode-codegen-scheduler-capability-scope.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/decode-fused-score-state-pv-tile-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/decode-fused-score-state-pv-tile-result.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/decode-fused-tile-occupancy-roofline-baseline.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-fused-xlane-score-pv-tile-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-fused-xlane-score-pv-tile-wd-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-generated-fused-pv-tile-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-generated-fused-score-state-pv-tile-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-generated-tile-codegen-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/decode-generated-tile-codegen-scope.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/decode-generated-tile-codex-prompt.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-generated-tile-phase2a-kstage-blocktile-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-isa-diff-gate-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-isa-diff-gate-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-physical-tile-pall-route-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-reg-store-devec-codegen-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-reg-store-devec-codex-prompt.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-score-broadcast-lifecycle-audit.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-score-broadcast-lifecycle-resolution-plan.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-score-broadcast-model-route-mmu-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-score-broadcast-route-materialization-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/decode-score-reuse-axis-ownership-scope.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/developer/developer.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/developer/developer.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/developer/layout.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/developer/layout.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| medium | docs/developer/speed.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/eb1-dependency-classification.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/eb5-ledger-report.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| high | docs/env_vars.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| medium | docs/env_vars.md | 0 | BubbleBeam/FutureSight is the only intended path | old Beam/FutureSight naming needs cleanup (stale_doc) | rename to canonical BubbleBeam/FutureSight, preserve only compatibility env vars and historical provenance |
| high | docs/g5-block-tile-oracle-result.md | 0 | BoltBeam owns policy/search/eval/reporting | tinygrad file appears to make policy/search decisions while classified as staying in tinygrad | split policy judgment into BoltBeam or mark as runner adapter with JSON evidence seam |
| ... | 435 more rows in JSON | | | | |

## Verification

- Generated from tracked files via `git ls-files`.
- Consumes `boltbeam_boundary_audit.json`.
- No files deleted, moved, renamed, or defaults changed.
