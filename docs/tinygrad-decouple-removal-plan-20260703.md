# Tinygrad Decouple-First Removal Plan - 20260703

Source audit verdict: `PASS_WITH_COMPAT_SHIMS`

## Counts

- **keep_runner**: 28
- **make_shim**: 4
- **port_first**: 248
- **stage_for_review**: 2646

## Execution Order

1. audit BubbleBeam/FutureSight and old Beam/FutureSign references
2. separate canonical path, compatibility aliases, and stale naming
3. confirm or add BoltBeam replacements for duplicate policy/search/report tools
4. summarize provenance docs before removal
5. remove stale docs/tools only after summary or replacement exists
6. prune unused upstream surfaces only after import/test evidence
7. leave active runtime/compiler dependencies alone

## Decouple First

| path | recommended_action | decoupled_first | decoupling_evidence |
|---|---|---|---|
| bench/pure-machine-search-default-path-census/default_route_table.json | port_first | False | needs_port |
| bench/pure-machine-search-default-path-census/fallback_table.json | port_first | False | needs_port |
| bench/pure-machine-search-default-path-census/latest.json | port_first | False | needs_port |
| bench/pure-machine-search-default-path-census/summary.md | port_first | False | needs_port |
| bench/qk-bandwidth-roofline-20260613/README.md | port_first | False | needs_port |
| bench/qk-decode-eval/candidates.json | port_first | False | needs_port |
| bench/qk-lifecycle-search/candidates.json | port_first | False | needs_port |
| bench/qk-lifecycle-search/evaluator_contract.json | port_first | False | needs_port |
| bench/qk-lifecycle-search/generated_candidates.json | port_first | False | needs_port |
| bench/qk-lifecycle-search/policy_exports.json | port_first | False | needs_port |
| bench/qk-lifecycle-search/refutations.json | port_first | False | needs_port |
| bench/qk-lifecycle-search/runner_bindings.json | port_first | False | needs_port |
| bench/qk-lifecycle-search/search_candidates.json | port_first | False | needs_port |
| bench/qk-lifecycle-search/search_policy.json | port_first | False | needs_port |
| bench/qk-lifecycle-search/search_schema.json | port_first | False | needs_port |
| bench/qk-lifecycle-search/summary.md | port_first | False | needs_port |
| bench/qk-lifecycle-search/template_schema.json | port_first | False | needs_port |
| bench/qk-lifecycle-search/templates.json | port_first | False | needs_port |
| bench/qk-prefill-theoretical-ceiling/roofline_floor.json | port_first | False | needs_port |
| bench/qk-pure-machine-search-gap/latest.json | port_first | False | needs_port |
| bench/tg-p14-amd-recovery-and-pure-attention-landing/practical_roofline_audit.json | port_first | False | needs_port |
| bench/tg-p14-amd-recovery-and-pure-attention-landing/practical_roofline_audit.md | port_first | False | needs_port |
| docs/amd-isa-active-surface-principles-audit-20260629.md | port_first | False | needs_port |
| docs/amd-isa-backend-e2e-roadmap-20260629.md | port_first | False | needs_port |
| docs/archive/8b-decode-remaining-gap-research-scope-20260618.md | port_first | False | needs_port |
| docs/archive/8b-decode-research-banks-roadmap-20260618.md | port_first | False | needs_port |
| docs/archive/amd-broad-backend-bb5a10-p8-global-direct-candidate-decision-result-20260620.md | port_first | False | needs_port |
| docs/archive/amd-broad-backend-bb5a10-p8-tta3-macro-candidate-result-20260620.md | port_first | False | needs_port |
| docs/archive/amd-broad-backend-bb5a10-ptm3-native-candidate-scope-20260620.md | port_first | False | needs_port |
| docs/archive/amd-broad-backend-bb5a10-tensile-layout-audit-20260619.md | port_first | False | needs_port |
| docs/archive/amd-broad-backend-roadmap-result-20260619.md | port_first | False | needs_port |
| docs/archive/amd-broad-backend-roadmap-scope-20260619.md | port_first | False | needs_port |
| docs/archive/amd-decode-bandwidth-roofline.md | port_first | False | needs_port |
| docs/archive/amd-decode-beyond-llama-roadmap.md | port_first | False | needs_port |
| docs/archive/amd-decode-demotion-search-20260616.md | port_first | False | needs_port |
| docs/archive/amd-decode-final-report.md | port_first | False | needs_port |
| docs/archive/amd-decode-flash-threshold-20260616.md | port_first | False | needs_port |
| docs/archive/amd-decode-lossy-quant-search.md | port_first | False | needs_port |
| docs/archive/amd-decode-memory-access-audit.md | port_first | False | needs_port |
| docs/archive/amd-decode-methodology-and-roadmap.md | port_first | False | needs_port |
| docs/archive/amd-decode-sequential-tax-profile-20260616.md | port_first | False | needs_port |
| docs/archive/amd-isa-decode-attention-ceiling-audit-scope-20260629.md | port_first | False | needs_port |
| docs/archive/amd-lds-research-consolidation-20260619.md | port_first | False | needs_port |
| docs/archive/amd-rocm-llamacpp-research.md | port_first | False | needs_port |
| docs/archive/amd-rocprofiler-r1p1-aqlprofile-replay-result-20260619.md | port_first | False | needs_port |
| docs/archive/amd-rocprofiler-r1p2-hcq-replay-result-20260619.md | port_first | False | needs_port |
| docs/archive/amd-rocprofiler-r1p2-v2-exporter-result-20260619.md | port_first | False | needs_port |
| docs/archive/amd-rocprofiler-r1p2-v2-exporter-scope-20260619.md | port_first | False | needs_port |
| docs/archive/amd-rocprofiler-reopen-tracks-scope-result-20260619.md | port_first | False | needs_port |
| docs/archive/amd-rocprofiler-thread-trace-audit-result-20260619.md | port_first | False | needs_port |
| docs/archive/amd-scheduler-tooling-t1b-att-aqlprofile-result-20260619.md | port_first | False | needs_port |
| docs/archive/attention-tail-after-b5-audit-result-20260622.md | port_first | False | needs_port |
| docs/archive/attention-tail-after-b5-audit-scope-20260622.md | port_first | False | needs_port |
| docs/archive/bank5-smoothquant-audit-20260618.md | port_first | False | needs_port |
| docs/archive/bank6-machine-search-infra-scope-20260618.md | port_first | False | needs_port |
| docs/archive/beam-hang-premise-audit-20260619.md | port_first | False | needs_port |
| docs/archive/candidate-template-generation-v0-result-20260621.md | port_first | False | needs_port |
| docs/archive/canonical-policy-handoff-audit-result-20260621.md | port_first | False | needs_port |
| docs/archive/canonical-policy-handoff-audit-scope-20260621.md | port_first | False | needs_port |
| docs/archive/cross-shape-generalization-search-targets-scope-20260623.md | port_first | False | needs_port |
| docs/archive/cross-vendor-isa-primitive-audit-and-search-result-20260623.md | port_first | False | needs_port |
| docs/archive/cross-vendor-isa-primitive-audit-and-search-scope-20260623.md | port_first | False | needs_port |
| docs/archive/decode-att-unblock-audit-result-20260620.md | port_first | False | needs_port |
| docs/archive/decode-attention-candidate-ab-result-20260620.md | port_first | False | needs_port |
| docs/archive/decode-ctx-slope-audit-result-20260623.md | port_first | False | needs_port |
| docs/archive/decode-ctx-slope-audit-scope-20260623.md | port_first | False | needs_port |
| docs/archive/decode-ctx-slope-lifecycle-primitive-audit-scope-20260624.md | port_first | False | needs_port |
| docs/archive/decode-dnr4-t3-candidate-grid-result-20260620.md | port_first | False | needs_port |
| docs/archive/decode-fused-coop-primitive-roadmap-scope-20260621.md | port_first | False | needs_port |
| docs/archive/decode-gap-audit-consolidated-20260622.md | port_first | False | needs_port |
| docs/archive/decode-machine-search-execution-result-20260623.md | port_first | False | needs_port |
| docs/archive/decode-machine-search-execution-scope-20260623.md | port_first | False | needs_port |
| docs/archive/decode-machine-search-readiness-package-result-20260623.md | port_first | False | needs_port |
| docs/archive/decode-machine-search-readiness-package-scope-20260623.md | port_first | False | needs_port |
| docs/archive/decode-mmvq-artifact-import-discovery-result-20260619.md | port_first | False | needs_port |
| docs/archive/decode-mmvq-large-project-p0-contract-inventory-result-20260619.md | port_first | False | needs_port |
| docs/archive/decode-mode-b-generated-tile-variant-search-scope-20260623.md | port_first | False | needs_port |
| docs/archive/decode-mode-b-search-result-20260623.md | port_first | False | needs_port |
| docs/archive/decode-native-renderer-dnr3c7a-resource-ledger-result-20260620.md | port_first | False | needs_port |
| docs/archive/decode-native-renderer-dnr3c9-new-info-ledger-20260620.md | port_first | False | needs_port |
| docs/archive/decode-owned-q8-artifact-parity-harness-result-20260620.md | port_first | False | needs_port |
| docs/archive/decode-parity-no-regression-audit-scope-20260623.md | port_first | False | needs_port |
| docs/archive/decode-q8-controlled-clock-policy-closeout-result-20260620.md | port_first | False | needs_port |
| docs/archive/decode-q8-controlled-clock-policy-closeout-scope-20260620.md | port_first | False | needs_port |
| docs/archive/decode-q8-model-route-timing-audit-result-20260620.md | port_first | False | needs_port |
| docs/archive/decode-q8-model-route-timing-audit-scope-20260620.md | port_first | False | needs_port |
| docs/archive/decode-q8-primitive-solution-audit-result-20260620.md | port_first | False | needs_port |
| docs/archive/decode-q8-primitive-solution-audit-scope-20260620.md | port_first | False | needs_port |
| docs/archive/decode-q8-producer-order-provenance-audit-result-20260620.md | port_first | False | needs_port |
| docs/archive/decode-q8-producer-order-provenance-audit-scope-20260620.md | port_first | False | needs_port |
| docs/archive/decode-q8-producer-resource-context-audit-result-20260620.md | port_first | False | needs_port |
| docs/archive/decode-q8-producer-resource-context-audit-scope-20260620.md | port_first | False | needs_port |
| docs/archive/decode-q8-research-route-hardening-result-20260619.md | port_first | False | needs_port |
| docs/archive/decode-role-contract-normalization-result-20260620.md | port_first | False | needs_port |
| docs/archive/decode-route-level-primitive-ledger-result-20260620.md | port_first | False | needs_port |
| docs/archive/decode-time-tax-audit-result-20260622.md | port_first | False | needs_port |
| docs/archive/exhaustive-gpu-lifecycle-primitive-audit-scope-20260624.md | port_first | False | needs_port |
| docs/archive/ffn-activation-gap-audit-result-20260622.md | port_first | False | needs_port |
| docs/archive/ffn-activation-gap-audit-scope-20260622.md | port_first | False | needs_port |
| docs/archive/ffn-wall-decomposition-audit-20260619.md | port_first | False | needs_port |
| ... | 180 more rows in JSON | | |

## Removal After Decouple

| path | removal_allowed_after_decouple | risk_if_wrong | recovery_path |
|---|---|---|---|

## Stage For Review

| path | tiny_principle_action | risk_if_wrong |
|---|---|---|
| .claude/commands/pure-search-loop.md | keep | breaks runtime/test/import path |
| .claude/loop.md | keep | breaks runtime/test/import path |
| .gitignore | keep | breaks runtime/test/import path |
| .python-version | keep | breaks runtime/test/import path |
| LICENSE | keep | breaks runtime/test/import path |
| README.md | keep | breaks runtime/test/import path |
| bench/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-cross-ring-dependency-probe/result.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/kernel-triage-v0/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1-featured-plus/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1-featured-plus/examples.jsonl | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1-featured-plus/prompts.jsonl | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1-featured/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1-featured/examples.jsonl | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/loop-live-L0/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/loop-live-L1/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/loop-live-L2/README.md | rename_or_summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/shadow-staged-v2/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/shadow-staged-v3/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/shadow-staged-v4/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/shadow-staged-v5/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/shadow-staged/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/shadow-v0/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/shadow-v0/runs/block-dot-blk-0-attn-output-weight-compile-gate/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/shadow-v0/runs/block-dot-blk-0-attn-output-weight/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/shadow-v0/runs/threeway-blk-0-attn-output-weight/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/targeted-outcomes-v1/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/targeted-outcomes-v1/examples.jsonl | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-adapter-smoke-v0-eval/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-adapter-smoke-v0-protocol-diagnostic/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-adapter-smoke-v0-rollout/rollouts.jsonl | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-adapter-smoke-v0-rollout/summary.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-adapter-smoke-v0/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-adapter-v0-attempt/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-baselines-v0/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-cost-model-v0/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-cost-model-v1-plus/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-coverage-plan-v1-plus/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-coverage-plan-v1/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v0/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1-featured-plus/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1-featured/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-protocol-diagnostic-v0/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-qwen3-8b-base-v0/rollouts.jsonl | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-qwen3-8b-base-v0/summary.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-decode-flywheel-proof-20260614/triage-sft-v0/README.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-decode-attention-ceiling/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-decode-attention-ceiling/loss_stack.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-decode-attention-ceiling/math_floor.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-decode-attention-ceiling/summary.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-g3-vs-owned-weight-parity/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-g3-vs-owned-weight-parity/per_role.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-g3-vs-owned-weight-parity/route_counts.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-g3-vs-owned-weight-parity/summary.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-g3-weight-promotion/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-g3-weight-promotion/route_counts.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-g3-weight-promotion/search_space_update.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-g3-weight-promotion/summary.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-grid/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-inc0/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-inc1/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-inc2/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-inc3/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-inc4/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-pc-source-trace/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-pc-source-trace/native_inst_stream.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-pc-source-trace/owned_disasm.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-pc-source-trace/summary.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-b/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-c/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-f/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-g/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-h/inmodel.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-h/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-i/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-j/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-k/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-l/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-m/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n0/disasm_native.txt | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n0/disasm_owned.txt | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n0/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n1a/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n1b/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n1b/uniformity_audit.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n2/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n2/native_trace.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n2/owned_trace.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n2/profiling_capability_audit.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n2/summary.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n2b/latest.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n2b/native_pmc.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n2b/owned_pmc.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n2b/summary.md | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n3/dynamic_target.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n3/n3f0_ctx_comparison.json | summarize_then_remove | breaks runtime/test/import path |
| bench/amd-isa-backend-phase-n3/n3f0_summary.md | summarize_then_remove | breaks runtime/test/import path |
| ... | 2546 more rows in JSON | |

## Required Tests Before Future Deletion

- `git diff --check`
- relevant tinygrad import/unit tests
- BoltBeam tests for moved policy/search/report modules
- boundary audit regeneration
