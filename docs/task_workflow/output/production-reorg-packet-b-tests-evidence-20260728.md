# Production Reorganization Packet B: Tests and Evidence

Date: 2026-07-28

Audited commit: `c78666bc8ef47a73207bed461bf483ab98696a07` (`exp`)

## Method and coverage

This is the read-only Packet B handoff reconciled against Packet A. No GPU command, test run, generated-output run,
branch change, source edit, commit, or push was performed during the audit. This report is the only authorized edit.

The packet covers exactly 380 tracked paths:

| Surface | Paths | Reconciliation |
|---|---:|---|
| Current fork-added or fork-modified `test/**` | 164 | 162 added, 2 modified from `6e1b61f16` |
| All tracked `bench/**` | 43 | Complete `git ls-files` set |
| All tracked `docs/artifacts/**` | 173 | Complete `git ls-files` set |

The test import partition is exact and mutually exclusive: 65 tests import `extra.qk`, 7 import another `extra`
package without importing `extra.qk`, 81 import `tinygrad` without importing `extra`, and 11 contain no direct
`tinygrad` or `extra` import. It is reproducible with:

```sh
git diff --name-status --diff-filter=AM 6e1b61f16..HEAD -- test
rg '^(from|import) extra\.qk' <path>
rg '^(from|import) extra\.' <path>
rg '^(from|import) tinygrad' <path>
```

Location and imports are discovery evidence, not sufficient proof of production ownership. A test follows the final
owner of the behavior it protects.

## Reconciled `extra.qk` tests

### Production regression candidates: 14

Disposition: `promote` to `master` with the corresponding Packet A implementation. Until that promotion occurs these
remain on the richer tier. They protect route admission and purity, memory-adaptive routing, current decode, flash
fallback, Q4K decode/prefill, and pure-search fail-loud behavior.

```text
test/unit/test_current_decode_execution_adapter.py
test/unit/test_flash_buffer_roles.py
test/unit/test_flash_decode_attention_spec.py
test/unit/test_memory_adaptive_allocation_observer.py
test/unit/test_memory_adaptive_model_integration.py
test/unit/test_memory_adaptive_policy.py
test/unit/test_memory_adaptive_route_manifest.py
test/unit/test_memory_adaptive_runtime_collector.py
test/unit/test_prefill_graph_gemm_route.py
test/unit/test_pure_search_guard_boundary.py
test/unit/test_q4k_prefill_route_spec.py
test/unit/test_q4k_w_f16_decode.py
test/unit/test_qk_route_purity.py
test/unit/test_route_admission_consistency.py
```

Record: `master | production regression | promote | same test path | promoted tinygrad route/memory/decode owner |
retain while shipped behavior exists | high | Packet A implementation must be promoted first`.

### Debug and qualification tests: 20

Disposition: `move`/retain on `dev`. These cover operational safety, capture, resource inspection, benchmark
entrypoints, isolated execution, reproducers, and qualification measurement.

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
```

Record: `dev | debug qualification test | move | same path on dev | Packet A operational/debug tools | retain while
the named diagnostic contract is supported | high | none`.

### Experimental tests: 26

Disposition: retain on `exp` while the corresponding bounded research candidate is active.

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

Record: `exp | experimental test | retain | same path on exp | MMQ/packed-WMMA/kernel-vocabulary/shared-attention
research | retain only while candidate is active | high | exact removal requires candidate closeout`.

`test/unit/test_mmq_ds4_logical_emitter.py` becomes a deletion candidate with Packet A's delete-ready
`mmq_ds4_probe_contract`, but its non-DS4 assertions must first be consolidated or shown non-unique.

### Mixed-owner unresolved tests: 5

```text
test/unit/test_attn_qo_register_compile.py
test/unit/test_current_prefill_execution_adapter.py
test/unit/test_generated_quant_binding_audit.py
test/unit/test_model_profiles.py
test/unit/test_runtime_specs.py
```

Record: `unresolved | unresolved | unresolved | unresolved | mixed Packet A active_validation/not_test_like/runtime
spec/profile owners | retain unchanged | high | exact imported modules do not yet have final per-path owners`.

## Other `extra` tests: 7

```text
test/unit/test_codebase_organization_audit.py
test/unit/test_flash_variant_fingerprint.py
test/unit/test_lowering_baseline.py
test/unit/test_lowering_fingerprint.py
```

Record: `master | production regression | retain | same path | Packet A production extra/audit authorities | retain
with repository-boundary and canonical fingerprint contracts | high | generator must remain production-owned`.

```text
test/unit/test_llama_bench_artifacts.py
test/unit/test_llm_decode_correctness.py
```

Record: `dev | debug qualification test | move | same path on dev | Packet A non-CLI extra/llm benchmark tools |
retain while model qualification is supported | high | none`.

```text
test/unit/test_gpu_lock.py
```

Record: `unresolved | unresolved | unresolved | unresolved | extra.usbgpu.tools.with_gpu_lock | retain unchanged |
high | Packet A leaves extra/usbgpu unresolved`.

## Remaining test groups: 92

The 81 `tinygrad`-only paths are the exact current fork A/M test set matching a direct `tinygrad` import after excluding
every direct `extra` importer enumerated above. Their record remains:

`unresolved | production regression, debug qualification, or experimental test | unresolved | unresolved | direct
fork-modified tinygrad implementation | retain unchanged | high | Packet A's 28 mixed not_test_like and 21
active_validation paths lack exact owners`.

The no-direct-import group is exactly:

```text
test/unit/fixtures/ffn_gate_up_pm4_pre_submit_real.json
test/unit/test_analyze_faults.py
test/unit/test_egpu_minimal_compute.py
test/unit/test_egpu_qualify.py
test/unit/test_measurement_authority.py
test/unit/test_size_accounting.py
test/unit/test_tinygpu_install_script.py
test/unit/test_tinygpu_native_source.py
test/unit/test_tinygpu_server_source.py
test/unit/test_tinygpu_wire_spec.py
test/unit/test_tinygrad_boundary.py
```

`test_measurement_authority.py`, `test_size_accounting.py`, and `test_tinygrad_boundary.py` are `master` production
policy regressions. `test_analyze_faults.py` and the two eGPU tests are `dev` diagnostics/qualification tests. The FFN
fixture and four TinyGPU source/install/wire tests remain unresolved pending exact Packet A consumer ownership.

## Bench evidence: 43

| Exact path group | Count | Owner/category/disposition | Consumer and retention |
|---|---:|---|---|
| `bench/codebase-organization-audit/**` | 4 | `master`, production authority, retain | Canonical organization audit; generator is Packet A production `extra/audit` |
| `bench/flash-variant-fingerprint/**` | 1 | `dev`, qualification evidence, move | Promote only if frozen as canonical production baseline |
| `bench/lowering-cpu-fingerprint/**` | 1 | `dev`, qualification evidence, move | Lowering investigation snapshot |
| `bench/lowering-refactor-baseline/**` | 2 | `dev`, qualification evidence, move | Baseline plus pass inventory |
| `bench/prefill-lds-single-buffer-probe-20260723/**` | 2 | `exp`, experimental evidence, retain | Active packed-WMMA probe only |
| `bench/prefill-pmc-per-dispatch/**` | 1 | `dev`, debug evidence, move | Retain until counters are banked compactly |
| `bench/prefill-pure-full-kernel/**` | 16 | `exp`, experimental evidence, retain/consolidate | Candidate search and timing; remove raw sessions after closeout |
| `bench/pure-machine-search-default-path-census/**` | 4 | `dev`, qualification evidence, move | Reusable route census |
| `bench/qk-search-spaces/default_route_manifest.json` | 1 | `master`, production manifest, retain | Packet A production route manifest |
| Remaining `bench/qk-search-spaces/**` | 11 | `exp`, experimental manifests, retain | Profiles, targets, regenerated variants, semantics, and search/refutation records |

Counts sum to 43. The `qk-search-spaces` README may move with a later split if it becomes the canonical operating
document for the production manifest; it is conservatively included in the 11-path experimental group now.

## Artifact evidence: 173

| Exact path group | Count | Owner/category/disposition | Consumer, retention, or recovery |
|---|---:|---|---|
| Three `qwen3-14b-prefill-*-frozen-20260718/**` directories | 18 | `dev`, qualification fixture, move | No production runtime consumer demonstrated; retain with staged qualification |
| `qwen3-14b-prefill-attn-qo-staged-{9119a7462-20260720,951d3615c-20260719}/**` | 62 | `dev`, qualification evidence, move/consolidate | Packet A active-validation candidate; compact after verdict |
| `qwen3-14b-prefill-ffn-gate-up-staged-3fa4cd619-20260719/**` | 22 | unresolved qualification evidence | Exact Packet A candidate owner is missing; retain unchanged |
| Three standalone shared-attention diagnostic JSON files | 3 | `dev`, debug evidence, move | Fault/attribution investigation records |
| `shared-attention-acc-slice-vgpr-attribution-20260723/**` | 7 | `dev`, debug evidence, move/consolidate | Maintained attribution document is consumer |
| `shared-attention-benchmark-20260723/{STATUS.md,summary.json}` | 2 | `dev`, compact qualification evidence, move | Retained campaign conclusion |
| `shared-attention-benchmark-20260723/raw/**` | 16 | `delete`, raw artifact, delete | Replaced by status/summary; recover from `8e829a1d3` |
| `shared-attention-benchmark-replay-20260723/summary.json` | 1 | `dev`, compact qualification evidence, move | Retained replay conclusion |
| `shared-attention-benchmark-replay-20260723/raw/**` | 16 | `delete`, replay output, delete | Replaced by summary; recover from `9d7b985e8` |
| `shared-attention-g2-lds-replay-20260723/{README.md,summary.json}` | 2 | `dev`, compact debug evidence, move | Retained replay conclusion/provenance |
| `shared-attention-g2-lds-replay-20260723/raw/**` | 2 | `delete`, replay output, delete | Replaced by README/summary; recover from `f3327febe` |
| `shared-attention-m10e1-20260723/shared_attention_proof.json` | 1 | `exp`, experimental proof, retain | Consumed by experimental shared-attention authority |
| `shared-attention-m10e1-20260723/README.md` | 1 | `exp`, experimental record, retain | Proof-bundle provenance |
| Other files in `shared-attention-m10e1-20260723/**` | 12 | `delete`, raw artifact, delete | Replaced by README/proof/docs; recover from `262744515` |
| `shared-attention-split-replay-20260725/**` | 2 | unresolved deletion candidate | Recover from `54afddb84`; exact compact replacement must be named before deletion |
| `shared-attention-wave-fence-replay-20260723/{README.md,summary.json}` | 2 | `dev`, compact debug evidence, move | Retained replay conclusion/provenance |
| `shared-attention-wave-fence-replay-20260723/raw/**` | 4 | `delete`, replay output, delete | Replaced by README/summary; recover from `1299e4531` |

The three standalone shared-attention JSON files are
`shared-attention-acc-slice-diagnostic-20260723.json`, `shared-attention-base6-address-negative-20260724.json`, and
`shared-attention-split-score-state-pv-20260724.json`. Artifact counts sum to 173. There are 50 high-confidence raw
deletion candidates plus two unresolved split-replay deletion candidates; no deletion is authorized before R7.

## Minimum promotion test slices

1. `extra/llm/cli.py` promotion: `test/unit/test_tinygrad_boundary.py`, plus a missing focused
   `python -m tinygrad.llm --help` or dispatch regression. Existing `extra/llm` tests do not cover CLI delegation.
2. Route manifest/admission: `test_memory_adaptive_route_manifest.py`, `test_qk_route_purity.py`,
   `test_route_admission_consistency.py`, and `test_pure_search_guard_boundary.py`.
3. Memory-adaptive seam: the allocation-observer, model-integration, policy, and runtime-collector tests; include the
   route-manifest test if slice 2 has not landed.
4. Decode/flash fallback: `test_current_decode_execution_adapter.py`, `test_flash_decode_attention_spec.py`,
   `test_flash_buffer_roles.py`, and `test_q4k_w_f16_decode.py`.
5. Prefill graph/Q4K: `test_prefill_graph_gemm_route.py` and `test_q4k_prefill_route_spec.py`.
   `test_current_prefill_execution_adapter.py` joins only after its mixed owners are split or promoted together.
6. Audit authority: `test_codebase_organization_audit.py`; add each fingerprint/baseline test only with its generator
   and canonical baseline.

## Blocking dependencies

Packet A did not provide exact owners for its 28 mixed `not_test_like`, 21 `active_validation`, five unresolved
reproducers, `extra/usbgpu`, or SQTT/ROC paths. Consequently the five mixed QK tests, GPU-lock test, 81-test
`tinygrad`-only group, FFN staged bundle, TinyGPU tests, and FFN fixture cannot yet receive final owners. They must
remain retained and unresolved. No path in this report authorizes R8-R11 removal until the reconciled R7 inventory and
cleanup ledger name its final owner, compact replacement where required, and recovery commit.
