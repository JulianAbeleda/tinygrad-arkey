# Generated Quant Runtime Binding Audit

Date: 2026-07-05

Source gate: `generated_quant_binding_audit`

Source artifact: `bench/generated-quant-runtime-binding-audit/latest.json`

Verdict: `GENERATED_QUANT_BINDING_AUDIT_READY`

## Purpose

This is the Phase 1 provenance inventory from `docs/generated-quant-runtime-execution-map-20260705.md`.
It answers which generated quant routes are already scheduler/codegen-owned, which routes are transitional
hand-authored UOp templates, and where custom source or custom binding surfaces still exist.

This is an audit and ownership map. It is not a route promotion gate yet.

## Route Classification

| Classification | Count | Meaning |
| --- | ---: | --- |
| `allowed.generated` | current manifest-generated defaults | Machine-authored or tinygrad scheduler generated provenance. |
| `transitional.hand_authored_uop` | 0 default routes | Hand-authored UOp templates are no longer accepted on the default path. |
| `banned.not_default_but_reachable_or_ledgered` | opt-in/research only | External handwritten kernels or rollback oracles that are not acceptable as default generated routes. |

Default debt:

| Route | Current issue | Replacement scope |
| --- | --- | --- |
| none | Strict default purity is now expected to pass. | Raw graph-GEMM remains opt-in research via `PREFILL_GRAPH_GEMM=1`. |

## Binding Findings

| Finding kind | Count | Interpretation |
| --- | ---: | --- |
| `binding.custom_kernel` | 108 | Direct custom kernel binding surfaces that need explicit ownership or retirement. |
| `binding.ops_custom` | 6 | UOp custom op surfaces. |
| `binding.ops_customi` | 3 | UOp custom immediate op surfaces. |
| `binding.ops_program` | 1 | Program construction surface. |
| `binding.ops_ins` | 3 | Instruction-level construction surface. |
| `source_builder.inline_asm` | 2 | Functions that assemble inline assembly source. |
| `source.inline_asm` | 2 | Literal inline assembly strings. |
| `source_builder.custom_source` | 1 | Function that assembles custom C-like source. |

Unknown binding findings: none in the scanned scope.

## Candidate Registry

The audit now validates the generated candidate registry in `tinygrad/llm/generated_candidates.py`.

| Registry check | Result |
| --- | ---: |
| Registered generated candidates | 6 |
| Non-generated candidates | 0 |
| Candidates pointing at unknown route ids | 0 |

Initial registered candidates:

| Candidate | Route |
| --- | --- |
| `quant_linear_prefill.prefill_v2_scheduler_matmul_default` | `prefill_v2_scheduler_matmul_default` |
| `quant_linear_prefill.q4k_int8_wmma_tensor_substrate` | `prefill_q4k_int8_wmma_generated_research` |
| `quant_linear_prefill.q4k_int8_wmma_tiled_substrate` | `prefill_q4k_int8_wmma_tiled_research` |
| `quant_linear_decode.q4k_g3_lanemap` | `decode_q4k_g3_generated` |
| `quant_linear_decode.q6k_generated_coop` | `decode_q6k_coop_generated` |
| `attention_decode.live_split_flash` | `decode_flash_live_split_g4_8b_kvboth` |

## Interpretation

The audit confirms the immediate blocker for the 14B path is not missing route discovery. The blocker is ownership:
the generated runtime needs typed descriptors for quant layouts, packed tile geometry, lowering provenance, and
candidate selection before another Q4_K-specific route is promoted.

The `custom_kernel` count is not automatically a bug. It marks every scanned binding point that must either become
descriptor-owned generated runtime behavior, remain an explicitly banned rollback oracle, or be removed from the
candidate path.

No new Q4_K route branches should be added before the descriptor/candidate layer exists.

## Current Architecture State

Implemented:

1. Descriptor types in `tinygrad/llm/runtime_specs.py` for quantized tensors, activation quantization, runtime ops, and
   generated candidates.
2. Quant format descriptors in `tinygrad/llm/quant_specs.py` for Q4_K, Q6_K, and Q8_1.
3. Generated candidate registry in `tinygrad/llm/generated_candidates.py`.
4. Prefill direct-packed route analysis can emit a `RuntimeOpSpec` without changing route behavior.
5. The audit gate validates candidate provenance and route-id references.

Remaining:

1. `prefill_pipe_role_selective_generated` remains opt-in raw-substrate research until its WMMA execution path no longer
   wraps instruction lists as `Ops.INS`.
2. Q4_K int8-WMMA prefill remains research; it is not the shipped default.
