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
| `allowed.generated` | 9 | Machine-authored or tinygrad scheduler generated provenance. |
| `transitional.hand_authored_uop` | 3 | Hand-authored UOp templates that need generated descriptor ownership before final promotion. |
| `banned.not_default_but_reachable_or_ledgered` | 4 | External handwritten kernels or rollback oracles that are not acceptable as default generated routes. |

Default debt:

| Route | Current issue | Replacement scope |
| --- | --- | --- |
| `prefill_q4k_direct_tile4x4_default` | Promoted default route with `hand_authored_uop_template` provenance. | Replace with generated 14B/32B quantized MMQ substrate under descriptor/candidate ownership. |

## Binding Findings

| Finding kind | Count | Interpretation |
| --- | ---: | --- |
| `binding.custom_kernel` | 106 | Direct custom kernel binding surfaces that need explicit ownership or retirement. |
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
| Registered generated candidates | 5 |
| Non-generated candidates | 0 |
| Candidates pointing at unknown route ids | 0 |

Initial registered candidates:

| Candidate | Route |
| --- | --- |
| `quant_linear_prefill.prefill_pipe_role_selective_generated` | `prefill_pipe_role_selective_generated` |
| `quant_linear_prefill.q4k_int8_wmma_tensor_substrate` | `prefill_q4k_int8_wmma_generated_research` |
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

1. `prefill_q4k_direct_tile4x4_default` is still transitional default debt.
2. Q4_K prefill promotion still needs route-bound correctness and 14B authority for a generated MMQ candidate.
