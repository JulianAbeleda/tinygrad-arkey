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

## Interpretation

The audit confirms the immediate blocker for the 14B path is not missing route discovery. The blocker is ownership:
the generated runtime needs typed descriptors for quant layouts, packed tile geometry, lowering provenance, and
candidate selection before another Q4_K-specific route is promoted.

The `custom_kernel` count is not automatically a bug. It marks every scanned binding point that must either become
descriptor-owned generated runtime behavior, remain an explicitly banned rollback oracle, or be removed from the
candidate path.

No new Q4_K route branches should be added before the descriptor/candidate layer exists.

## Next Phase

1. Add descriptor types in `tinygrad/llm/runtime_specs.py` for quant layouts, tile geometry, lowering provenance,
   and candidate metadata.
2. Register current generated and transitional routes through those descriptors without changing runtime behavior.
3. Convert this audit into a descriptor validation gate so promoted defaults require generated ownership.
