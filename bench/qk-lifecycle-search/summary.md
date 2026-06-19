# Primitive lifecycle search seed - 2026-06-19

Read-only seed ledger. It does not run hardware or route a model path.

## State counts

- `closed`: 1
- `deferred`: 1
- `diagnostic`: 1
- `pass_research`: 1
- `pass_strong_policy_gated`: 1
- `project_level`: 2
- `refuted`: 1

## Ranked candidates

- `prefill_tensile_artifact_full`: `pass_strong_policy_gated`; next: Decide external artifact policy; if accepted, harden shape/fallback matrix.
- `decode_q8_artifact_lifecycle`: `pass_research`; next: Policy decision: accept research-only artifact route or keep default off.
- `decode_q8_native_codegen`: `project_level`; next: Only fund as AMD scheduler/codegen project, not primitive search.
- `prefill_tensile_codegen_transfer`: `project_level`; next: Treat as reusable AMD renderer/scheduler project, using Tensile as oracle.
- `prefill_route_a_asm_lds`: `diagnostic`; next: Wait for P2 double-buffer/occupancy result before promoting.

## Live questions

- Is external artifact policy acceptable for research routes?
- Does Claude's Route A/P2 dependency-free LDS work beat the current diagnostic state?
- Should q8 decode artifact route remain research-only or become a maintained opt-in?
- Is a reusable AMD renderer/scheduler project funded, or are native codegen rows closed for now?
