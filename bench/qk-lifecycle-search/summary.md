# Primitive lifecycle search - 2026-06-19

Read-only seed ledger. It does not run hardware or route a model path.

## State counts

- `closed`: 1
- `deferred`: 1
- `diagnostic`: 2
- `pass_research`: 1
- `pass_strong_policy_gated`: 1
- `project_level`: 2
- `refuted`: 1

## Ranked candidates

- `prefill_tensile_artifact_full`: `pass_strong_policy_gated`; next: Decide external artifact policy; if accepted, harden shape/fallback matrix.
- `decode_spec_weight_amortization_lifecycle`: `diagnostic`; next: Run SDB-1/SDB-2: verify budget model and T-cheap verify design audit.
- `decode_q8_artifact_lifecycle`: `pass_research`; next: Policy decision: accept research-only artifact route or keep default off.
- `decode_q8_native_codegen`: `project_level`; next: Only fund as AMD scheduler/codegen project, not primitive search.
- `prefill_tensile_codegen_transfer`: `project_level`; next: Treat as reusable AMD renderer/scheduler project, using Tensile as oracle.

## Live questions

- Can target verify be made T-cheap enough for spec decode (<=1.5x one T==1 pass)?
- Is external artifact policy acceptable for research routes?
- Does Claude's Route A/P2 dependency-free LDS work beat the current diagnostic state?
- Should q8 decode artifact route remain research-only or become a maintained opt-in?
- Is a reusable AMD renderer/scheduler project funded, or are native codegen rows closed for now?

## PLS completion

- `PLS-1 refutation memory`: 7 entries; validation `True`
- `PLS-2 runner bindings`: 7 bindings; validation `True`
- `PLS-3 policy exports`: 2 research policy candidates; defaults remain off
- `PLS-4 generator`: 8 generated rows, 4 pruned by refutations

## Generated legal rows

- `decode_q8_sidechannel_native_after_codegen_capability`: requires fused multi-output RMSNorm/q8 producer, hipcc-quality schedule or imported equivalent
- `decode_spec_tcheap_verify_forward`: requires T=K+1 target verify <=1.5x one pass, low-sync accept/commit, greedy byte-exact KV protocol
- `prefill_tensile_artifact_hardened_shapes`: requires artifact policy yes, shape/fallback matrix, versioned HSACO contract
- `prefill_tensile_native_renderer_transfer`: requires software-pipelined K-loop, spill-free accumulators, renderer/scheduler capability
