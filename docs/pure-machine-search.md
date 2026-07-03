# Pure Machine Search

Pure machine search means the default runtime path is generated or spec-driven, selected by BubbleBeam/FutureSight policy
and verified by tinygrad gates. Handwritten/owned routes may exist as rollback, historical baseline, or ceiling
comparison, but they are not the target path.

## Current State

- Q4_K decode GEMV defaults to generated G3 LaneMap where structurally eligible.
- Q6_K decode GEMV defaults to a spec-driven generated coop route.
- 8B long-context decode attention defaults to generated live-split/KV_BOTH.
- 14B-style G=5 decode attention uses the generated block-tile route for its validated shape.
- Prefill uses the generated/spec-driven role-selective schedule.

The local authority is `bench/qk-search-spaces/default_route_manifest.json`; the runtime census is
`extra/audit/pure_machine_search_default_path_census.py`.

## Ownership Boundary

- BoltBeam owns candidate generation, route policy, evaluation, roofline attribution, and ledgers.
- tinygrad owns runtime execution, backend/compiler lowering, and focused hardware gates.
- tinygrad should not grow a second search-policy/evaluator stack.

## Promotion Rule

A generated route can become default only when it has:

- correctness evidence,
- route-bound/no-hidden-fallback evidence,
- rollback,
- W==D or equivalent authority timing,
- practical-roofline justification when absolute parity is the question.

Local microbenchmarks and isolated kernels are diagnostic only.

## What Remains

The hard work is not adding more handwritten kernels. It is improving the generator, lowering, route policy, and
measurement stack until generated candidates can cover more shapes and move closer to practical roofline without
manual special cases.
