# docs/ - Current Map

This directory is intentionally small. Old probe logs, obsolete handoffs, and upstream marketing/demo docs were removed
from the tracked tree; use git history for archaeology.

## Start Here

- `../README.md` - repo purpose, current performance snapshot, and run commands.
- `pure-machine-search.md` - project rule: machine search over reusable compiler primitives; strict purity audit terms.
- `asm-tool-vs-hand-kernel-policy-scope.md` - policy separating backend-emitted ASM and compiler primitives from
  hand-authored full-kernel schedules.
- `8b-prefill-lifecycle-compiler-primitives-scope.md` - 8B prefill plan for reducing the 5k hand oracle into
  machine-searched compiler primitives.
- `8b-prefill-e2e-mvp-lifecycle-ownership-scope.md` - two-push plan: opt-in E2E pipe MVP first, full lifecycle ownership
  second.
- `8b-prefill-path1-mixed-mvp-scope.md` - active Path 1 MVP: normal AMD lifecycle plus generated pipe primitive route,
  with a fail-closed whole-prefill gate.
- `8b-prefill-ffn-gate-up-lds-primitive-scope.md` - next primitive track for reducing the `ffn_gate_up` LDS hand
  oracle into a generated compiler-owned LDS WMMA primitive.
- `8b-prefill-hybrid-lds-dbuf-primitive-scope.md` - feasibility and guardrails for hand-authored reusable LDS/DBUF
  compiler primitives without reverting to a full hand-tuned kernel.
- `8b-prefill-generated-lifecycle-performance-integration-scope.md` - current post-DBUF scope: route identity,
  fail-closed e2e binding, per-role timing, and lifecycle-density gates needed before another performance primitive.
- `8b-prefill-generated-dbuf-clustering-blocker-scope.md` - current narrow blocker: combine D3 next-slot DBUF cadence
  with phase-scoped LDS fragment residency / WMMA clustering before another e2e promotion attempt.
- `8b-prefill-lifecycle-compression-audit-20260709.md` - layered audit showing where the 58-TFLOPS gap is already
  visible before e2e: final stream density and epoch-unsafe stage suppression.
- `8b-prefill-epoch-aware-stage-movement-scope.md` - implementation scope for the next primitive: safe producer-epoch
  keyed D3 stage movement/suppression.
- `8b-prefill-generated-pipe-lowerer-mvp-scope.md` - first generated pipe lowerer slice: diagnostic compiler-owned
  b128/WMMA/store structure and the remaining wait-policy/route-transport blockers.
- `8b-prefill-pipe-mvp-rest-scope.md` - remaining opt-in pipe MVP gates using existing harnesses only.
- `8b-prefill-hybrid-machine-search-over-backend-atom-scope.md` - hybrid-machine-search classification and done gates for machine search
  over a hand-coded reusable DBUF backend atom; explicitly hybrid, not pure generated and not full hand-kernel ownership.
- `model-agnostic-route-dehardcoding-scope-20260710.md` - scope for moving 8B/14B route shape constants into reusable
  profile/policy data while reusing the existing route manifest, policy loader, and gates.
- `model-route-plan-implementation-scope-20260710.md` - implementation scope for making GGUF load produce model facts
  and a primitive route plan consumed by Q4/Q6 install.
- `model-fact-routing-consolidation-scope-20260710.md` - consolidation scope for proving route-plan parity,
  carrying route roles into primitives, and keeping live scripts profile-driven.
- `model-routing-decouple-prune-scope-20260710.md` - prune scope for deleting unearned route dispatch scaffolding,
  decoupling Q4/Q6 defaults from route policy, and repairing the stale E2E harness surface before core cleanup.
- `budget-reduction-lanes-3-5-scope-20260710.md` - executable scope for the next budget lanes: Q4/Q6 install-loop
  dedupe, model admission extraction, and staged AMD ISA / prefill machine-search extraction.
- `handwritten-kernel-exhaustive-lowering-scope-20260706.md` - exhaustive lowering scope for converting handwritten
  route surfaces into pure tinygrad/codegen paths.
- `pure-machine-search-roadmap.md` - current route/search roadmap and promotion rules.
- `bubblebeam-futuresight-terminology-20260625.md` - naming and ownership terms.
- `maintenance-cleanup-20260703.md` - cleanup decisions and removed surfaces.

## Runtime Boundary

- `tinygrad-runtime-operational-policy-r10.md` - runtime operating policy.
- `tinygrad-runtime-client-separation-roadmap-20260630.md` - client/runtime separation plan.
- `tinygrad-runtime-client-separation-implementation-status-20260630.md` - implementation status.
- `tinygrad-client-context-envelope-v1.md` - context envelope contract.
- `tinygrad-repo-index-adapter-boundary-v1.md` - repo index and adapter boundary.

## Local References

- `quickstart.md`, `env_vars.md`, `dtypes.md`, `nn.md` - retained tinygrad reference docs that are still useful for
  running and debugging the fork.

## Authorities

- BoltBeam owns route policy, candidate/search schema, evaluation policy, roofline reports, and ledgers.
- tinygrad owns runtime execution, compiler/backend lowering, and hardware gates.
- `tinygrad/llm/route_policy.py` consumes `boltbeam.route_policy.v1`.
- `tinygrad/llm/route_ops.py` is the runtime adapter for QK route implementations.
- `bench/qk-search-spaces/default_route_manifest.json` is the local route-state manifest.
- `extra/audit/pure_machine_search_default_path_census.py` is the local generated-default census.
- `extra/qk/lowering_phase_registry.py`, `extra/qk/lowering_done_criteria.py`, and
  `extra/qk/exhaustive_lowering_report.py` are the completion-authority sources for handwritten-surface lowering
  (work queue + audit + phase metadata + L3/L4/L5 gate criteria).

## Boundary Rule

tinygrad hot paths must not import `extra.qk.*`, `extra.qk.quant.*`, or `extra.audit.*` directly. Use
`tinygrad/llm/route_ops.py` for LLM routes and `tinygrad/codegen/experimental.py` for default-off codegen probes.
`test/unit/test_tinygrad_boundary.py` enforces this.
