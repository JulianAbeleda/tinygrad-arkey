# docs/ - Current Map

This directory is intentionally small. Old probe logs, obsolete handoffs, and upstream marketing/demo docs were removed
from the tracked tree; use git history for archaeology.

## Start Here

- `../README.md` - repo purpose, current performance snapshot, and run commands.
- `pure-machine-search.md` - project rule: generated/spec-driven routes are the target path.
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
- `extra/qk/lowering_phase_registry.py` and `extra/qk/exhaustive_lowering_report.py` track handwritten-surface
  lowering phases without duplicating route/runtime facts.

## Boundary Rule

tinygrad hot paths must not import `extra.qk.*`, `extra.qk.quant.*`, or `extra.audit.*` directly. Use
`tinygrad/llm/route_ops.py` for LLM routes and `tinygrad/codegen/experimental.py` for default-off codegen probes.
`test/unit/test_tinygrad_boundary.py` enforces this.
