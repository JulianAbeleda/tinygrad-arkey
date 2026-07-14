# docs/ - Current Map

Closed campaign scopes and superseded probe logs live in git history, not the current tree.

## Start Here

- `../README.md` - repo purpose, current performance snapshot, and run commands.
- `prefill-current-state.md` - current prefill status and evidence pointers.
- `pure-machine-search.md` and `pure-machine-search-roadmap.md` - search rules and promotion gates.
- `prefill-flag-graveyard.md` - compact verdicts for removed flags and paths.

## Runtime Boundary

- `tinygrad-runtime-operational-policy-r10.md` - runtime operating policy.
- `tinygrad-runtime-client-separation-roadmap-20260630.md` - client/runtime separation plan.
- `tinygrad-runtime-client-separation-implementation-status-20260630.md` - implementation status.

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

## Boundary Rule

tinygrad hot paths must not import `extra.qk.*`, `extra.qk.quant.*`, or `extra.audit.*` directly. Use
`tinygrad/llm/route_ops.py` for LLM routes and `tinygrad/codegen/experimental.py` for default-off codegen probes.
`test/unit/test_tinygrad_boundary.py` enforces this.
