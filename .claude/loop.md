# Project `/loop` task - generated-route health, one bounded step

Default task for a bare `/loop`. Keep tinygrad's hot LLM path on BubbleBeam/FutureSight generated routes. This is a
maintenance and audit loop, not owned-kernel reconstruction.

Authorities:
- Route census: `extra/audit/pure_machine_search_default_path_census.py --check --strict-final-default`
- Runtime manifest: `extra/qk/route_manifest.py`
- Decode authority: `extra/qk/decode_runtime_overhead.py`
- Correctness authority for attention changes: `extra/qk/prefilled_route_parity.py`
- Candidate policy/search ownership: BoltBeam `QK_ROUTE_POLICY`

Hard rules:
- Generated route first. Owned/handwritten paths are rollback or historical comparator only, never the default target.
- If a route regresses, first classify it as route-policy, correctness, performance, or harness debt.
- Do not add new handwritten kernels. Fix the generator, route policy, lowering, or manifest instead.
- Do not branch on model names. Use structural shape/quant/target predicates.
- Do not add a new env flag unless it is a rollback flag or a bounded experiment with manifest provenance.
- Promotion needs correctness, route-bound evidence, rollback, and W==D/practical-roofline justification.
- Commit or revert each bounded step; never leave a dirty tree as a handoff.

One fire:
1. Run the route census. If it fails, fix the first hidden fallback, stale manifest row, or boundary violation.
2. If the census passes, inspect the latest open/refuted route row in `bench/qk-search-spaces/default_route_manifest.json`.
3. If the issue is tinygrad-side, make the smallest runtime/codegen/test change and update the manifest artifact.
4. If the issue is search/evaluation-side, write the BoltBeam task prompt/artifact and stop; do not duplicate BoltBeam in tinygrad.
5. Verify with compileall and the targeted unit tests before committing.

Stop with: verdict, changed files, verification, commit SHA if committed, and one next action.
