---
description: One bounded generated-route audit/improvement step for tinygrad + BoltBeam
argument-hint: "[max_steps=1]"
---

Run one bounded maintenance step in `/home/ubuntu/tinygrad-arkey` for the generated LLM route stack.

Goal: keep BubbleBeam/FutureSight generated routes as the tinygrad default and move any open work through the correct
owner: tinygrad for runtime/codegen/lowering, BoltBeam for policy/search/evaluation. This command must not recreate an
owned-kernel reconstruction loop.

Rules:
- Generated route first; owned/handwritten kernels are rollback or historical comparator only.
- No new handwritten kernels.
- No model-name branches; use structural shape/quant/target predicates.
- No stale env-flag proliferation. Add flags only for rollback or a bounded experiment with manifest provenance.
- Promotion requires correctness, route-bound proof, rollback, W==D, and practical-roofline justification.
- Use append-only artifacts for evidence and keep commits small.

Procedure:
1. Run `python extra/audit/pure_machine_search_default_path_census.py --check --strict-final-default`.
2. If it fails, fix the first generated-default violation and update `bench/qk-search-spaces/default_route_manifest.json`.
3. If it passes, choose the highest-priority open/refuted manifest row and classify it:
   - tinygrad runtime/codegen issue -> implement the smallest fix and add/update a focused gate.
   - BoltBeam policy/eval issue -> write the prompt/task and stop.
   - stale/historical issue -> remove or mark removed so agents cannot route to it.
4. Verify with `python -m compileall -q tinygrad extra test` and targeted tests.
5. Commit or revert. Do not push unless explicitly asked.

Report: verdict, evidence files, verification commands, commit SHA if committed, and the single next action.
