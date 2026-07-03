# Maintainability Cleanup - 20260703

This pass tightened the tinygrad/BoltBeam boundary after the decouple audit.

## Removed From tinygrad

- Generated `FILE_INDEX.md` navigation files. These were stale/regenerable and encouraged agents to treat old probes as live authority.
- Dead tests for removed tinygrad search scripts: `test_qk_demote_search.py`, `test_qk_flash_search.py`, and the tinygrad copy of the search-spec tests.
- Pure search/policy helpers that now have BoltBeam ports and no live tinygrad import path:
  `qk_artifact_cache_inventory.py`, `qk_decode_primitive_candidate_template.py`, `qk_descriptor_policy.py`,
  `qk_search_spec.py`, and `qk_semantic_candidate.py`.

## Kept In tinygrad

- Runtime/compiler/hardware harnesses.
- Runner adapters that execute tinygrad or emit evidence JSON for BoltBeam.
- `docs/archive/` provenance, because current docs still cite it for historical traceability.

## Boundary Rule

tinygrad owns execution, kernels, compiler/backend lowering, and hardware evidence. BoltBeam owns model facts,
candidate/search schema, evaluation policy, ledgers, roofline attribution, and reports. BubbleBeam/FutureSight is
the only current route path; old Beam/FutureSign wording is historical or compatibility-only.
