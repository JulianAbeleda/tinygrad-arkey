# Route authority centralization result

Date: 2026-07-29

Scope: `docs/route-authority-centralization-scope-20260729.md`

Status: CPU implementation complete. AMD route/token/performance recertification remains intentionally deferred; no eGPU was touched and no benchmark claim changed.

## Result

- Master inference owns selected route descriptors and execution.
- EXP qualification executes the production owners.
- EXP owns provenance, search readiness, benchmarks, historical evidence, and explicit oracles.
- Master imports no EXP research module.
- The unused global production policy/interface shadow was deleted.
- The duplicate EXP promoted-prefill selector was deleted.
- Flash G4/G5 binding consumes the executor-owned `FlashDecodeRouteConfig` values.
- Packed-WMMA coverage, geometry, manifest guards, search-readiness workloads, and qualification enumerate `PACKED_WMMA_ROUTES` instead of maintaining separate tables.
- The 315-line EXP packed-WMMA runtime copy was deleted. EXP injects its isolated correctness canary through the production verifier seam.
- Production graph-GEMM uses the canonical candidate identity function. EXP's default admitted-set validation uses that same function.
- EXP's duplicate graph census was deleted; whole-model qualification observes the production census.
- Q4 decode and flash qualification execute production emitters/executors. Legacy Q4/Q6/flash specifications remain only as research/parity oracles.
- Runtime role normalization has one vocabulary and the packed-WMMA dispatcher cycle is removed.
- The strategy memory planner consumes `ScannedMemoryBudget` and can no longer establish an independent admitted-VRAM budget.
- EXP's route manifest derives flash and packed applicability from production descriptors and retains evidence/provenance rather than selecting production work.

## Removed authorities and duplicate code

- `tinygrad/llm/production_route_policy.py`
- `tinygrad/llm/production_route_interface.py`
- `test/unit/test_production_route_interface.py`
- `extra/llm_research/prefill/packed_wmma_prefill_candidates.py`
- `extra/llm_research/decode/flash_decode_attention_executor.py`
- EXP copy of `automatic_promoted_prefill_graph_policy`
- EXP copy of the graph candidate census
- Master graph executor's private candidate-set hashing implementation
- Manually copied master `PACKED_WMMA_GEOM` values
- JSON copies of production flash and packed-WMMA shape guards

Net change against the starting branch heads:

- master: 148 added, 283 deleted, net -135 lines across 19 files.
- EXP: 666 added, 900 deleted, net -234 lines across 52 files. The additions include the 322-line executable scope and focused regression coverage.

## CPU validation

Master full unit suite:

```text
462 passed, 11 skipped, 4 xfailed, 8 subtests passed
```

EXP directly affected qualification/runtime gate:

```text
229 tests passed before stale-test cleanup
27 stale authority/measurement tests passed after cleanup
49 manifest/packed descriptor tests passed
34 promoted-policy tests passed
14 flash production/oracle tests passed
```

The entire EXP historical unit tree was also sampled in one run:

```text
1597 passed, 27 skipped, 4 xfailed, 73 failed
```

Five stale current-authority assertions exposed by the consolidation were repaired and pass. The remaining failures are outside the changed authority surface: unguarded AMD execution on a host with no AMD device, missing ROCm LLVM metadata tools, archived AMD ISA/compiler fixture drift, and existing attention research campaigns. They do not occur in master's complete green suite and are not represented as cleanup success.

## Static closure checks

- No master import of `extra.llm_research`.
- No production-route shadow module or caller.
- No packed-WMMA implementation import from the deleted EXP owner.
- No EXP default graph census implementation.
- No EXP promoted-prefill selection implementation.
- One master candidate-set identity implementation.
- One master packed-WMMA route-row table.
- `packed_wmma_prefill.py` does not import `prefill_routes.py`.
- Current EXP audit, qualification, and test sources do not reference the deleted packed runtime or flash executor.

## Deferred AMD recertification

When the AMD host is intentionally brought back, run the existing isolated qualification paths against production owners:

1. Confirm Q4/Q6 decode and G4/G5 flash route binding.
2. Run packed-WMMA six-row canaries through `packed_wmma_production_canary.py`.
3. Confirm whole-model route census observes production dispatch.
4. Confirm 8B and 14B token parity.
5. Compare the existing 14B pp512/1024/2048/4096 measurements without changing claims unless new evidence is recorded.

AMD recertification is a hardware evidence refresh, not unfinished code centralization.
