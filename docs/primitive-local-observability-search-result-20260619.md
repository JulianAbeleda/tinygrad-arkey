# PLO RESULT - primitive-local observability/search tooling implemented through PLO-6

Executed the build from `primitive-local-observability-search-scope-20260619.md`. This is a tooling result, not a
kernel/model route. No defaults changed, no model path changed, no hardware search was launched by default.

## Result

Implemented `extra/qk_primitive_ledger.py`, a read-only observability tool that ingests existing artifacts and emits
the first primitive-local ledger under `bench/qk-primitive-observability/`.

It implements the six scoped layers:

| phase | status | deliverable |
|---|---|---|
| PLO-1 read-only ledger collector | DONE | `ledger.jsonl`, `ledger.json`, `summary.md` |
| PLO-2 schema/evidence validators | DONE | `validation.json`, fail-fast evidence-level checks |
| PLO-3 candidate runner wrapper | DONE (replay-only) | `runner_registry.json`, `runner_smoke.json` |
| PLO-4 deterministic failure classifier | DONE | per-observation `classification` field + summary next actions |
| PLO-5 guided search memory | DONE | `search_memory.json`, `search_sessions.json` |
| PLO-6 optional trace/counter plugin inventory | DONE (inventory-only) | `trace_plugins.json` |

## First ledger [M]

The generated ledger initially had 10 observations and reconstructed the required project states. The follow-up audit
extends it to 11 observations by ingesting the TPE-7a rebindable-node artifact:

- q8/MMVQ lifecycle: **DEFERRED** behind codegen capability;
- pure-tinygrad WMMA bounded sweep: **KILL**;
- Tensile extraction TPE-5: **PASS** / generalizes;
- TPE-6 block transfer: **REDIRECT** to graph integration;
- TPE-7a rebindable node: **PASS** as a graph-protocol prerequisite;
- spec decode shortcut: **CLOSED**;
- reuse-free flash-prefill: **REFUTED**.

The runner smoke is replay-only and passes against existing artifacts. Trace/counter inventory found no `rocprofv3`
or `rocprof-compute` binary on PATH in this shell, but did find committed tinygrad SQTT examples and rocprof trace
artifacts. This matches the principle: Level-4 counters are optional plugins, not blockers.

## Principle check

- **Full primitive boundary:** observations include primitive, phase, role, shape, candidate, correctness, timing,
  metadata, runtime, evidence levels, bottleneck inference, gate, and provenance.
- **Evidence hierarchy:** every row labels available evidence; counter-free root-cause claims are not upgraded to
  Level 4.
- **No stale path resurrection:** classifier marks killed/refuted/closed rows as `do_not_reopen_without_new_evidence`.
- **No monolithic profiler:** ROCm/SQTT support is inventory-only; the base ledger runs from artifacts.
- **No model route/default change:** tool is read-only unless a future runner wrapper is explicitly invoked.

## Next use

Use the runtime-boundary adapter on the TPE-6B graph-integration arc. The key question is whether graph-capturable
Tensile launches preserve the TPE-6 **1.53x GPU FFN speedup** end-to-end by removing naive per-op host sync.

The smallest next build is not a new search loop. It is to run `extra/qk_primitive_ledger.py` after each TPE-6B probe
so graph-boundary evidence lands in the same ledger and the classifier can distinguish `graph_capture_missing`,
`program_cache_miss`, and `host_sync`.

## Files

`extra/qk_primitive_ledger.py`, `bench/qk-primitive-observability/*`, this doc. Provenance:
`primitive-local-observability-search-scope-20260619.md`,
`what-makes-a-performance-primitive-efficient-20260618.md`,
`performance-primitive-external-research-audit-20260619.md`.
