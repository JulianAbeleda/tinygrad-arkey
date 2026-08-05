# NV decode overlap - Route B3.1 aligned logical/physical DAG tooling record

Date: 2026-08-04

Authority: B3 exhaustive execution scope section 8 (Phase B3.1). Status:
tooling implemented and hermetic; no real-decode conclusion yet. Gate
G-B3-D remains ahead at B3.2.

## 1. What was built

New CPU-only tooling, no GPU required, no runtime behavior change:

- `extra/llm_research/decode/route_b3_dag_attribution.py`:
  - range-aware edge builder for both arms (one edge per (dep, new) pair,
    kind priority WAW > WAR > RAW, canonical `DepsTracker` semantics);
  - logical arm: edges over logical buffer ranges (planner reuse disabled);
  - physical arm: edges over planner arena ranges via the placement manifest;
  - edge attribution: SEMANTIC (logical edge exists) versus PLANNER_ALIAS
    (arena-reuse introduced) versus UNKNOWN (unresolved identity), each with
    exact arena, overlap range, and logical buffer ids;
  - stable call identity with per-call `identity_sha256` (pointer-free),
    ordered signature alignment gate that fails closed
    (`ALIGNMENT_CONFOUNDED`);
  - DAG metrics for both arms via the existing full-token tooling: serialized
    span, duration-weighted critical path, 2/3-queue schedules, per-group rows;
  - planner-added edge ranking by critical-path impact, top recoverable
    buffers by overlap bytes, resource-pair table, unknown accounting;
  - `PlannerManifestCollector` for the live seam, plus schema validation and
    deterministic JSON emission.
- `tinygrad/schedule/memory.py`: the existing `_memory_manifest_collectors`
  seam now passes placement evidence (`offsets`, `nbytes`, `first`, `last`)
  to collectors. Zero default-path change when no collector is installed
  (proven by the byte-identical equivalence test below).
- `test/unit/test_route_b3_dag_attribution.py`: 13 hermetic tests covering
  the section 8.6 cases 1-12 plus the RecordingDepsTracker parity control
  from the risk register.

## 2. Hermetic results

```bash
.venv/bin/python -m pytest -q test/unit/test_route_b3_dag_attribution.py
```

13 passed. Regression run over the B3.0 suites plus graph suites:
`test_full_token_dag_capture.py`, `test_cuda_graph_multi_stream_schedule.py`,
`test_graph_admission.py`, `test_graph_topology.py` -> 39 passed.

Synthetic self-test (schema fixture, anchored at
`docs/task_workflow/output/nv-decode-overlap-b3-1-synthetic-fixture-
20260804.json`):

```bash
PYTHONPATH=/home/ubuntu/tinygrad-arkey .venv/bin/python extra/llm_research/decode/route_b3_dag_attribution.py --synthetic
```

Expected fixture outcome (asserted): two logical independent chains become
physically chained by arena reuse; logical CP 35.0 us vs physical CP 63.0 us
(+28.0 us / +80%); 6 SEMANTIC RAW edges preserved; 8 PLANNER_ALIAS edges
(3 WAW, 2 WAR, 3 RAW) attributed to exact arena ranges and logical buffers;
0 UNKNOWN; alignment PASS.

## 3. Equivalence proof for the memory.py seam

`test_planner_collector_absent_present_byte_identical` runs
`memory_plan_rewrite` on a real scheduled linear (CPU) with a collector
installed and without one. After normalizing the global `Ops.UNIQUE`
rendering counter, the planned linears are byte-identical, and two separate
collector runs record byte-identical placement manifests (arena, offset,
aligned size, lifetime, held status).

## 4. Deliverables

- tooling source: `extra/llm_research/decode/route_b3_dag_attribution.py`;
- tests: `test/unit/test_route_b3_dag_attribution.py`;
- schema fixture: `docs/task_workflow/output/nv-decode-overlap-b3-1-synthetic-
  fixture-20260804.json`;
- this implementation record.

No live capture was run; the live seam for B3.2 (aligned DEFAULT_PHYSICAL and
PLANNER_FREE_LOGICAL arms over the d512 CUDA route) consumes this tooling.
