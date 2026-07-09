# Prefill Baseline Audit Bundle Scope

Date: 2026-07-09

## Goal

Create one starting-point audit artifact for the current generated baseline, without adding a new benchmark path.
The bundle reads the existing authorities and lower-layer probes, then reports what is known and what still needs
to be measured.

## Reused Inputs

| Layer | Existing tool/artifact | Purpose |
| --- | --- | --- |
| Whole-prefill authority | `extra/qk/prefill_whole_synced.py` | Baseline and candidate tok/s, route attribution, role routing. |
| Scheduler gate | `extra/qk/prefill_v2_schedule_table_gate.py` | Whether the banked generated schedule table still applies. |
| Route census | `extra/qk/prefill/prefill_route_census.py` | Normalized instruction/wait/LDS/global counters for generated and hand routes. |
| Shape matrix | `extra/qk/prefill/hand_vs_generated_shape_matrix.py` | Active-shape generated-vs-hand timing and structural deltas. |

## Done Definition

- A single JSON report exists at `bench/prefill-baseline-audit/latest.json`.
- The report includes baseline whole-prefill tok/s and route attribution.
- If a candidate authority artifact exists, the report computes candidate-vs-baseline deltas.
- If lower-layer artifacts exist, the report summarizes their key blockers rather than rerunning or duplicating them.
- If lower-layer artifacts are missing, the report prints exact commands to produce them with existing harnesses.

## Command

```sh
PYTHONPATH=. python3 extra/qk/prefill/baseline_audit_bundle.py --json
```

The collector is intentionally non-invasive: it is a diagnosis index over existing probes, not a replacement for them.
