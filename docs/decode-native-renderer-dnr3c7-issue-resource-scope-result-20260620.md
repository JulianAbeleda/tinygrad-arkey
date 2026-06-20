# Decode Native Renderer DNR-3C7 Issue/Resource Scope Result - 2026-06-20

## Verdict

`SCOPE_DNR3C7_ISSUE_RESOURCE_ATTRIBUTION_READY`

DNR-3C7 scopes the next valid native decode step after DNR-3C6 refuted local static-count attribution.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3c7_issue_resource_scope.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7_issue_resource_scope_result.json
```

## Evidence

| item | value |
|---|---:|
| native DNR-2 | `171.523us` |
| best static variant | `163.177us` |
| hipcc/LLD oracle | `93.54us` |
| native gap to oracle | `77.983us` |
| best static gap to oracle | `69.637us` |
| local static movement explained | `8.346us` |
| local static movement share | `10.70%` |

The best local static path is `load_b128_dsload_b128_no_markers`. It closes global-load and LDS counts but leaves almost
all of the oracle gap.

## Scoped Tracks

| track | purpose | status |
|---|---|---|
| DNR-3C7A static resource ledger | compare native/C4/oracle VGPR, SGPR, private, LDS, occupancy, wave/workgroup shape, and live ranges | next |
| DNR-3C7B PMC counter attribution | use counters where local SQTT body decode is not usable | after C7A |
| DNR-3C7C issue/interleaving candidate | build only if C7A/C7B names a cause; change issue behavior, not just counts | blocked on C7A/C7B |
| DNR-3C7D SQTT/body tooling reopen | repair trace attribution only if PMC/static ledgers cannot answer the issue/resource question | optional reopen |

## What We Are Missing

The missing piece is not q8 dot4, q4/q8 addressing, load width, LDS read count, or marker count. Those are correct or
bounded. What is missing is attribution for why the oracle issues the same high-level work much faster:

- resource/occupancy metadata: live VGPR/SGPR pressure, private spill status, wave occupancy, and launch/resource shape;
- issue/interleaving behavior: whether the oracle overlaps loads, unpack/select, dot4, scale conversion, and reduction
  while native serializes them;
- counter-grade bottleneck evidence: memory wait vs issue occupancy vs VALU/SALU pressure vs cache locality;
- a legal schedule objective for search or BEAM. Right now there is no proven objective that says which q8 schedule is
  better before timing it.

## Do Not Do

- Do not add dead branches to match oracle branch count.
- Do not tune `s_clause` / `s_delay_alu` counts without a measured marker-placement win.
- Do not reopen load-shape or LDS-reduction count patches as standalone work.
- Do not start BEAM/search: the legal search space is still missing an issue/resource objective.
- Do not promote native DNR-3C4: it remains far behind the oracle despite correctness.

## Decision

If q8 work continues, the best path is either:

1. keep the q8 artifact oracle route as the practical decode path; or
2. fund DNR-3C7A/C7B to produce issue/resource attribution before any new native emitter work.

No renderer defaults changed.
