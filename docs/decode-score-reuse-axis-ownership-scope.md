# Decode score reuse / axis ownership scope

Goal: prove and implement the next primitive after the PALL lifecycle route timeout: compute a lane-sharded q.k score once and reuse it across PV output columns inside a generated physical tile.

Why this exists:

The PALL lifecycle route is correctness-clean and route-clean, and it emits fdot2, LDS, and cross-lane ISA. W==D still timed out because the generated lifecycle nests score work under the PV output-column axis. The column-scaling artifact confirms this: 130 output columns took `52.43x` the 1-column runtime.

Two paths to test:

| Path | Question | Success means | Failure means |
|---|---|---|---|
| Score-once split state | Can generated code compute online m/l once without a PV column axis? | q.k can be lifted out of output-column ownership. | The primitive gap is deeper than axis ownership. |
| Chunked score-broadcast fused PV | Can generated code compute q.k once per token per PV chunk, then update multiple PV columns from that score? | Route a chunked score-reuse lifecycle next. | Need a new first-class codegen primitive for cross-axis score broadcast/reuse. |

Canonical probe:

`extra/qk_decode_physical_tile_score_reuse_paths_probe.py`

Artifact:

`bench/qk-decode-primitive-space/score_reuse_paths_latest.json`

Decision rule:

If chunked score-broadcast passes with sublinear column scaling, route it next behind a default-off flag. If only score-once passes, do not route; implement a first-class generated axis-ownership primitive that permits score values to be owned outside the PV output-column axis and broadcast into column updates.

Current result:

| Artifact | Verdict | Meaning |
|---|---|---|
| `bench/qk-decode-primitive-space/score_reuse_paths_latest.json` | `SCORE_REUSE_PATHS_PASS__BROADCAST_PROBE_READY` | Both score-once state and chunked score-broadcast PV are expressible in generated code with fdot2, LDS, cross-lane, and no spill. |

Measured score-broadcast scaling:

| PV columns | Median seconds | Runtime multiple vs 1 col | Numeric |
|---:|---:|---:|---|
| 1 | 0.004612 | 1.00x | pass |
| 8 | 0.007910 | 1.72x | pass |
| 32 | 0.017295 | 3.75x | pass |

Comparison to old lifecycle scaling:

| Shape | 32-col multiple vs 1 col |
|---|---:|
| Old PALL lifecycle, score nested under column axis | 13.05x |
| New chunked score-broadcast probe | 3.75x |

Decision:

The missing primitive is not fundamentally blocked, but the implemented route is chunked. It removes the per-output-column q.k wall by broadcasting q.k across 32 PV columns at a time; it does not yet achieve one q.k pass across all 128 PV columns. The next implementation step is to make the chunked route model-clean, then run route/materialization and a bounded W==D falsification.

Route attempt:

| Artifact | Verdict | Meaning |
|---|---|---|
| `bench/qk-decode-primitive-space/score_broadcast_chain_latest.json` | `SCORE_BROADCAST_CHAIN_READY__ROUTE_NEXT` | The standalone route chain is numerically correct for both long and short context shapes. |
| `bench/qk-decode-primitive-space/score_broadcast_route_latest.json` | `SCORE_BROADCAST_ROUTE_FAIL` | Model route capture fails with an AMD MMU fault before a clean route verdict. |

Route implementation:

The default-off flag is `DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE=1`. The route is implemented as:

| Stage | Kernel shape |
|---|---|
| State | score-once physical state kernel used by final combine |
| PV | four no-spill 32-column PV chunks; each chunk recomputes q.k and broadcasts the score across its 32 columns |
| Combine | state + four PV chunks into final output |

Guardrail:

`DECODE_ATTN_SCORE_BROADCAST_CHUNKS=1/2/3` is diagnostic-only. Those reduced paths duplicate earlier PV chunks into later output-column ranges and are not full-width correctness candidates. They require `DECODE_ATTN_SCORE_BROADCAST_DIAGNOSTIC_CHUNKS=1` and must not be used for W==D.

Current blocker:

The standalone chain is clean, but model route integration faults. That means the next work is route/materialization debugging for the chained tensor path, not primitive search and not W==D.

Refined route/materialization diagnosis:

| Artifact | Verdict | Meaning |
|---|---|---|
| `bench/qk-decode-primitive-space/score_broadcast_direct_latest.json` | `SCORE_BROADCAST_DIRECT_READY__MODEL_CAPTURE_NEXT` | Direct `flash_decode_attention_whole_cache` passes numerically with the score-broadcast route. |
| `bench/qk-decode-primitive-space/score_broadcast_chain_latest.json` | `SCORE_BROADCAST_CHAIN_READY__ROUTE_NEXT` | The standalone state + four PV chunks + combine chain passes numerically. |
| `bench/qk-decode-primitive-space/score_broadcast_route_latest.json` | `SCORE_BROADCAST_ROUTE_FAIL` | Full model route capture still faults with an AMD MMU fault. |

Rejected fix:

Forcing `.realize()` barriers inside `flash_decode_attention_whole_cache` is not valid under model/JIT capture because device realization is disallowed while constructing the graph.

Applied fix:

The score-broadcast route now uses static `Smax` kernel ranges and keeps `Tc_u` only as an in-range guard. This removes symbolic split-count ranges, but does not fix the full model MMU fault.

Current narrow blocker:

The route is valid outside full model capture and invalid inside full model capture. The next target is the model/JIT materialization boundary for multi-consumer custom-kernel chains, not the physical score primitive.
