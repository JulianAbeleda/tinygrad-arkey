# GEMV G3 Codegen Lowering Scope

## Goal

Convert the G2 LaneMap representation into the physical generated kernel shape needed for fast Q4_K decode GEMV.

G2 proved three things:

- The lane map is expressible: `lane = block_group * 8 + word_col`.
- The packed Q4_K word address is expressible and coalesced across `word_col`.
- A generated Tensor/scheduler route can be token-correct and route-clean.

G2 did not prove speed. `Q4K_GEMV_SCHEDULER=5` is only `14.2 / 14.2 / 14.1 / 14.0` tok/s @ctx512/1024/2048/4096 versus owned `103.4 / 101.5 / 98.8 / 94.2`.

G3 is therefore a codegen-lowering project, not a representation project.

## Problem Statement

The generated route has the right address algebra but does not lower into the owned physical kernel shape.

Required physical shape:

```text
one row per wave
lane = block_group * 8 + word_col
one packed uint32 Q4_K word per lane per group-pair
one loaded word reused for multiple nibble contributions in registers
block-group K split across lanes
cross-lane generated reduction
single output store per row
```

Current G2.3 generated route failure:

```text
Q4K_GEMV_SCHEDULER=5
  -> generated Tensor/scheduler route
  -> no owned warp custom kernel
  -> no lane-partition bridge
  -> tokens match
  -> ~14 tok/s
  -> SEARCH_GENERATED_WD_FAIL
```

This means the remaining wall is lowering: generated code does not preserve the one-word-per-lane in-register dequant/reduce structure.

## Execution Phases

### G3.0: Codegen mismatch capture

Status: complete. `extra/qk_gemv_g3_codegen_capture.py` emits `bench/qk-gemv-g3-codegen-capture/latest.json` with verdict `G3_CODEGEN_MISMATCH_CAPTURED`.

Result:

| Arm | owned gate/up | bridge gate/up | named gate/up programs |
|---|---:|---:|---:|
| `owned` | 72 | 0 | 72 |
| `bridge` | 0 | 72 | 72 |
| `g2_lanemap` | 0 | 0 | 0 |

Interpretation: G2 LaneMap is route-clean, but it has no generated gate/up program shape. It lowers into generic Tensor programs instead of a one-word-per-lane in-register dequant/reduce kernel.

Goal: capture the generated program shape for `owned`, `bridge`, and `g2_lanemap` arms and name the exact mismatch before changing lowerers.

Gate:

- Emit `bench/qk-gemv-g3-codegen-capture/latest.json`.
- Build one decode ctx for each arm.
- Record program names, global/local sizes, route counts, source keyword summaries, and structural verdict.

Expected current verdict:

```text
G3_CODEGEN_MISMATCH_CAPTURED
```

### G3.1: One-word-per-lane lowering hook

Status: next.

Goal: make generated code preserve `word_col` as a lane-local packed-word load, rather than lowering the LaneMap path as a generic Tensor graph.

Required evidence:

- A generated program for gate/up exists without custom bridge names.
- Rendered source/linear metadata shows wave32 row ownership and packed uint32 word load per lane.
- Program shape is closer to owned/bridge: one row-wave style kernel, not scalarized generic dequant/reduce graph.

Kill:

- If generated lowering cannot bind `word_col` to lane-local load without a broader `add_gpudims` rewrite, classify as `SEARCH_BLOCKED_BY_CODEGEN: lane-local packed-word load binding missing`.

### G3.2: In-register multi-nibble dequant

Goal: keep one loaded packed word live and reuse it for the relevant nibble contributions.

Required evidence:

- Rendered source or UOp/codegen summary shows a single packed word load feeding multiple shifts/masks/FMAs.
- No full dequant materialization.
- No scalar row loop over all positions.

Kill:

- If the graph can express address coalescing but cannot express reuse of loaded packed words across nibble contributions, classify as `SEARCH_BLOCKED_BY_CODEGEN: packed-word reuse primitive missing`.

### G3.3: Generated lane reduction

Goal: combine lane partials through generated cross-lane reduction, not the lane-partition bridge.

Required evidence:

- Route-clean program counts remain true.
- Generated reduction is visible as cross-lane lowering or equivalent generated reduction.
- No `q4k_lane_partition_gemv_*` program fires.

### G3.4: W==D promotion decision

Goal: decide whether the lowered generated route is promotable.

Gate:

- Run interleaved W==D @ctx512/1024/2048/4096.
- Tokens match.
- Generated route remains route-clean.
- Promotion threshold: generated route reaches the chosen owned/FutureSight threshold.

Decision labels:

```text
GEMV_PURE_SEARCH_GENERATED
SEARCH_GENERATED_WD_FAIL
SEARCH_BLOCKED_BY_CODEGEN
SEARCH_BLOCKED_BY_RUNTIME
```

## Non-Goals

- Do not change defaults.
- Do not route BubbleBeam/FutureSight to the generated arm until G3.4 passes.
- Do not solve attention TILE+COMBINE.
- Do not solve prefill.
- Do not claim purity while the bridge remains the selected fast path.

## Current Completion

G3.0 is complete when the capture artifact exists and records the current mismatch. G3 overall is complete only when either a generated route is promoted or the missing lowering primitive is precisely classified.
