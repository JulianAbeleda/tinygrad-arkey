# GEMV Pure Search-Generated Route Scope

## Goal

Replace the BubbleBeam/FutureSight Q4_K GEMV lane-partition custom bridge with a generated/search-owned route.

Current truth:

```text
BUBBLEBEAM_FUTURESIGHT=1
  -> FutureSight selects lane_partition_q4k
  -> model routes through extra/qk_q4k_lane_partition_gemv.py
  -> GEMV_NOT_PURE__SEARCH_SELECTED_CUSTOM_BRIDGE
```

Target truth:

```text
BUBBLEBEAM_FUTURESIGHT=1
  -> BubbleBeam selects a generated Q4_K GEMV candidate
  -> no owned warp GEMV and no lane-partition custom bridge fire
  -> GEMV_PURE_SEARCH_GENERATED
```

This scope is GEMV-only. It does not include decode attention TILE+COMBINE, prefill, or a broad BubbleBeam rewrite.

## Current Authority

| Item | Authority |
|---|---|
| Pure-search roadmap | `docs/pure-machine-search-roadmap.md` |
| Layout/codegen execution plan | `docs/layout-codegen-full-scope-20260625.md` |
| GEMV search-space manifest | `bench/qk-search-spaces/decode_ffn_gemv_gfx1100_v1.json` |
| Current GEMV purity gate | `bench/qk-gemv-purity-gate/latest.json` |
| Current BubbleBeam W==D artifact | `bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-162422.json` |

Current accepted starting verdict:

```text
GEMV_NOT_PURE__SEARCH_SELECTED_CUSTOM_BRIDGE
```

This means FutureSight selected the right measured structure, but the implementation is still a custom-kernel bridge.

## First Target

Target only the FFN gate/up Q4_K GEMV shape first:

```text
in_features = 4096
out_features = 12288
phase = decode T==1
profile = Qwen3-8B-Q4_K_M / gfx1100 / wave32
```

Do not target in the first pass:

- FFN down projection.
- Attention q/o projection.
- Decode attention TILE+COMBINE.
- Prefill graph GEMM.

## Generated Route Requirements

The generated route must express the owned/lane-partition physical structure without calling the custom bridge.

Required structure:

- Packed uint32 word lane map: adjacent lanes own adjacent packed Q4_K words.
- In-register multi-nibble dequant: one loaded packed word feeds its eight nibble contributions without redundant loads.
- Block-group-K split: wave lanes cover the owned `block_group * 8 + word_col` decomposition.
- Generated reduction: cross-lane or equivalent generated reduction over the row partials.
- Single row output store.

Forbidden route attribution for a pure pass:

- No `q4k_gemv_warp_kernel` / owned warp GEMV program.
- No `q4k_lane_partition_gemv_kernel` / lane-partition custom bridge program.
- No `lane_partition_gateup` program count in the BubbleBeam/generated arm.
- No manual custom-kernel route standing in for generated code.

## Execution Phases

### G0: Bind current artifacts

Goal: make the starting point reproducible.

Tasks:

- Run `extra/qk_gemv_purity_gate.py`.
- Confirm `GEMV_NOT_PURE__SEARCH_SELECTED_CUSTOM_BRIDGE`.
- Confirm `extra/qk_validate_search_provenance.py` passes.

Gate:

- Current gate remains not-pure for the right reason: lane-partition custom bridge is used, owned warp GEMV is not used in the BubbleBeam arm, and tokens match.

### G1: Generated-route candidate skeleton

Status: implemented as `q4k_gemv_generated_skeleton` / `Q4K_GEMV_SCHEDULER=2` for route attribution. This is a correctness/attribution skeleton, not a speed claim.

Goal: introduce a default-off candidate lane that can be attributed separately from owned and bridge routes.

Tasks:

- Add a generated-route arm to the existing W==D harness or a narrow companion harness.
- Ensure route attribution can distinguish:
  - owned warp GEMV,
  - lane-partition custom bridge,
  - scheduler/generated GEMV.
- Keep `BUBBLEBEAM_FUTURESIGHT=1` canonical and do not change defaults.

Gate:

- The purity gate can report a generated arm even before it is fast.
- If the generated arm falls back to owned or bridge code, classify it as not pure.

### G2: Minimal codegen representation

Goal: express the lane-partition physical structure in generated code.

Executable scope: `docs/gemv-g2-minimal-codegen-representation-scope.md`.

Current G2 result: G2.0-G2.2 pass (`G2_LANEMAP_ADDRESS_BUILDER_PASS`), and G2.3 runtime binding is route-clean/token-correct but fails speed (`SEARCH_GENERATED_WD_FAIL`, `14.2 / 14.2 / 14.1 / 14.0` tok/s). The next blocker is codegen lowering for one-word-per-lane in-register dequant/reduce, not LaneMap/address algebra.

Current G3.0 result: `G3_CODEGEN_MISMATCH_CAPTURED` in `bench/qk-gemv-g3-codegen-capture/latest.json`. Owned and bridge each expose a named wave32 gate/up program 72 times; G2 LaneMap exposes zero named gate-up programs and lowers into generic Tensor programs.

Current G3.1 result: `G3_LANEMAP_PROMOTABLE` via `Q4K_GEMV_SCHEDULER=6`. The generated G3 LaneMap arm emits `q4k_g3_lanemap_gemv_12288_4096`, is token-correct, route-clean, and matches owned/FutureSight throughput. Remaining purity blocker: BubbleBeam/FutureSight still selects the custom bridge instead of the G3 generated candidate.

Tasks:

- Reuse the layout/codegen plan rather than inventing a second IR.
- Implement the smallest representation needed for the gate/up shape:
  - LaneMap-aware range/lane binding,
  - packed-word index as a generated address expression,
  - generated dequant/reduce/store path.
- Keep the implementation q4k/gate-up-specific until the purity gate proves value.

Gate:

- Rendered/generated route shows lane-owned packed-word access and does not call the bridge.
- Tokens match the owned/bridge reference.

Kill:

- If the RANGE/AxisType model cannot represent the `block_group * 8 + word_col` map without a broad rewrite, stop and classify as `SEARCH_BLOCKED_BY_CODEGEN` with the exact missing representation.

### G3: BubbleBeam candidate binding

Goal: make BubbleBeam select the generated candidate as a search candidate, not a manual bridge.

Tasks:

- Add a candidate record with `search_generation_status = search_generated` only after the route no longer calls owned/bridge custom kernels.
- Keep the bridge candidate classified as `search_selected_custom_bridge`.
- Update `bench/qk-search-spaces/decode_ffn_gemv_gfx1100_v1.json` with the generated route status.

Gate:

- `extra/qk_validate_search_provenance.py` passes.
- No candidate marked `search_generated` enables manual/custom flags or routes through the bridge.

### G4: GEMV purity gate

Goal: flip the route-classification gate.

Tasks:

- Run `extra/qk_gemv_purity_gate.py` against the generated arm.
- Require route attribution to prove:
  - no owned warp GEMV in the generated/BubbleBeam arm,
  - no lane-partition custom bridge,
  - generated scheduler/codegen route used,
  - tokens match.

Target verdict:

```text
GEMV_PURE_SEARCH_GENERATED
```

Failure verdicts:

```text
SEARCH_BLOCKED_BY_CODEGEN
SEARCH_BLOCKED_BY_RUNTIME
SEARCH_GENERATED_WD_FAIL
SEARCH_FOUND_LOCAL_ONLY
```

### G5: W==D performance decision

Goal: decide whether the generated route is promotable, useful but slower, or blocked.

Tasks:

- Compare owned, bridge, and generated arms at ctx 512/1024/2048/4096.
- Use interleaved W==D authority.
- Record tokens, route counts, tok/s, and verdict.

Promotion gate:

- Tokens match.
- Generated route is within the chosen threshold of owned/bridge route.
- Route attribution proves generated route only.

Kill gate:

- If generated route plateaus near the known scheduler ceiling rather than the bridge/owned route, classify as `SEARCH_GENERATED_WD_FAIL` or `SEARCH_BLOCKED_BY_CODEGEN`, depending on attribution.

## Non-Goals

- Do not re-chase cheaper attention combine.
- Do not attempt attention TILE+COMBINE fusion here.
- Do not broaden to down/q/o projections before gate/up passes.
- Do not mark FutureSight as pure while it uses the custom bridge.
- Do not use upstream tinygrad BEAM terminology for this route; this is BubbleBeam/FutureSight.

## Completion Criteria

This scope is complete when one of these is true:

1. `GEMV_PURE_SEARCH_GENERATED` is achieved for FFN gate/up and W==D passes.
2. The route is classified as blocked with a precise missing primitive or runtime boundary.
3. The generated route is correct but too slow, with W==D evidence and route attribution showing why.
