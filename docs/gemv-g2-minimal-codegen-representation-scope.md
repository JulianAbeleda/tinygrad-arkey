# GEMV G2 Minimal Codegen Representation Scope

## Goal

Turn the G1 generated GEMV skeleton into a generated route that can express the owned/FutureSight lane-partition physical structure without using the custom bridge.

The narrow question for G2 is:

```text
Can generated code express lane = block_group * 8 + word_col for Q4_K gate/up,
then use that lane map to generate packed-word load, dequant, reduction, and store?
```

First target only:

```text
phase = decode T==1
projection = FFN gate/up
shape = 4096 x 12288
profile = Qwen3-8B-Q4_K_M / gfx1100 / wave32
```

G2 is not a full generic layout system. It is the smallest proof needed to decide whether the current GEMV purity blocker is local and solvable, or whether tinygrad codegen lacks a required representation.

## Current Facts

| Item | Current state |
|---|---|
| Fast selected route | BubbleBeam/FutureSight selects the lane-partition Q4_K GEMV bridge |
| Purity verdict | `GEMV_NOT_PURE__SEARCH_SELECTED_CUSTOM_BRIDGE` |
| Fast bridge W==D | ~103.7 / 101.7 / 99.4 / 94.5 tok/s @ctx512/1024/2048/4096 |
| Generated skeleton | `q4k_gemv_generated_skeleton`, env `Q4K_GEMV_SCHEDULER=2` |
| Skeleton verdict | route-clean and token-correct, but slow |
| Skeleton W==D | ~22.5 / 22.4 / 22.2 / 22.0 tok/s @ctx512/1024/2048/4096 |

The G1 skeleton proves attribution can distinguish owned, custom bridge, and generated-route arms. It does not prove the generated route can express the physical structure that makes the bridge fast.

## Required Physical Structure

The generated route must represent these pieces without calling `extra/qk_q4k_lane_partition_gemv.py`:

- Lane ownership: `lane = block_group * 8 + word_col`.
- Packed-word indexing: adjacent lanes own adjacent Q4_K packed uint32 words.
- In-register dequant: one packed word feeds eight nibble contributions.
- Block-group K split: lanes cover a block-group and word-column decomposition, not a scalar row loop.
- Generated reduction: partial sums reduce through generated codegen primitives.
- Single output store per row.

A route that is correct but scalarized or looped through the wrong physical layout should remain `SEARCH_GENERATED_WD_FAIL`, not a promotion candidate.

## Execution Phases

### G2.0: Static representation probe

Status: complete. `extra/qk_gemv_g2_representation_probe.py` emits `bench/qk-gemv-g2-representation-probe/latest.json` with verdict `G2_REPRESENTATION_PROBE_PASS`.

Result: the existing UOp/RANGE algebra can express `lane = block_group * 8 + word_col` and the Q4_K packed-word index with unit stride across `word_col`. The blocker is therefore not local address algebra; the next blocker is binding this representation into generated load/dequant/reduce/store code without the lane-partition custom bridge.

Goal: prove the packed-word address math is expressible before touching runtime routing.

Build:

- Add a narrow checker, preferably `extra/qk_gemv_g2_representation_probe.py`.
- Construct the lane algebra directly:

```text
lane = 0..31
block_group = lane // 8
word_col = lane % 8
blk = block_group * blocks_per_group + local_block
base = (row * k_blocks + blk) * 36
word_idx = base + 4 + (group // 2) * 8 + word_col
```

- Assert the generated expression preserves stride-1 packed-word access across `word_col`.
- Reuse existing layout/coalescing helpers where possible, such as `extra/qk_layout_coalesce_check.py`, rather than inventing a second analysis vocabulary.

Gate:

- Emit `bench/qk-gemv-g2-representation-probe/latest.json`.
- Verdict is one of:

```text
G2_REPRESENTATION_PROBE_PASS
G2_REPRESENTATION_PROBE_FAIL
SEARCH_BLOCKED_BY_CODEGEN
```

Kill:

- If this requires a broad `add_gpudims` rewrite before proving the Q4_K lane map, stop and classify the blocker as:

```text
SEARCH_BLOCKED_BY_CODEGEN: RANGE/AxisType cannot express lane-partition reduce split locally
```

### G2.1: Minimal Q4_K LaneMap object

Goal: create a minimal representation for the gate/up route without generalizing all tensor-core swizzles.

Build:

- Add a Q4_K-specific LaneMap object or helper, preferably isolated under `extra/` until the representation proves value.
- Represent `lane -> (block_group, word_col)` explicitly.
- Keep the object serializable into the search-space manifest so BubbleBeam can later bind it as a candidate primitive.

Gate:

- The LaneMap reproduces the owned bridge's packed-word ownership for the target shape.
- It does not depend on the custom bridge implementation to compute ownership.

### G2.2: Generated packed-address builder

Goal: produce the packed-word address expression from the LaneMap.

Build:

- Add a generated address/index builder for Q4_K packed words.
- Compare generated indices against the owned/FutureSight bridge index formula for sampled rows, block groups, and word columns.
- Keep this a structural/index test first; do not require model execution in this step.

Gate:

- Index equality passes for the target gate/up shape.
- Stride/coalescing assertions pass for adjacent word columns.

### G2.3: Generated dequant skeleton

Goal: replace the current slow G1 generated skeleton with a generated route that uses the LaneMap and packed-address builder.

Build:

- Add a default-off generated route mode, either a new `Q4K_GEMV_SCHEDULER` value or a separate explicit env.
- Generate packed-word load, nibble extraction, scale/min handling, accumulation, reduction, and store.
- Do not call the lane-partition bridge.

Gate:

- Tokens match the owned/bridge reference.
- Route attribution proves no owned warp GEMV and no lane-partition bridge.
- If correct but still near the G1 skeleton ceiling, classify as `SEARCH_GENERATED_WD_FAIL`.

### G2.4: Purity gate integration

Goal: make the existing gate report the G2 generated route clearly.

Build:

- Extend `extra/qk_gemv_purity_gate.py` only if the current generated-arm reporting is insufficient.
- Preserve the existing FutureSight bridge verdict until BubbleBeam selects a truly generated route.

Gate:

- `extra/qk_validate_search_provenance.py` passes.
- `extra/qk_gemv_purity_gate.py` reports the generated route separately from owned and bridge routes.

### G2.5: W==D decision

Goal: decide whether the representation is promotable, locally useful, or blocked.

Run:

- Interleaved W==D across ctx 512/1024/2048/4096.
- Compare owned, FutureSight bridge, G1 skeleton, and G2 generated route.

Decision labels:

```text
GEMV_PURE_SEARCH_GENERATED
SEARCH_GENERATED_WD_FAIL
SEARCH_BLOCKED_BY_CODEGEN
SEARCH_BLOCKED_BY_RUNTIME
```

Promotion requires:

- Token correctness.
- Generated-only route attribution.
- W==D within the chosen promotion threshold of the owned/FutureSight route.
- Candidate provenance marked `search_generated` only after the route no longer uses manual/custom flags or bridge programs.

## Non-Goals

- Do not solve attention TILE+COMBINE.
- Do not solve prefill.
- Do not generalize tensor-core LayoutFn/LaneMap before the Q4_K gate/up proof.
- Do not mark FutureSight pure while it routes through the lane-partition bridge.
- Do not rename BubbleBeam/FutureSight back to upstream BEAM terminology.

## Completion Criteria

G2 is complete when one of these is true:

1. The generated Q4_K gate/up route is token-correct, route-clean, and fast enough to promote.
2. The generated route is route-clean and token-correct but too slow, with W==D evidence proving `SEARCH_GENERATED_WD_FAIL`.
3. The representation cannot be expressed locally, with a precise `SEARCH_BLOCKED_BY_CODEGEN` artifact naming the missing primitive.
