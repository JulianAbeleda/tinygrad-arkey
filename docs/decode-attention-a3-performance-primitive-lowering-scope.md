# Decode Attention A3 Performance Primitive Lowering Scope

## Goal

Turn the A2 lifecycle-clean generated decode-attention skeleton into a performance candidate.

A2 already proves the hard lifecycle properties:

- generated route fires
- owned `owned_flash_tile_gqa_whole` does not fire
- owned `owned_flash_combine` does not fire
- tokens match the owned baseline
- `E_49152` is absent
- selected-route buffer identity is preserved

A3 starts from that baseline and adds/searches performance primitives one at a time. Do not regress lifecycle
cleanliness while chasing speed.

## Current baseline

A2 candidate:

- `decode_attention_generated_wholecache_skeleton`
- flag: `DECODE_ATTN_GENERATED_WHOLECACHE=1`
- tool: `extra/qk_decode_attention_purity_capture.py --a2`
- artifact: `bench/qk-decode-attention-wholecache-skeleton/latest.json`
- verdict: `DECODE_ATTENTION_A2_GENERATED_WHOLECACHE_ROUTE_CLEAN`

Captured A2 generated programs:

- `flash_score_whole_cache_32_128`
- `flash_max_32`
- `flash_prob_32`
- `flash_gmax_32`
- `flash_partial_coop_vec_whole_cache_32_128`
- `flash_den_32`
- `flash_combine_32_128`

Known remaining blockers:

- `v_dot2`
- cross-lane reduction
- LDS-staged tile layout
- TILE+COMBINE lifecycle controls

## Non-goals

- Do not promote A2/A3 until W==D proves it.
- Do not re-enable the owned tile inside a generated candidate.
- Do not accept a local kernel win if `E_49152` returns.
- Do not hand-write a new whole attention kernel and call it pure search.
- Do not chase all primitives at once. Each primitive needs a separate before/after artifact.

## Required measurement before changing performance code

Add an A3 baseline profiler for A2:

- route-clean gate: reuse `extra/qk_decode_attention_purity_capture.py --a2`
- W==D sweep: compare default owned route vs A2 at ctx `512 / 1024 / 2048 / 4096`
- program attribution: per-program timing/counts if available
- ISA attribution for A2 generated programs:
  - count `v_dot2`
  - count LDS ops
  - count cross-lane ops
  - VGPR/scratch/spill

Suggested artifact:

- `bench/qk-decode-attention-a3-baseline/latest.json`

Passing baseline verdict:

- `DECODE_ATTENTION_A3_BASELINE_CAPTURED`

This baseline is not expected to be fast. It tells us which generated program is the largest bottleneck and which
primitive should be attempted first.

## Phase A3.1: whole-cache score primitive

Target program:

- `flash_score_whole_cache_32_128`

Reason:

- A2 computes QK scores from the whole cache using a scalar generated reduction.
- The owned tile uses packed fp16 dot behavior and cross-lane cooperation.
- This is the cleanest place to prove `v_dot2` lowering can attach to the generated route.

Candidate direction:

- Add a generated/codegen primitive or lowering for fp16 pair dot accumulation.
- Keep the whole-cache input contract.
- Preserve program attribution with a stable name, for example:
  - `flash_score_whole_cache_vdot2_32_128`

Gate:

- route-clean gate still passes
- tokens match
- `E_49152` absent
- ISA shows `v_dot2` or the intended dot primitive appears
- W==D or isolated program timing improves enough to justify continuing

Kill:

- If `v_dot2` appears but W==D is flat and program attribution says score is not material, stop this lane.
- If lowering requires owned inline asm for the whole kernel, classify as not pure.
- If generated score requires sliced K inputs and brings back `E_49152`, reject.

Expected verdicts:

- `A3_1_VDOT2_SCORE_TRANSFERS`
- `A3_1_VDOT2_SCORE_NO_TRANSFER`
- `A3_1_BLOCKED_BY_RENDERER`
- `A3_1_FAIL__E_49152_REINTRODUCED`

## Phase A3.2: cross-lane reduction primitive

Target programs:

- `flash_score_whole_cache_*`
- `flash_partial_coop_vec_whole_cache_*`
- potentially `flash_combine_*`

Reason:

- Generated reductions currently use scheduler-visible reductions that do not map to the owned tile's cross-lane
  strategy.
- The owned route uses wave-level cooperation to avoid scalar per-lane duplicate work.

Candidate direction:

- Expose/search a LaneMap-style cross-lane reduction primitive for attention.
- Reuse the GEMV G3 LaneMap lessons where possible, but do not assume the GEMV lane map transfers directly.
- Keep TILE and COMBINE distinguishable in attribution.

Gate:

- route-clean gate still passes
- tokens match
- ISA shows intended cross-lane primitive
- W==D improves or a single target program improves with no lifecycle regression

Kill:

- If cross-lane lowering only works by hand-owned whole-kernel assembly, park it.
- If it increases VGPR/scratch enough to erase the win, record `NO_TRANSFER`.
- If it changes numerics beyond token agreement, require a tighter correctness artifact before continuing.

Expected verdicts:

- `A3_2_CROSS_LANE_TRANSFERS`
- `A3_2_CROSS_LANE_NO_TRANSFER`
- `A3_2_BLOCKED_BY_CODEGEN`

## Phase A3.3: LDS-staged tile layout

Target programs:

- score/partial path, especially K/V reads from whole cache

Reason:

- The owned tile stages K/V through LDS and uses a tile layout designed for reuse.
- A2's whole-cache skeleton is lifecycle-clean but still emits direct generated memory traffic.

Candidate direction:

- Add a generated LDS-staged tile layout candidate.
- The candidate must still read the whole cache buffer directly.
- Search over tile size, split count, local axes, and LDS footprint.

Gate:

- route-clean gate still passes
- tokens match
- `E_49152` absent
- ISA/resource audit shows LDS usage and no spill/scratch regression
- W==D improves over A2

Kill:

- If LDS staging only helps isolated kernels but W==D regresses, classify `NO_TRANSFER`.
- If LDS footprint reduces occupancy below the owned baseline, record the occupancy wall.
- If the scheduler cannot express the tile layout without custom whole-kernel code, classify `SEARCH_BLOCKED_BY_CODEGEN`.

Expected verdicts:

- `A3_3_LDS_TILE_TRANSFERS`
- `A3_3_LDS_TILE_NO_TRANSFER`
- `A3_3_BLOCKED_BY_CODEGEN`

## Phase A3.4: TILE+COMBINE lifecycle controls

Target lifecycle:

- split-KV tile
- per-split metadata
- global combine

Reason:

- Owned attention is two programs: tile plus combine.
- Pure search needs to reason about the pair as one candidate lifecycle, even if codegen still emits multiple programs.

Candidate direction:

- Make the candidate manifest describe TILE+COMBINE together.
- Record split count, combine policy, score/partial/den/combine program counts, and memory intermediates.
- Allow search to compare lifecycle bundles, not isolated kernels.

Gate:

- route-clean gate still passes
- tokens match
- W==D improves
- unknown-bucket/materialization attribution stays closed

Kill:

- If a faster TILE creates a slower COMBINE or extra graph materialization, reject the bundle.
- If isolated timing wins do not transfer to W==D, record `NO_TRANSFER`.

Expected verdicts:

- `A3_4_LIFECYCLE_BUNDLE_TRANSFERS`
- `A3_4_LIFECYCLE_BUNDLE_NO_TRANSFER`
- `A3_4_BLOCKED_BY_RUNTIME`

## Overall promotion gate

A3 is promotable only if all are true:

- `DECODE_ATTN_GENERATED_WHOLECACHE=1` route is selected by BubbleBeam/search, not manually forced.
- Owned tile/combine do not fire.
- `E_49152_present == false`.
- selected-route buffer identity is true.
- Tokens match baseline.
- W==D meets or exceeds the owned-route threshold across ctx points.
- ISA/resource audit shows no hidden spill/scratch wall.
- Search-space manifest records included and excluded primitives.

Promotion verdict:

- `DECODE_ATTENTION_PURE_SEARCH_GENERATED_PROMOTABLE`

If lifecycle is clean but speed does not transfer:

- `DECODE_ATTENTION_A3_LIFECYCLE_CLEAN_SPEED_NOT_PROMOTABLE`

## Execution order

1. Build A3 baseline profiler and artifact.
2. Attempt A3.1 whole-cache score `v_dot2` lowering.
3. Only if A3.1 transfers or the profiler points elsewhere, attempt cross-lane reduction.
4. Only after score/reduction attribution is clear, attempt LDS tile layout.
5. Bundle TILE+COMBINE lifecycle controls last.

## Current recommended next command

Implement step 1:

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_a3_baseline.py
```

Expected output:

- `bench/qk-decode-attention-a3-baseline/latest.json`
- verdict `DECODE_ATTENTION_A3_BASELINE_CAPTURED`
