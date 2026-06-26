# Decode Attention Online-Softmax+PV Tile P4 Codegen Decision Scope

## Goal

Exhaustively decide what the next real implementation move is after P3.

P3 proved the structural online-PV tile route is clean and preserves split-KV parallelism, but it also showed that the route still lacks:

- lane-owned online update of `m`;
- lane-owned online update of `l`;
- cross-lane or equivalent reduction schedule for `m/l/acc[D]`;
- packed-dot score production inside or directly fused with the tile lifecycle.

P4 must not add another metadata fusion. It must classify every plausible lowerable target and decide whether current codegen can bind it now.

## Inputs

- `bench/qk-decode-attention-online-pv-lanemap/latest.json`
- `bench/qk-decode-attention-online-pv-tile/latest.json`
- `bench/qk-search-spaces/decode_attention_online_softmax_pv_tile_v1.json`
- Existing lowerings:
  - `extra/qk_warp_reduce_lowering.py`
  - `extra/qk_fdot2_lowering.py`
  - `extra/qk_lane_partition_reduce.py`
- Prior negative controls:
  - A3.1 vdot2 score: no material transfer
  - A3.2/A3.2b cross-lane: not attention-complete / no transfer
  - A3.6/A3.7 metadata: no transfer
  - A3.9 partial PV: no transfer
  - A3.10 prob+partial: regression

## Exhaustive target matrix

| Target | Current owner | Lowering exists? | Bindable now? | Reason |
|---|---|---|---|---|
| score dot | `flash_score_whole_cache_32_128` | partial `V_DOT2_LOWERING` | no as a P4 win | score remains a separate program; A3.1 already showed no W==D transfer by itself |
| per-split max `m` | `flash_max_32` | cross-lane max lowering exists for lane reductions | no | current max reduce is not inside the online-PV tile lane/dataflow |
| denominator `l` | `flash_den_32` plus denominator lane contribution | cross-lane add lowering exists for lane reductions | no | global denominator is still a separate program; tile only outputs partial denominator contribution |
| PV accumulator `acc[D]` | `flash_online_pv_tile_whole_cache_32_128` | register accumulator exists | partially | accumulator is inside tile, but there is no cross-lane reduction over GQA/lane-owned partials to lower |
| final combine | `flash_combine_32_128` | add/max reductions exist | no as speed target | two-kernel combine was already audited/refuted as bounded speed lever; combine still must be lifecycle-accounted |
| LDS staging | optional | native LDS exists | no as blind next step | decode LDS has prior trap: can reduce occupancy or duplicate cache-served reads unless tied to full dataflow rewrite |

## Decision rule

P4 should produce exactly one of:

```text
ONLINE_PV_TILE_P4_BINDABLE_CODEGEN_TARGET_FOUND
ONLINE_PV_TILE_P4_NEEDS_DATAFLOW_REWRITE_BEFORE_CODEGEN
ONLINE_PV_TILE_P4_BLOCKED_BY_CODEGEN
ONLINE_PV_TILE_P4_NO_ACTION_PRIOR_NEGATIVE_CONTROLS
```

Definitions:

- `BINDABLE_CODEGEN_TARGET_FOUND`: a missing primitive has a current UOp site that can be changed directly and is not already refuted.
- `NEEDS_DATAFLOW_REWRITE_BEFORE_CODEGEN`: lowerings exist, but the current route has no correct site where they can attach to create the primitive-complete tile.
- `BLOCKED_BY_CODEGEN`: the dataflow site exists, but renderer/lowering cannot emit the required primitive.
- `NO_ACTION_PRIOR_NEGATIVE_CONTROLS`: every plausible direct move has already been tried and refuted.

## Pass criteria

P4 is complete when it records:

- target matrix;
- current route signature;
- lowerings available in repo;
- prior negative controls;
- exact next implementation scope.

## Expected next implementation if P4 chooses dataflow rewrite

Create P5 scope for an online-state tile rewrite:

```text
flash_online_state_pv_tile_whole_cache_32_128
```

Required change:

- move per-split `m` and `l` update into the online-PV tile lifecycle;
- keep `Hkv*S` workgroups and `Hd+1` local lanes;
- keep whole-cache identity and no `E_49152`;
- then attempt cross-lane/packed-dot lowering against real in-tile reduction sites.
