# Packed Prefill Generated-Tile Scope

This backlog is derived from a BoltBeam practical roofline report. It ranks tinygrad work by reclaimable
pp512 time against llama's measured Q4 packed-matmul rate, not by broad code ownership.

## Conclusion

The next tinygrad work is a generated packed-prefill tile route. The current Q4_K direct-output path is a
fit/safety floor, but its topology is the bottleneck: 32/64-thread workgroups, no LDS, no scratch, and a
Q4 lane axis reduced inside the output element. The route needs an explicit cooperative token/row/lane tile.

## Generated Schedule Requirements

- Route family: `generated_packed_prefill_tile`
- Default state: `off` via `PREFILL_QK_GENERATED_TILE=1`
- Strict gate: `PREFILL_ROUTE_STRICT=1` must fail on hidden fallback
- Current axes: row GLOBAL, token mostly UPCAST/serial, q4 lane4 REDUCE, kblock REDUCE
- Target axes: row tile GLOBAL/LOCAL, token tile GLOBAL/LOCAL, q4 lane4 LOCAL/cooperative, kblock REDUCE
- First shape: `[512,17408,5120]`

## Ranked Work

| priority | role | quant | shape | current us | current GB/s | target GB/s | reclaim us | launch resources | route id |
|---:|---|---|---|---:|---:|---:|---:|---|---|
| 1 | ffn_gate_up | Q4_K | `[512,17408,5120]` | 1901133.712 | 2.110 | 25.999 | 1746865.600 | global=(32, 272, 1), local=(64, 1, 1), threads=64, vgpr=185, sgpr=16, lds=0, scratch=0 | `prefill_q4_k_generated_tile_ffn_gate_up_512_17408_5120` |
| 2 | ffn_down | Q4_K | `[512,5120,17408]` | 556040.883 | 1.803 | 25.999 | 517473.855 | global=(32, 160, 1), local=(32, 1, 1), threads=32, vgpr=185, sgpr=16, lds=0, scratch=0 | `prefill_q4_k_generated_tile_ffn_down_512_5120_17408` |
| 3 | attn_qo | Q4_K | `[512,5120,5120]` | 464256.614 | 2.541 | 25.999 | 418883.640 | global=(32, 80, 1), local=(64, 1, 1), threads=64, vgpr=185, sgpr=16, lds=0, scratch=0 | `prefill_q4_k_generated_tile_attn_qo_512_5120_5120` |
| 4 | attn_kv | Q4_K | `[512,1024,5120]` | 82479.051 | 2.145 | 25.999 | 75673.105 | global=(32, 16, 1), local=(64, 1, 1), threads=64, vgpr=177, sgpr=16, lds=0, scratch=0 | `prefill_q4_k_generated_tile_attn_kv_512_1024_5120` |
| 5 | ffn_down | Q6_K | `[512,5120,17408]` | 648188.161 | 2.256 |  |  | global=(32, 80, 1), local=(64, 1, 1), threads=64, vgpr=225, sgpr=16, lds=0, scratch=0 | `prefill_q6_k_generated_tile_ffn_down_512_5120_17408` |
| 6 | attn_kv | Q6_K | `[512,1024,5120]` | 38153.247 | 2.254 |  |  | global=(32, 32, 1), local=(32, 1, 1), threads=32, vgpr=209, sgpr=16, lds=0, scratch=0 | `prefill_q6_k_generated_tile_attn_kv_512_1024_5120` |

## Implementation Path

1. Add a `PackedPrefillTileSpec` data object for Q4_K with row tile, token tile, lane tile, k-block policy,
   accumulator dtype, output layout, and strict role/shape guards.
2. Lower that spec through a generated UOp emitter. The first emitter should keep lossless fp32 accumulation and
   direct `[tokens, rows]` output; an external lane-partial probe is acceptable only as a short-lived microgate.
3. Wire `tinygrad/llm/prefill_routes.py` behind `PREFILL_QK_GENERATED_TILE=1`, with tensor-role filters so the
   first target can be only `ffn_gate_up`.
4. Add route-manifest metadata with provenance `machine_authored_generated` once the emitter is spec-driven.
5. Gate ffn_gate_up first, then attn_qo and ffn_down Q4_K. Add Q6_K only after the Q4 topology moves.

## Exhaustion Rule

Close a candidate quickly if the bound hot-row kernel stays in the ~2 GB/s class. Continue only when the generated
tile changes the substrate class, visible as wider workgroups/cooperative lanes and a multi-x per-kernel GB/s move.

