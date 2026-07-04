# Packed Prefill Generated-Tile Scope

This backlog is derived from a BoltBeam practical roofline report. It ranks tinygrad work by reclaimable
pp512 time against llama's measured Q4 packed-matmul rate, not by broad code ownership.

## Conclusion

The first fix is schedule selection on the existing Q4_K direct-output route. The original trace made the path look
like a bandwidth problem, but the useful interpretation is dequant amortization: when token/row work is not register
tiled, Q4_K unpack/dequant is repeated too often. A correct `LOCAL:0:16,LOCAL:1:16,UPCAST:0:4,UPCAST:1:4` schedule
moves 14B pp512 from `135.7` to `173.6 tok/s`.

The remaining gap is still the packed-prefill matmul substrate, but it is narrower: tinygrad needs a correct grouped
or staged reduction that preserves the 4x4 register tile and amortizes dequant over a larger token tile. Naive
`GROUP` on the current custom UOp body is fast but wrong.

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

## 2026-07-04 Candidate Results

### Promoted Schedule

Fable's audit corrected the framing: the current route was effectively too close to 512 independent GEMVs because
dequant work was not sufficiently amortized across tokens. The safe schedule change is:

```text
LOCAL:0:16, LOCAL:1:16, UPCAST:0:4, UPCAST:1:4
```

This is now the default Q4_K direct-packed prefill schedule. Rollback:

```text
PREFILL_Q4K_DIRECT_SCHEDULE=legacy
```

Clean 14B pp512 timing:

| route | pp512 tok/s | elapsed us | verdict |
|---|---:|---:|---|
| old Q4 direct-packed default | 135.7 | 3772608.7 | baseline |
| Q4 4x4 register-tiled schedule | 173.6 | 2950068.1 | promoted |

The tempting grouped schedule:

```text
LOCAL:0:64, GROUP:0:10, UPCAST:1:4
```

looked much faster (`~214 tok/s` when applied to `ffn_gate_up`, `~275 tok/s` when applied to K=5120 Q4 roles), but it
is numerically invalid on real 14B `blk.0.ffn_gate`: `rel_rmse ~= 1.26`. Do not use `GROUP` on this direct-output Q4
custom UOp until the grouped reduction semantics are fixed.

### Refuted Cooperative-Lane Probes

The first generated-UOp cooperative-lane probes are correct but refuted for speed on 14B `ffn_gate_up`.

| candidate | tile | output | whole pp512 tok/s | ffn_gate_up GB/s | verdict |
|---|---|---|---:|---:|---|
| current direct-packed floor | current | direct `[tokens, rows]` | 135.7 | 2.11 | baseline |
| generated tile | rows=4, tokens=8, lanes=8 | external 8-lane partial reduce | 79.2 | 0.99 | refuted |
| generated direct-warp | rows=1, tokens=4, lanes=8 | in-kernel warp reduce | 98.4 | 1.29 | refuted |
| generated direct-warp | rows=2, tokens=2, lanes=8 | in-kernel warp reduce | 86.7 | 1.05 | refuted |
| generated direct-warp | rows=4, tokens=1, lanes=8 | in-kernel warp reduce | 83.5 | 1.00 | refuted |

Correctness for the best direct-warp mode passed against the existing lossless direct-packed route on real 14B
`blk.0.ffn_gate`: `rel_rmse=1.64e-6`, `max_abs=3.81e-5`.

This exhausts the "simple generated UOp cooperative lane" family for the 14B hot row. The failure mode is clear:
external lane partials add too much lifecycle, while a one-wave in-kernel combine removes that lifecycle but loses too
much row/token tile throughput.

## Next Work

1. Fix or replace grouped reduction for the direct-output Q4 custom UOp so that a grouped K-superblock schedule is
   numerically correct.
2. Re-test the fast-but-wrong `GROUP:0:10` family after the reduction fix; the measured speed suggests the schedule
   shape is valuable if semantics can be made correct.
3. Only after correct grouping plateaus, add the dequant-to-fp16-LDS prologue feeding WMMA or an int8/dot path.
