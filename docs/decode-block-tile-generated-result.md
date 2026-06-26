# Decode generated block-tile result

Date: 2026-06-26

## Verdict

The generated decode tile can now express the owned-kernel topology far enough to pass the ordered gates:

- `BLOCK_TILE_MICROGATE_PASS`
- `FUSED_XLANE_SCORE_PV_ROUTE_CLEAN__ECONOMICS_NEXT`
- `ISA_VEC_AUTHORITATIVE_PASS`

This resolves the previous narrow blocker that the generated path could not express a 4-warp, TK=16, LDS-staged decode tile. It does **not** reach owned decode throughput. The remaining blocker is performance transfer/economics, not correctness, route purity, materialization, LDS size, or wide-load ISA.

## Implemented pieces

- `extra/qk_decode_isa_vectorization_gate.py`
  - Authoritative ISA vectorization gate.
  - Counts RDNA3 `global_load_b128|b96|b64|b32` plus existing `d16/dword*` markers.
  - Stores route cleanliness, numeric correctness, full marker dict, LDS bytes, and disassembly artifact.

- `extra/qk_decode_attention_block_tile_microgate.py`
  - Proves the generated 128-thread block tile numerically against a NumPy oracle.
  - Uses fp16-staged K/V oracle semantics to match the tile contract.

- `extra/qk_flash_decode.py`
  - Adds `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`.
  - Keeps it default-off behind existing `DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1` plus `DECODE_ATTN_BLOCK_TILE=1`.

- `extra/qk_decode_attention_fused_xlane_score_pv_route_gate.py`
  - Recognizes the block-tile target when `DECODE_ATTN_BLOCK_TILE=1`.

## Gate artifacts

- `bench/qk-decode-attention-block-tile-microgate/latest.json`
- `bench/qk-decode-attention-fused-xlane-score-pv-route/latest.json`
- `bench/qk-decode-isa-vectorization/latest.json`
- `bench/qk-decode-runtime-overhead/result.json`

## Ordered gate results

| Gate | Result | Key evidence |
|---|---:|---|
| Block microgate | PASS | max_abs <= 2.29e-05, rel_rmse <= 1.34e-07 across Tc 32/128/130/256 |
| Route cleanliness | PASS | token match, materialization clean, owned absent, generated block tile present |
| ISA vectorization | PASS | LDS 8192B, wide_load_count 34, scratch 0 |
| W==D transfer | PARTIAL | ctx512+ improves vs prior generated tile, but remains far below owned baseline |

## ISA comparison: previous generated tile vs block tile

| Marker | Previous generated xlane | Generated block tile | Direction |
|---|---:|---:|---|
| LDS bytes | 256 | 8192 | fixed toward owned shape |
| wide load count | 10 | 34 | improved |
| `global_load_d16` | 0 | 32 | fixed |
| `global_load_b64` | 10 | 2 | still present |
| cross-lane ops | 20 | 10 | improved, not owned-level |
| scratch | 0 | 0 | clean |
| VGPR | 80 | 56 | improved |
| `s_barrier` | 0 | 1 | expected for TK staging |

## W==D result

| ctx | Previous generated tok/s | Block-tile generated tok/s | Owned/baseline tok/s | Block vs previous | Block vs owned |
|---:|---:|---:|---:|---:|---:|
| 128 | 82.7 | 82.4 | 82.4 | -0.4% | 100.0% |
| 512 | 7.2 | 19.0 | 103.5 | +163.9% | 18.4% |
| 1024 | 4.1 | 11.8 | 101.8 | +187.8% | 11.6% |
| 4096 | 1.1 | 3.5 | 94.6 | +218.2% | 3.7% |

Runtime conclusion from `qk_decode_runtime_overhead.py`: GPU-bound, host-sync median 2.6% of wall. Runtime/host overhead is not the main blocker.

## Interpretation

The generated path crossed the main representational fork:

- 4-warp workgroup is expressible.
- TK=16 block staging is expressible.
- 8192B LDS staging is present in ISA.
- Wide-load markers are present in ISA.
- Route is clean and byte/token-compatible with the owned route.

The performance gap therefore moves from "cannot express the tile" to "expressed tile does not schedule/economically behave like the owned HIP tile." The honest next label is:

`SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING`

Use this label narrowly: the generated UOp tile is correct and structurally close, but comgr/LLVM output still does not transfer to owned throughput.

## Next action

Do not write another attention layout. The next useful step is an ISA/dynamic scheduling audit between `owned_flash_tile_gqa_whole` and `flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` focused on why the structurally-correct generated tile remains slow:

- instruction scheduling around LDS load/use and `s_waitcnt`,
- DS read/write mix and barrier placement,
- block loop unrolling differences,
- fdot2 count/placement,
- occupancy/resource limits from LDS/VGPR/SGPR,
- whether the generated tile emits extra scalar control or address arithmetic in the hot loop.
