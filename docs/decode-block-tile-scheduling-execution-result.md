# Decode block-tile scheduling execution result

Date: 2026-06-26

Source prompt: `docs/decode-block-tile-scheduling-codex-prompt.md`

## Verdict

The generated block-tile route is structurally correct, but the remaining decode gap is not solved by the cheap occupancy fix or by the inline-reduce codegen experiment.

Current best label:

`SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING`

This is now better localized: the gap appears inside the generated tile's emitted code/economics, not in route purity, materialization, missing LDS staging, missing wide loads, or host/runtime overhead.

## STEP 0 — isolated per-kernel timing

New tool:

- `extra/qk_decode_block_tile_isolated_timing.py`

Artifact:

- `bench/qk-decode-block-tile-isolated-timing/latest.json`

Method: eager `custom_kernel` + `DEBUG=2`, tile kernel only. The parser records the per-kernel `tm Xus/` field, not the cumulative time after the slash.

| ctx | generated block tile | owned tile | generated / owned |
|---:|---:|---:|---:|
| 512 | 1024.04 us | 7.78 us | 131.6x |
| 4096 | 7375.04 us | 30.98 us | 238.1x |

Interpretation: the full-route long-context gap is not primarily Python/runtime/combine overhead. The generated attention tile itself is far slower than the owned HIP tile at matched-ish long-context occupancy.

## STEP 1 — H2 occupancy experiment

Implemented default-off split override inside the block-tile route:

- `DECODE_ATTN_BLOCK_TILE_FIXED_S=1`
- `DECODE_ATTN_BLOCK_TILE_L=<concrete L>`
- `DECODE_ATTN_FUSED_XLANE_SCORE_PV_S=48`

For ctx512, tested `S=48,L=11`.

Gate result:

- Route clean: PASS
- ISA vec: `ISA_VEC_AUTHORITATIVE_PASS`
- Occupancy: 384 workgroups, 4.0 wg/CU

W==D ctx512 result:

| variant | ctx512 tok/s |
|---|---:|
| prior generated block tile | 19.0 |
| fixed `S=48,L=11` | 14.9 |
| owned/baseline | 103.5 |

Interpretation: increasing short-context occupancy by forcing 48 tiny splits worsens throughput. H2 is not the main lever and should not be promoted.

## STEP 2 — H1 reducer/dataflow experiment

Implemented default-off inline reducer experiment:

- `DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE=1`

Intent: replace the staged generated reducer with the inline AMD warp sum and move ISA `cross_lane` from 10 toward owned's 5.

Gate results:

| Gate | Result |
|---|---:|
| microgate | `BLOCK_TILE_MICROGATE_PASS` |
| route clean | `FUSED_XLANE_SCORE_PV_ROUTE_CLEAN__ECONOMICS_NEXT` |
| ISA vec | `ISA_VEC_AUTHORITATIVE_PASS` |
| marker movement | FAIL |

ISA comparison:

| marker | baseline block tile | inline-reduce experiment | desired |
|---|---:|---:|---:|
| cross_lane | 10 | 10 | toward 5 |
| ds_read | 3 | 4 | not up |
| s_waitcnt | 32 | 36 | not up |
| v_dot2 | 3 | 3 | unchanged |
| wide_load_count | 34 | 34 | unchanged |
| LDS bytes | 8192 | 8192 | unchanged |

Interpretation: the inline reducer does not produce the desired ISA movement and slightly worsens wait/read counts. Per the guardrail, do not claim it worked and do not promote it.

## Important correction

The prompt's wording says the owned tile "does NOT reduce per token" and "token-shards the q.k." The actual owned source `extra/qk_owned_flash_decode.hip` still computes an e-sharded q.k per lane and performs a per-token `__shfl_xor` reduction ladder inside `owned_flash_tile_gqa_whole`.

The real observed difference is not zero-reduce vs per-token-reduce. It is emitted code quality/economics around the same conceptual operation:

- owned HIP tile: low kernel time, lower static cross-lane marker count, efficient hand-authored loop shape,
- generated UOp tile: same high-level topology but much slower emitted code.

## Stop condition status

Not all theoretical cleanups were exhausted. Specifically, a true alternative token-shard layout and H3 predication cleanup were not promoted.

But the executed gates already show:

- structural generation works,
- route is clean,
- LDS staging works,
- wide loads work,
- occupancy forcing does not help,
- inline reducer does not move the marker,
- isolated tile timing shows a 100x+ tile-level gap.

Practical next action is no longer another attention route. The next useful work is a lower-level codegen/scheduler audit comparing the generated block tile and owned HIP at the hot-loop instruction level.

## Recommended next audit

Build a dedicated owned-vs-generated hot-loop diff tool that extracts:

- `s_waitcnt` placement and dependency distance,
- DS read/write placement relative to fdot2,
- branch structure and loop unrolling,
- exact `ds_bpermute` count and whether extra read/write instructions are generated around it,
- address arithmetic in the inner `tt` loop,
- `v_cndmask` / predication count,
- live VGPR pressure through the online-softmax recurrence,
- whether the generated UOp path prevents comgr from seeing the same scheduling window as the HIP source.

Do not promote `DECODE_ATTN_BLOCK_TILE_FIXED_S` or `DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE` based on current data.
