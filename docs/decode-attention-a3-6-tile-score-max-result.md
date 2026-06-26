# Decode Attention A3.6 Tile Score+Max Result

## Verdict

`A3_6_TILE_SCORE_MAX_NO_TRANSFER`

A3.6 replaces the A3.5 identity tile placeholder with the smallest compile-safe generated tile payload.

The candidate `flash_tile_score_max_32_128` owns per-split max metadata and replaces the separate `flash_max_32` program. The score buffer remains produced by `flash_score_whole_cache_32_128` for compatibility with the existing probability and partial paths.

## Files

- Gate: `extra/qk_decode_attention_a3_6_tile_score_max_gate.py`
- Artifact: `bench/qk-decode-attention-a3-6-tile-score-max/latest.json`

## Command

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_a3_6_tile_score_max_gate.py
```

## Gate Result

| Check | Result |
|---|---|
| A2 generated whole-cache route clean | pass |
| owned tile/combine absent | pass |
| `E_49152` absent | pass |
| token sample matches owned baseline | pass |
| generated tile program present | pass: `flash_tile_score_max_32_128` |
| separate `flash_max_32` removed | pass |
| generated combine present | pass: `flash_combine_32_128` |
| full TILE+COMBINE bundle bound | pass |
| W==D transfer over A2 | fail |

## W==D Result

| ctx | owned tok/s | A2 generated tok/s | A3.6 tile-max tok/s | A3.6 vs A2 | A3.6 vs owned |
|---:|---:|---:|---:|---:|---:|
| 512 | 104.9 | 78.1 | 78.0 | 99.9% | 74.4% |
| 1024 | 103.0 | 75.4 | 75.3 | 99.9% | 73.1% |
| 2048 | 100.5 | 69.5 | 69.5 | 100.0% | 69.2% |
| 4096 | 95.6 | 60.4 | 60.2 | 99.7% | 63.0% |

## Captured A3.6 Bundle Signature

| Program class | Programs | Status |
|---|---|---|
| score | `flash_score_whole_cache_32_128` | present |
| tile/metadata | `flash_tile_score_max_32_128` | present |
| separate max | `flash_max_32` | absent |
| probability metadata | `flash_prob_32`, `flash_gmax_32`, `flash_den_32` | present |
| partial | `flash_partial_coop_vec_whole_cache_32_128` | present |
| combine | `flash_combine_32_128` | present |

## Implementation Note

The first attempted A3.6 payload fused score and max into a multi-output custom kernel. That hit a UOp multi-output shape limitation: the grouped output could not be used as a shaped tensor by the next kernel. The committed A3.6 payload is therefore the compile-safe minimal real payload: keep score generation separate, but replace `flash_max_32` with a generated tile-named max program.

## Interpretation

A3.6 proves the lifecycle bundle can bind a real downstream responsibility, not just an identity placeholder.

It also shows that moving only the per-split max responsibility into a tile-named program is not enough. Throughput is essentially flat versus A2 and still far below owned attention. The gap is not the standalone max kernel; it is the larger tile lifecycle: score, online softmax state, partial PV, and likely cross-lane/LDS scheduling need to move together.

## Decision

Promote nothing from A3.6.

Next executable step: A3.7 should target a larger real tile payload: either score+probability metadata or partial PV production. The goal is to remove a meaningful downstream memory pass, not just rename or isolate metadata.
