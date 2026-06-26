# Decode Attention A3.5 Minimal Tile Placeholder Result

## Verdict

`A3_5_TILE_PLACEHOLDER_NO_TRANSFER`

A3.5 binds the smallest generated `flash_*tile*` program into the A2 whole-cache decode attention route so the TILE+COMBINE lifecycle gate can move past `ROUTE_BINDING_MISSING`.

This is not a fast tile. It is a correctness and lifecycle-binding placeholder:

- preserve A2 whole-cache/no-`E_49152` hygiene
- keep owned tile/combine off
- emit a named generated tile program
- keep tokens identical
- allow W==D to classify transfer/no-transfer

## Files

- Gate: `extra/qk_decode_attention_a3_5_tile_placeholder_gate.py`
- Artifact: `bench/qk-decode-attention-a3-5-tile-placeholder/latest.json`

## Command

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_a3_5_tile_placeholder_gate.py
```

## Gate Result

| Check | Result |
|---|---|
| A2 generated whole-cache route clean | pass |
| owned tile/combine absent | pass |
| `E_49152` absent | pass |
| token sample matches owned baseline | pass |
| generated tile program present | pass: `flash_tile_placeholder_32_128` |
| generated combine program present | pass: `flash_combine_32_128` |
| full TILE+COMBINE bundle bound | pass |
| W==D transfer over A2 | fail |

## W==D Result

| ctx | owned tok/s | A2 generated tok/s | A3.5 placeholder tok/s | A3.5 vs A2 | A3.5 vs owned |
|---:|---:|---:|---:|---:|---:|
| 512 | 105.0 | 78.5 | 77.6 | 98.9% | 73.9% |
| 1024 | 103.3 | 75.9 | 74.5 | 98.2% | 72.1% |
| 2048 | 100.7 | 69.9 | 68.0 | 97.3% | 67.5% |
| 4096 | 95.8 | 60.7 | 58.1 | 95.7% | 60.6% |

## Captured A3.5 Bundle Signature

| Program class | Programs | Status |
|---|---|---|
| score | `flash_score_whole_cache_32_128` | present |
| tile | `flash_tile_placeholder_32_128` | present |
| metadata | `flash_max_32`, `flash_prob_32`, `flash_gmax_32`, `flash_den_32` | present |
| partial | `flash_partial_coop_vec_whole_cache_32_128` | present |
| combine | `flash_combine_32_128` | present |

## Interpretation

A3.5 is useful even though it is slower.

It proves the lifecycle gate can now distinguish three states:

1. A3.4: no generated tile route bound.
2. A3.5: generated tile route bound but no transfer.
3. Future A3.6: real generated tile route bound and measured against A2/owned.

The current placeholder is just an identity copy over the score buffer. The slowdown is expected because it adds an extra generated program and memory pass without replacing any work.

## Decision

Promote nothing from A3.5.

Next executable step: A3.6 should replace the identity placeholder with the smallest real generated tile payload. The first useful payload should fuse at least score plus one downstream lifecycle responsibility, such as local max/metadata or partial PV production, while preserving the same A2 hygiene and bundle attribution.
