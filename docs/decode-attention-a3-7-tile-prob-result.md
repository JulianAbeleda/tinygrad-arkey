# Decode Attention A3.7 Tile Probability Result

## Verdict

`A3_7_TILE_PROB_NO_TRANSFER`

A3.7 expands the real generated tile payload beyond max-only metadata.

The split tile-prob route uses two tile-named generated programs:

- `flash_tile_score_max_32_128`: replaces `flash_max_32`
- `flash_tile_prob_32_128`: replaces `flash_prob_32`

This avoids the multi-output grouped-shape limitation hit during A3.6 while still removing two downstream metadata programs from the A2 route.

## Files

- Gate: `extra/qk_decode_attention_a3_7_tile_prob_gate.py`
- Artifact: `bench/qk-decode-attention-a3-7-tile-prob/latest.json`

## Command

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_a3_7_tile_prob_gate.py
```

## Gate Result

| Check | Result |
|---|---|
| A2 generated whole-cache route clean | pass |
| owned tile/combine absent | pass |
| `E_49152` absent | pass |
| token sample matches owned baseline | pass |
| `flash_tile_score_max_32_128` present | pass |
| `flash_tile_prob_32_128` present | pass |
| separate `flash_max_32` removed | pass |
| separate `flash_prob_32` removed | pass |
| full TILE+COMBINE bundle bound | pass |
| W==D transfer over A2 | fail |

## W==D Result

| ctx | owned tok/s | A2 generated tok/s | A3.7 tile-prob tok/s | A3.7 vs A2 | A3.7 vs owned |
|---:|---:|---:|---:|---:|---:|
| 512 | 104.9 | 78.5 | 76.6 | 97.6% | 73.0% |
| 1024 | 103.1 | 75.8 | 74.4 | 98.2% | 72.2% |
| 2048 | 100.6 | 69.9 | 69.4 | 99.3% | 69.0% |
| 4096 | 95.7 | 60.7 | 61.3 | 101.0% | 64.1% |

## Captured A3.7 Bundle Signature

| Program class | Programs | Status |
|---|---|---|
| score | `flash_score_whole_cache_32_128` | present |
| tile metadata | `flash_tile_score_max_32_128`, `flash_tile_prob_32_128` | present |
| separate max/prob | `flash_max_32`, `flash_prob_32` | absent |
| global metadata | `flash_gmax_32`, `flash_den_32` | present |
| partial | `flash_partial_coop_vec_whole_cache_32_128` | present |
| combine | `flash_combine_32_128` | present |

## Interpretation

A3.7 proves the metadata pair is not the main performance gap.

It successfully removes both separate max and prob kernels while keeping route/materialization/correctness clean. Throughput regresses at short/mid contexts and only has a small long-context uptick at ctx4096. That is not enough to promote and indicates the remaining gap is dominated by heavier lifecycle stages: score memory traffic, partial PV production, or the absence of an owned-style online-softmax+PV tile.

## Decision

Promote nothing from A3.7.

Next executable step: A3.8 should target partial PV production. The tile payload must remove or replace `flash_partial_coop_vec_whole_cache_32_128`, because metadata-only tile work has now failed to transfer.
