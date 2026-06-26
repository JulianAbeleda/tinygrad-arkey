# Decode Attention A3.10 Tile Prob+Partial-PV Result

## Verdict

`A3_10_TILE_PROB_PARTIAL_PV_NO_TRANSFER`

A3.10 fused probability generation with partial PV production.

Candidate:

- `flash_tile_prob_partial_pv_whole_cache_32_128`

It removed both:

- `flash_prob_32`
- `flash_partial_coop_vec_whole_cache_32_128`

while preserving A2 whole-cache/no-`E_49152` hygiene.

## Files

- Gate: `extra/qk_decode_attention_a3_10_tile_prob_partial_pv_gate.py`
- Artifact: `bench/qk-decode-attention-a3-10-tile-prob-partial-pv/latest.json`

## Command

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_a3_10_tile_prob_partial_pv_gate.py
```

## Gate Result

| Check | Result |
|---|---|
| A2 generated whole-cache route clean | pass |
| owned tile/combine absent | pass |
| `E_49152` absent | pass |
| token sample matches owned baseline | pass |
| `flash_tile_prob_partial_pv_whole_cache_32_128` present | pass |
| old `flash_prob_32` absent | pass |
| old `flash_partial_coop_vec_whole_cache_32_128` absent | pass |
| full TILE+COMBINE bundle bound | pass |
| material W==D transfer over A2 | fail |

## W==D Result

| ctx | owned tok/s | A2 generated tok/s | A3.10 prob+partial tok/s | A3.10 vs A2 | A3.10 vs owned |
|---:|---:|---:|---:|---:|---:|
| 512 | 104.8 | 78.4 | 72.6 | 92.6% | 69.3% |
| 1024 | 103.1 | 75.7 | 70.7 | 93.4% | 68.6% |
| 2048 | 100.7 | 69.8 | 66.2 | 94.8% | 65.7% |
| 4096 | 95.7 | 60.6 | 58.8 | 97.0% | 61.4% |

## Captured A3.10 Bundle Signature

| Program class | Programs | Status |
|---|---|---|
| score | `flash_score_whole_cache_32_128` | present |
| max | `flash_max_32` | present |
| fused prob+partial | `flash_tile_prob_partial_pv_whole_cache_32_128` | present |
| old prob | `flash_prob_32` | absent |
| old partial | `flash_partial_coop_vec_whole_cache_32_128` | absent |
| global metadata | `flash_gmax_32`, `flash_den_32` | present |
| combine | `flash_combine_32_128` | present |

## Interpretation

A3.10 is the first true fused producer-consumer payload, and it is worse than A2.

This means the simple generated fusion shape is not equivalent to the owned tile lifecycle. It removes two launches/intermediates, but the fused kernel likely loses parallelism/memory shape enough to dominate the saved lifecycle cost.

The likely blocker is not merely too many metadata/partial kernels. The missing primitive is an owned-style online-softmax + PV tile shape: cooperative reductions, correct lane ownership, and possibly LDS/cross-lane scheduling. Incremental fusion without those primitives is not enough.

## Decision

Promote nothing from A3.10.

Next executable step: A3.11 may test score+prob+partial, but the A3.10 regression is strong enough that the better next move is to scope the primitive-complete online-softmax+PV tile path unless we specifically want to exhaust the incremental sequence.
