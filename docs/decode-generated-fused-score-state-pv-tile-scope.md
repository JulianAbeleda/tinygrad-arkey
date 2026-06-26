# Decode generated fused score + online-state + PV tile scope

## Goal

Move generated decode attention from a route-clean but speed-refuted fused-PV-only lifecycle into a generated tile that fuses score computation, online softmax state, and PV accumulation.

Current generated fused-PV-only route:

```text
flash_score_whole_cache_32_128
-> flash_max_32
-> flash_fused_pv_tile_whole_cache_32_128
-> flash_gmax_32
-> flash_den_32
-> flash_combine_32_128
```

Target route:

```text
flash_fused_score_state_pv_tile_whole_cache_32_128
-> flash_state_gmax_32_128
-> flash_state_combine_32_128
```

Longer-term target:

```text
flash_fused_score_state_pv_tile_whole_cache_32_128
-> compact generated combine
```

## Baseline being improved

Latest W==D for the route-clean fused PV tile:

| ctx | owned baseline tok/s | fused PV tile tok/s | delta |
|---:|---:|---:|---:|
| 128 | 82.4 | 82.6 | +0.24% |
| 512 | 103.5 | 72.1 | -30.34% |
| 1024 | 101.8 | 70.1 | -31.14% |
| 4096 | 94.6 | 58.5 | -38.16% |

Interpretation:

The local/cooperative `d` fused PV tile fixed the catastrophic split x-lane wall, but still loses because the generated route keeps separate score, max, gmax, den, and combine lifecycle stages.

## Required candidate builder

```python
flash_fused_score_state_pv_tile_whole_cache_kernel(...)
```

Expected generated program identity:

```text
flash_fused_score_state_pv_tile_whole_cache_32_128
```

Expected output layout:

```text
po[(h, s, d)]
  d < Hd      -> unnormalized PV accumulator
  d == Hd     -> split denominator l
  d == Hd + 1 -> split max m
```

So the output width is:

```text
W = Hd + 2
```

## Required tile lifecycle

A valid candidate must do all of these inside one generated tile builder:

1. Load `q[h,e]`.
2. Load `K[kv,t,e]`.
3. Compute `score = q.k / sqrt(Hd)`.
4. Maintain online softmax state `(m,l)` over tokens in the split.
5. Accumulate `acc[d] += p * V[kv,t,d]` for local/cooperative `d` ownership.
6. Emit compact `(acc, l, m)` partials.

## Why this is hard

The builder combines three hard UOp shapes:

| Shape | Axis/reduction pressure |
|---|---|
| score | reduce over `e` for q.k |
| online state | recurrence over token `t` |
| PV | local/cooperative output `d` plus token accumulation |

The known risk is a multi-reduction/multi-granularity store wall. If the UOp store-group idiom cannot express this safely, classify the failure instead of tuning blindly.

## Gates

| Gate | Requirement | Failure verdict |
|---|---|---|
| P0 scope gate | scope + blocker artifact exists | `FUSED_SCORE_STATE_PV_TILE_SCOPE_INCOMPLETE` |
| P1 structural gate | target builder exists and emits target program identity | `FUSED_SCORE_STATE_PV_TILE_BLOCKED__NO_GENERATED_TILE_BUILDER` |
| P2 standalone numeric | fixed seeded tensors match NumPy reference | `FUSED_SCORE_STATE_PV_TILE_FAIL__NUMERIC` |
| P3 route gate | target route fires, owned absent, no `E_49152`, tokens match | `FUSED_SCORE_STATE_PV_TILE_FAIL__ROUTE` |
| P4 lifecycle gate | old score/max programs absent from route | `FUSED_SCORE_STATE_PV_TILE_FAIL__SCORE_OR_MAX_NOT_FUSED` |
| P5 W==D | candidate beats fused-PV-only and approaches owned baseline | `FUSED_SCORE_STATE_PV_TILE_REFUTED__WD` |

## Kill gates

| Failure | Classification |
|---|---|
| cannot express score reduce + online recurrence + local `d` PV in one builder | `FUSED_SCORE_STATE_PV_TILE_BLOCKED__MULTI_REDUCTION_STORE_SHAPE` |
| standalone numeric fails | reducer/state semantics bug |
| route clean but score/max programs still present | lifecycle not actually fused |
| route clean but W==D still loses heavily | missing dot/LDS/vectorization/compact-combine economics |
| route requires owned precompiled binary | not pure generated/search route |

## Do-not-promote rule

Standalone numeric success is not enough. Promotion requires:

| Requirement | Authority |
|---|---|
| standalone numeric correctness | fused score-state-PV gate |
| token match | route gate |
| target program fires | route gate |
| owned tile/combine absent | route gate |
| no `E_49152` | materialization gate |
| no old `flash_score_whole_cache` / `flash_max_32` | lifecycle gate |
| W==D competitive | decode eval |

## First expected verdict

Until the target builder exists, the canonical gate should report:

```text
FUSED_SCORE_STATE_PV_TILE_BLOCKED__NO_GENERATED_TILE_BUILDER
```
