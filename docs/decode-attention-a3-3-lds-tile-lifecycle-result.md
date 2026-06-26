# Decode Attention A3.3 LDS Tile Lifecycle Result

## Verdict

`A3_3_BLOCKED_BY_ROUTE_BINDING`

A3.3 asked whether decode attention can move from the A2 generated whole-cache skeleton toward the owned tile lifecycle by exposing a generated/search-owned LDS-staged tile primitive in the actual decode route.

Result: not yet. The A2 route remains clean, but `DECODE_ATTN_LDS_TILE=1` does not bind a new LDS/tile decode attention program. It still captures the same A2 generated programs:

- `flash_score_whole_cache_32_128`
- `flash_max_32`
- `flash_prob_32`
- `flash_gmax_32`
- `flash_partial_coop_vec_whole_cache_32_128`
- `flash_den_32`
- `flash_combine_32_128`

No `flash_*lds*` or `flash_*tile*` generated decode attention program appears.

## Command

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_a3_3_lds_tile_gate.py
```

## Artifact

- `bench/qk-decode-attention-a3-3-lds-tile/latest.json`

## Gate Result

| Check | Result |
|---|---|
| A2 generated whole-cache route clean | pass |
| owned attention tile/combine absent in A3.3 arm | pass |
| `E_49152` absent | pass |
| token sample matches owned baseline | pass |
| generated LDS/tile decode attention program present | fail |
| W==D candidate benchmark | skipped, because no LDS/tile candidate was route-bound |

## Standalone Generated LDS Evidence

The repo does already contain standalone generated LDS flash-attention code in `extra/gemm/amd_flash_attention.py`.

The gate recorded:

| Evidence | Present |
|---|---:|
| `AddrSpace.LOCAL` | yes |
| barriers | yes |
| `SHAPED_WMMA` | yes |
| cross-lane / warp-reduce logic | yes |

This matters, but it is not a decode promotion path by itself. It proves generated LDS attention-style code exists in the repo; it does not prove the decode T=1 whole-cache lifecycle can bind that representation without reintroducing materialization or changing the split-KV lifecycle.

## Baseline Numbers Captured During Gate

These are not an A3.3 candidate result. They are the owned and A2 references captured while proving the route-binding blocker.

| ctx | owned tok/s | A2 generated tok/s | A2 vs owned |
|---:|---:|---:|---:|
| 512 | 105.0 | 78.3 | 74.6% |
| 1024 | 103.1 | 75.7 | 73.4% |
| 2048 | 100.4 | 70.1 | 69.8% |
| 4096 | 95.5 | 60.6 | 63.5% |

## Interpretation

The current blocker is not that generated UOp code can never express LDS, barriers, WMMA, or cross-lane behavior. The blocker is route binding and lifecycle shape:

- A2 has the correct decode lifecycle hygiene: whole-cache input, no owned tile/combine, no `E_49152`.
- The standalone generated LDS flash attention kernel has the right primitive family, but it is shaped for full attention, not the current decode whole-cache split lifecycle.
- A3.3 needs a decode-specific generated tile lifecycle candidate, not just a flag around A2 and not a standalone full-attention kernel.

## Decision

Promote nothing from A3.3.

Next executable step: A3.4 should define and bind a decode-specific TILE+COMBINE lifecycle candidate manifest. That candidate should describe tile program, combine program, split policy, intermediate buffers, materialization guarantees, and primitive requirements together. Only after that route exists should W==D be used as the promotion authority.
