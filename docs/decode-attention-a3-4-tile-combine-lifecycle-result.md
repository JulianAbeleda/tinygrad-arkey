# Decode Attention A3.4 TILE+COMBINE Lifecycle Result

## Verdict

`A3_4_ROUTE_BINDING_MISSING`

A3.4 turns decode attention from isolated primitive probes into a lifecycle bundle problem.

The owned route is not just a score kernel. It is a TILE+COMBINE lifecycle:

- TILE: split-KV tile work, score, online softmax state, partial PV, metadata.
- COMBINE: global log-sum-exp combine over partials.

Pure machine search needs to see that pair as one candidate bundle. Otherwise search can improve a local piece while losing on the actual decode lifecycle.

## Files

- Manifest: `bench/qk-search-spaces/decode_attention_tile_combine_a3_4.json`
- Gate: `extra/qk_decode_attention_a3_4_tile_combine_gate.py`
- Artifact: `bench/qk-decode-attention-a3-4-tile-combine/latest.json`

## Command

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_a3_4_tile_combine_gate.py
```

## Gate Result

| Check | Result |
|---|---|
| lifecycle manifest exists | pass |
| A2 generated whole-cache route clean | pass |
| owned tile/combine absent in A3.4 arm | pass |
| `E_49152` absent | pass |
| token sample matches owned baseline | pass |
| generated combine program present | pass: `flash_combine_32_128` |
| generated partial/score/metadata programs present | pass |
| generated tile program present | fail |
| full TILE+COMBINE bundle bound | fail |
| W==D candidate benchmark | skipped, because no tile bundle was bound |

## Bundle Manifest

The manifest is `decode_attention_tile_combine_a3_4` and defines one candidate bundle:

- candidate id: `decode_attention_generated_tile_combine_lifecycle_a3_4`
- base route: `decode_attention_generated_wholecache_skeleton`
- tile requirement: `flash_*tile*`
- combine requirement: `flash_*combine*`
- split-policy knobs: `split_count`, `tile_k`, `tile_d`, `workgroup_shape`, `waves_per_tile`
- materialization guarantees: whole-cache input, no `E_49152`, no owned tile, no owned combine
- primitive requirements: whole-cache identity, vector dot, cross-lane reduction, LDS/tile-staged KV lifecycle, online softmax state, bundle attribution

## Captured A3.4 Signature

The A3.4 arm still routes the same A2 generated programs:

| Program class | Programs | Status |
|---|---|---|
| score | `flash_score_whole_cache_32_128` | present |
| metadata | `flash_max_32`, `flash_prob_32`, `flash_gmax_32`, `flash_den_32` | present |
| partial | `flash_partial_coop_vec_whole_cache_32_128` | present |
| combine | `flash_combine_32_128` | present |
| tile | none | missing |

Because the generated tile program is missing, the lifecycle bundle is not route-bound.

## Reference Numbers Captured During Gate

These are reference numbers only. They are not an A3.4 candidate result because no A3.4 bundle was bound.

| ctx | owned tok/s | A2 generated tok/s | A2 vs owned |
|---:|---:|---:|---:|
| 512 | 105.1 | 78.0 | 74.2% |
| 1024 | 103.1 | 75.3 | 73.0% |
| 2048 | 100.9 | 69.5 | 68.9% |
| 4096 | 95.8 | 60.4 | 63.0% |

## Interpretation

A3.4 solved the representation/documentation problem, not the routing/codegen problem.

Before A3.4, the project could talk about `v_dot2`, cross-lane, LDS, score, partial, and combine as separate probes. Now there is a concrete lifecycle manifest that says what must be true for a generated/search-owned decode attention candidate to be real.

The immediate blocker is precise:

- A2 already provides lifecycle hygiene: whole-cache input, no owned attention, no `E_49152`.
- A3.4 adds candidate-bundle attribution.
- The missing piece is a generated tile program that can replace the score/partial split with a tile lifecycle while preserving A2 hygiene.

## Decision

Promote nothing from A3.4.

Next executable step: build the minimal generated tile placeholder route for A3.5. It does not need to be fast first. It must create a named generated tile program, preserve whole-cache/no-`E_49152` hygiene, match tokens, and let the lifecycle gate move from `ROUTE_BINDING_MISSING` to a measurable transfer/no-transfer result.
