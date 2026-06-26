# Decode Attention A2 Whole-Cache Skeleton Result

## Verdict

`DECODE_ATTENTION_A2_GENERATED_WHOLECACHE_ROUTE_CLEAN`

A2 passes the lifecycle-clean generated route gate.

The generated attention skeleton now reads the whole `assigned_kv` cache buffer directly instead of consuming
`assigned_kv[0,0]` / `assigned_kv[1,0]` sliced views. This removes the `E_49152` materialization regression seen in
A1 while keeping the owned AMDGCN tile and combine disabled.

## Artifact

- `bench/qk-decode-attention-wholecache-skeleton/latest.json`
- Tool: `extra/qk_decode_attention_purity_capture.py --a2`
- Candidate flag: `DECODE_ATTN_GENERATED_WHOLECACHE=1`

## Result table

| Check | Owned baseline | A1 generated sliced-KV | A2 generated whole-cache |
|---|---:|---:|---:|
| Tokens match baseline | yes | yes | yes |
| Owned tile fires | 1 | 0 | 0 |
| Owned combine fires | 1 | 0 | 0 |
| Generated flash programs | 0 | 6 | 7 |
| `E_49152` present | no | yes | no |
| Selected route buffer identity | yes | no | yes |
| Promotion status | current default | not promotable | attribution-only pass |

## Captured generated programs

A2 captures these generated flash programs:

- `flash_score_whole_cache_32_128`
- `flash_max_32`
- `flash_prob_32`
- `flash_gmax_32`
- `flash_partial_coop_vec_whole_cache_32_128`
- `flash_den_32`
- `flash_combine_32_128`

The additional whole-cache score kernel is the key change from A1. It computes QK scores from the full cache buffer
directly. The whole-cache partial kernel reads V from the same full cache buffer. That avoids sliced K/V inputs and
keeps the old `E_49152` copy path out of the generated route.

## Interpretation

A2 proves the route/lifecycle side of generated decode attention is achievable:

```text
generated whole-cache attention
  -> no owned_flash_tile_gqa_whole
  -> no owned_flash_combine
  -> no E_49152
  -> token-correct on sampled decode
```

This does not prove performance parity. The A2 skeleton is still a scalar/generated lifecycle skeleton. It is expected
to be slower than the owned AMDGCN route because the performance primitives are still missing.

Remaining blockers:

- `v_dot2`
- cross-lane reduction
- LDS-staged tile layout
- TILE+COMBINE lifecycle search controls

## Note on `buffer_identity_inputs`

The older `buffer_identity_inputs` checker is tied to the owned-kernel sentinel name. For A2, the relevant guard is
`selected_route_buffer_identity`, which is true when the selected generated route has no `E_49152`/full-MAXC copy
kernels. The A2 artifact records:

- `E_49152_present: false`
- `full_maxc_copy_kernels: []`
- `selected_route_buffer_identity: true`

## Decision

Do not promote A2 as a speed route.

Use A2 as the new clean generated attention baseline for the next phase:

`A3 = performance primitive lowering`

The next work should add/search the missing low-level primitives against this lifecycle-clean generated skeleton,
starting with `v_dot2` or cross-lane reduction.
