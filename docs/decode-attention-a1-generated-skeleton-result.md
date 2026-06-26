# Decode Attention A1 Generated Skeleton Result

## Verdict

`DECODE_ATTENTION_A1_FAIL__E_49152_REINTRODUCED`

This is a useful blocker result, not an ambiguous failure.

The A1 generated skeleton successfully bypassed the owned AMDGCN attention tile and produced byte-identical tokens,
but it reintroduced the full-KV materialization signature `E_49152_32_3`.

## Artifact

- `bench/qk-decode-attention-generated-skeleton/latest.json`
- Tool: `extra/qk_decode_attention_purity_capture.py --a1`
- Candidate flag: `DECODE_ATTN_GENERATED_SKELETON=1`

## Result table

| Check | Owned baseline | A1 generated skeleton | Meaning |
|---|---:|---:|---|
| Tokens match | yes | yes | Generated math is functionally correct for the sampled decode path. |
| Owned tile fires | 1 | 0 | A1 successfully bypasses `owned_flash_tile_gqa_whole`. |
| Owned combine fires | 1 | 0 | A1 successfully bypasses `owned_flash_combine`. |
| Generated flash programs | 0 | 6 | A1 route is generated/attributable. |
| `E_49152` present | no | yes | A1 regresses KV materialization. |
| Buffer identity | yes | no | A1 does not preserve whole-cache KV identity. |
| Promotion status | current default | not promotable | Route ownership works; lifecycle cleanliness fails. |

## Captured generated programs

A1 captures these generated flash programs:

- `flash_max_32`
- `flash_prob_32`
- `flash_gmax_32`
- `flash_partial_coop_vec_32_128`
- `flash_den_32`
- `flash_combine_32_128`

This proves the generated skeleton exists and is selected. The problem is not that BubbleBeam/codegen cannot route
to generated attention at all. The problem is that the generated route currently consumes sliced KV views and causes
full-cache materialization.

## Interpretation

A1 splits decode-attention purity into two separate facts:

1. Route ownership is achievable now.
2. Lifecycle cleanliness is not achieved yet.

The generated skeleton path is token-correct and owned-kernel-free, but it is not a valid replacement for the shipped
owned route because it reintroduces the historical materialization tax:

```text
generated flash route
  -> assigned_kv[0,0] / assigned_kv[1,0] sliced KV inputs
  -> E_49152_32_3 materialization
  -> buffer identity lost
```

The owned route avoids this by passing the whole `assigned_kv` cache buffer to `owned_flash_tile_gqa_whole` and
offsetting K/V internally.

## Next blocker to solve

The next implementation target is not `v_dot2` yet.

First target:

- Add a generated whole-cache KV attention skeleton path that accepts the whole `assigned_kv` buffer and indexes K/V
  internally, instead of passing `assigned_kv[0,0]` and `assigned_kv[1,0]` sliced views.

Required gate:

- generated flash programs fire
- owned flash programs do not fire
- `E_49152_present == false`
- `buffer_identity_inputs == true`
- tokens match owned baseline

Only after that lifecycle gate passes should performance primitives matter:

- `v_dot2`
- cross-lane reduction
- LDS-staged tile layout
- TILE+COMBINE lifecycle search controls

## Decision

Do not promote A1.

Proceed to A2 as:

`generated_whole_cache_kv_attention_skeleton`

This is the smallest next step because it attacks the concrete blocker proven by A1 without conflating it with speed
or ISA-level performance work.
