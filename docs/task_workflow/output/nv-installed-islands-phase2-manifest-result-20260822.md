# NV installed-island Phase 2 semantic island manifest

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`

Evidence: `docs/task_workflow/evidence/nv-installed-islands-20260822/phase2/`

## Verdict

`MEASURED` Every tinygrad (596) and llama (762) decode node is assigned exactly
once to a disjoint island boundary with zero unmapped nodes. The positional
rope mapping is confirmed by adjacency, and the mixed K/V projection family is
kept visible rather than silently split.

## Island boundaries

```text
I_Q     Q projection partials -> Q completion -> Q RMSNorm -> Q rope complete
I_K     K projection partials -> K completion -> K RMSNorm -> K rope/store complete
I_V     V projection -> V handoff/store ready
I_ATTN  Q/K/V ready -> flash score -> combine -> O input ready
I_O     O input ready -> O projection/residual complete
I_FFN   FFN norm ready -> gate/up -> activation -> down/residual complete
I_TAIL  final norm -> vocab -> sampler feedback ready
```

## Physical and semantic cardinality

`MEASURED` node_sum is the profile-domain sum of per-node intervals and
reproduces the locked census exactly (tinygrad `4677.920 us`, llama
`3878.254 us`). Physical counts and node_sum are reported separately so
cardinality differences stay visible.

| island | tg physical | tg node_sum us | llama physical | llama node_sum us |
| --- | ---: | ---: | ---: | ---: |
| I_Q | 125 | 495.072 | 180 | 444.935 |
| I_K | 72 | 139.648 | 180 | 242.017 |
| I_KV_MIXED | 80 | 253.056 | 0 | 0.000 |
| I_V | 18 | 74.176 | 72 | 124.737 |
| I_ATTN | 72 | 331.488 | 108 | 224.037 |
| I_O | 36 | 335.040 | 36 | 259.809 |
| I_FFN | 183 | 2659.552 | 180 | 2273.339 |
| I_TAIL | 10 | 389.888 | 6 | 309.380 |

`MEASURED` `I_KV_MIXED` is not an island; it is the 1024x4096 K/V projection
family (`q4k_g3_lanemap_gemv_1024_4096`, `q4k_warp_coop_q8_dp4a_partial_1024_4096`)
plus the 26 `r_8_32_4_4` completion nodes. Its K-vs-V split is deferred to
Phase 7 partitioned projection routes and is not averaged away here.

## Rope positional confirmation

`MEASURED` The per-layer 4-node subsequence is exactly

```text
reduce_output_rmsnorm_32_128  (Q norm)
reduce_output_rmsnorm_8_128   (K norm)
E_16_32_4_2                    (Q rope)
E_8_8_16_2                     (K rope + K/V store)
```

This subsequence occurs 36 times (one per layer). The single
`E_16_4_2_8_16_2_4_4` node is the pre-projection head rope, not a per-layer
Q/K rope. The positional mapping passes, so the three `E_*` rope families are
now usable for candidates with these identities fixed.

## Producer-to-consumer boundaries

`MEASURED` From the retained dependency captures:

```text
I_Q -> I_ATTN  via q_rope -> flash score
I_K -> I_ATTN  via k_rope_store -> flash score
I_V -> I_ATTN  via V store -> flash score
I_ATTN -> I_O  via flash combine -> O projection
I_O -> I_FFN   via O residual -> FFN norm
I_FFN -> I_Q   via down residual -> next-layer attn norm (cross-layer)
I_FFN -> I_TAIL via final down -> final norm -> vocab
```

`MEASURED` These edges are the same producer/consumer seams the later phases
decompose. No model-name or fixed-block-list dispatch is introduced.

## Ledger snapshot

```text
node_sum   = 4677.920 us (tinygrad) / 3878.254 us (llama)
union      = 4671.500 us (tinygrad, locked) / 3878.254 us (llama PDL-off)
overlap    = 6.420 us (tinygrad) / 0 us (llama PDL-off)
wall       = 4771.423 us (fresh control, Phase 1)
host_gap   = unmeasured single-domain
useful_body = unmeasured
booked_recovery = 0.000 us
remaining_to_240 = 604.756 us
```
