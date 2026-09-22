# Current252 generated Q6-V promotion

The final 18 Q6 attention-V FP16 overlays were replaced with the existing
graph-owned generated Q6-V producer and main while retaining every current234
route. Fresh current234/current252/current234 medians are 54.935668, 53.302259,
and 55.000969 ms. The candidate improves 1.666059 ms or 3.031% against the mean
control median.

The candidate selects 252 projection mains and producers, 252 unique canonical
weight bases, and zero V/down FP16 overlays. It passes token 198, 20/20 exact
recurrent replay, finite full logits, and rtol 0.02/atol 0.5 against current234
(max absolute 0.16928625, mean 0.01915582). The explicit compiler pp512 route
now includes both `attn_v` and `ffn_down` in its default Q6 role set. Setting
`NV_COMPILER_Q6_IMMA_PP512_ROLES=ffn_down` restores the preceding current234
composition.
