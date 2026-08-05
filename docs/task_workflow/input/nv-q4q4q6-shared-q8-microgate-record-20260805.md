# Mixed Q4/Q4/Q6 shared-Q8 attention-group microgate

Date: 2026-08-05
Verdict: **performance PASS; model numerical qualification still required**

The real Qwen3-8B Q/K/V group is Q4_K `4096x4096`, Q4_K `1024x4096`,
and Q6_K `1024x4096`. This replaces the inapplicable all-Q6 proposal.

## Full model census

The native authority load has 36 blocks. Q is Q4_K in 36/36 and K is Q4_K in
36/36. V is Q6_K in 18/36 and Q4_K in 18/36: there are exactly 18
`(Q4,Q4,Q6)` groups and 18 `(Q4,Q4,Q4)` groups. Therefore this gate covers
half of the layers only; a separate Q4/Q4/Q4 consumer is required before an
all-layer claim is possible.

`extra/llm_research/decode/q4q4q6_shared_q8_microgate.py` compares the exact
installed three-kernel families (including Q6's external partial reduction)
against one Q8_1 pack and three direct DP4A consumers. The Q4 consumer keeps
its affine Q4_K term exactly in the Q8 domain: per 32-element group it computes
`activation_scale * (d*scale*dot(q4,x8) - dmin*min*sum(x8))`.

Native NV, RTX 5090, A/B/A:

| arm | us / Q,K,V group |
| --- | ---: |
| installed midpoint | 124.194 |
| shared-Q8 | 86.696 |
| delta | **-37.498** |

The group is not bit-identical to the fp16-activation route, by design: its
maximum output error in the synthetic real-shape gate was 0.18688 absolute,
0.00597 relative. It must therefore remain research-only until a full-logit,
generated-token, and all-layer native token-wall A/B establishes an acceptable
model contract. No production route is admitted by this document.
