# P6-C/O exact llama Q4_K attention-O family graph A/B

Date: 2026-08-04. Route: `DEV=CUDA`, d512, RTX 5090 / driver 595.84. Status:
**exact ABI and epilogue resolved; complete llama-policy family substitution NO-GO.**
This is diagnostic evidence, not native-NV residual credit or a production/default change.

## Exact llama policy and semantic ABI

An observational `cudaStreamEndCapture` tap recorded 72 llama `4096x4096` Q4_K MMVQ
nodes in strict `N/F` alternation for 35 layers, followed by `N/N`. Thus the graph has
37 non-fused nodes (36 attention-Q plus the final attention-O) and 35 fused attention-O
nodes. The unique final non-fused node is global Q/O-population ordinal 71; the A/B
preserves it rather than incorrectly treating all 36 O projections as fused.

Every fused O node uses the `mul_mat_vec_q<Q4_K,1,true,false>` entry. `FusionArgs` is
32 bytes with only `x_bias` populated; `gate` and `gate_bias` are null and `glu_op=0`.
The semantic operation is:

```text
output = Q4_K(weight) @ q8_1(activation) + residual
```

The exact synthetic `4096x4096` launch passed against independently decoded Q4_K and
q8_1 plus a random fp32 residual: max absolute error `4.2915e-6`, RMSE `1.0811e-6`,
and exact argmax. Raw oracle: `/tmp/q4_attn_o_oracle.json`, SHA-256
`96a1e414aee8757581d94a623d7b445f4229716415ba7cf14ec4ed9892bb4d8b`.
The live ABI JSONL SHA-256 is
`7ae649505d88018172d83a7a9a4e8c43391da2942b858275270206ece6ecc8ba`.

## Correctness-preserving graph substitution

`scratchpad/cuda_decode_q4_attention_o_llama_graph_ab.py` maps the ordered Q/O
population, selects odd global ordinals 1 through 69, and leaves ordinal 71 unchanged.
For each of the 35 selected O roles it replaces tinygrad's O GEMV plus its immediately
dependent fp32 residual-add node with:

1. fp16-to-fp32 boundary adapter;
2. exact llama `quantize_q8_1`;
3. exact llama fused Q4_K MMVQ writing directly to the residual-add output buffer.

The fused MMVQ depends on both q8 production and the original residual producer. All
consumers of the removed add are moved to the fused node. Each replacement adds three
nodes and removes two, so the expected graph-node delta is `+35`. All arms emitted the
same 32 token IDs. This token equality is supplemented by the exact isolated numerical
oracle above; token equality alone is not treated as a full numerical proof.

## Reverse bracket

Each arm has 31 steady samples after graph construction.

| arm | median ms/token | p5 | p95 | MAD |
| --- | ---: | ---: | ---: | ---: |
| control A | `5.608072` | `5.598901` | `5.623713` | `0.006714` |
| llama-policy fused O family | `5.630468` | `5.620224` | `5.650632` | `0.003106` |
| control B | `5.610438` | `5.602509` | `5.622527` | `0.004499` |

Control midpoint is `5.609255 ms/token`. The exact llama-policy substitution changes
wall by `+0.021213 ms/token` (`+0.378%`): it is slower. P6-C/O is therefore **NO-GO**
and contributes zero residual-ledger recovery.

The causal interpretation is bounded: llama's fused residual epilogue removes 35
standalone adds, but importing the route also requires fp16-to-fp32 conversion and q8
production and increases this tinygrad graph by one node per replaced role. In this
token, those costs exceed the eliminated adds. This does not show that residual fusion
is intrinsically bad in a tinygrad-owned kernel that avoids the adapter/q8 tax.

Raw bracket SHA-256:

- `/tmp/q4_attn_o_exact_control_a.json`: `50d712504ee3eec895c8435b8222094f6103f39c37afe60743fe3a9834954ef7`
- `/tmp/q4_attn_o_exact_ab.json`: `f1cd4c3366161d321b0c284e7073a07f56ddd4550396b7147b1324eec8148b08`
- `/tmp/q4_attn_o_exact_control_b.json`: `c892c30146f4ae36a354384d173fd9d21c95f5ebfe10081c82445b0db2bdb7ee`

No route/default changed; no native-NV residual is debited.
