# NV sm_120: role-agnostic Q6_K primitive installation, result 2026-10-10

Change: `model_route_plan._default` gives every Q6_K weight the decode reads as a GEMV a primitive route entry
(parts=1: the coop emitter wherever its validate admits the shape, else the partial path). Before, only the
ffn_down, lm_head and attn_v roles got one. The measured attn_v placement (4 parts) is kept. An embedding is never
replaced (`_install_qk_primitives` skips nn.Embedding). Rollback: `TINYGRAD_Q6K_ROLE_AGNOSTIC_DISABLE=1` (the old
role list). Commits: 4ad75dfb1 (rule), 24f60a6bd (rollback, `extra/llm_research/decode/decode_route_logits.py`).

## Route plan (CPU, from the GGUF headers)

| model | arm | entries | Q6_K entries |
|---|---|---|---|
| Qwen3-8B-Q4_K_M | rollback | 253 | 37: attn_v 1024x4096 (4 parts) x18, ffn_down 4096x12288 x18, output 151936x4096 |
| Qwen3-8B-Q4_K_M | rule | 253 | the same 37 |
| Qwen3.8-27B-Q4_K_M | rollback | 193 | 1: output 248320x5120 |
| Qwen3.8-27B-Q4_K_M | rule | 209 | 17: attn_output 5120x6144 x16, output |

The 8B's plan is identical, so the 8B is not re-measured.

## In-model, RTX 5090 (BoltBeam `screen pipeline --provider tinygrad --role-time in-model`, ctx 128)

| model | tok/s rollback | tok/s rule | ms/token | launches/token | % of the 92.8 tok/s limit |
|---|---|---|---|---|---|
| Qwen3.8-27B-Q4_K_M | 18.61 | 54.46 | 53.74 -> 18.36 | 1851 -> 1867 | 20% -> 59% |

The removed kernel: `r_40_32_4_24_2_2_2_4_8_*` (attn_output on the default lowering) 16 x 2234 us = 35.74 ms/token.
Its replacement: `q6k_gen_coop_5120_6144_inkernel` 16 x 21.1 us = 0.34 ms/token, 1,224 GB/s (72% of 1,693.9 GB/s).
Top kernels after: ffn_gate_up Q4_K 4.26 ms, ffn_down Q4_K 2.66 ms, Q8_0 10240x5120 1.76 ms, 6144x5120 1.13 ms,
5120x6144 1.08 ms, 48x5120 0.93 ms per token.

## Logits (decode_route_logits.py, teacher-forced, control = rollback arm)

| model | steps | argmax equal | max abs | max abs / step max | top-5 | bitwise |
|---|---|---|---|---|---|---|
| 27B (depth 64) | 8 | yes | 0.0065 | 4.75e-4 | 5/5 | no |

Not bitwise: the coop kernel sums d*sc*q*x per lane then a warp ladder; the control graph reduces in its own order.

## Gates

- Correctness: argmax equal on every step, max |dlogit| <= 1e-3 of the step max: PASS.
- Wall vs the +7% decode bar: +193%.
- The 8B ran both arms anyway (queued before the GPU budget was cut): 239.17 / 239.22 tok/s, 418 launches/token
  in each, logits bitwise equal (16 steps). It confirms the CPU route-plan check.

## Run + search on the 27B after (BoltBeam `kinds-27b-p2`, search on)

Run: 17.1 ms per token (58.3 tok/s, a separate run from the A/B above); the limit is 10.8 ms (92.8 tok/s); 63% of the
limit; 6.4 ms per token lost.

Where the token goes (in-model, per token):

| role | ms | floor ms | lost ms | calls |
|---|---|---|---|---|
| ffn_gate_up Q4_K 17408x5120 | 4.26 | 3.79 | 0.47 | 64 |
| ffn_down Q4_K 5120x17408 | 2.66 | 1.90 | 0.76 | 64 |
| lm_head Q6_K 248320x5120 | 0.65 | 0.62 | 0.04 | 1 |
| attn_qo Q6_K 5120x6144 (attn_output) | 0.34 | 0.24 | 0.09 | 16 |
| not attributed (Q8_0 GEMVs, attention, recurrent blocks, norms) | 10.45 | | | |

The Q8_0 GEMVs (about 6 ms per token by kernel name) are in "not attributed": their programs carry no semantic
identity in the capture yet (open; see the BoltBeam report).

Search (BubbleBeam proposed / FutureSight rejected / BoltBeam reproduced; BoltBeam's fastest less the floor against
the model's own per call):

| role | proposed | rejected | reproduced | BoltBeam us | model us | verdict |
|---|---|---|---|---|---|---|
| ffn_gate_up Q4_K | 5 | 0 | 1 | 31.2 | 66.6 (fused w1w3) | not reproduced |
| ffn_down Q4_K | 5 | 0 | 1 | 38.5 | 41.5 | not reproduced |
| attn_qo Q6_K 5120x6144 | 11 | 3 | 2 | 20.1 | 21.1 | not reproduced |
| lm_head Q6_K | 11 | 3 | 5 | 672.0 | 654.8 | reproduced; the model's own kernel is faster |
| Q8_0 roles (6) | 27-32 | 0 | 0 | 3.7-41.1 | n/a | not reproduced |
