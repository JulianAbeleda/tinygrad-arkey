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
