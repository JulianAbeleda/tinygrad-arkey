# NV sm_120: generic block-quant (Q8_0) decode GEMV route, result 2026-10-10

Route `decode_q8_0_gemv`, record `tinygrad/llm/generated/decode-q8-0-gemv-route-policy.json`
(promoted NV sm_120, formats `[Q8_0]`). Rollback: `TINYGRAD_Q8_0_GEMV_DISABLE=1`.

## Rule

The kernel is emitted from (rows, k), the block format's layout, and device facts (wave size, SM count,
max threads per CTA). No model dimension is written in the route. A weight is admitted only where
`BlockQuantGEMVRouteSpec.validate` accepts its shape (k % block_elems == 0, rows >= 1); the model-level
gate is `DECODE_ROUTE_SHAPE_PROBES["decode_q8_0_gemv"]` through `DecodeRouteShapeGate`.

## Geometry (measured, cold L2, RTX 5090, peak 1,693.9 GB/s)

12 Q8_0 shapes from 0.3 to 165 MB, lanes_per_block {1,2,4,8,16} x warps_per_cta {1,2,4,8,16}:
16 lanes/block (one uint16 code unit per lane) with 2 warps/CTA was best or within noise of best on every
shape: 1.49 to 1.66 TB/s at >= 33 MB; 4 and 8 lanes within 3%; 1 lane/block ~3x slower.
Family check (same emitter, other layouts): Q5_0 10240x5120 1,339 GB/s (79%); Q4_0 1,353 GB/s (80%);
Nemotron-3-Nano-4B real Q5_0 `blk.0.ssm_in` 17504x3136 (k not a multiple of 512) 1,263 GB/s (75%),
rel err 1.2e-7 vs the ggml dequantizer; its Q8_0 `output` 131072x3136 1,681 GB/s (99%).

## In-model (BoltBeam `screen pipeline --provider tinygrad --role-time in-model`, 10 tokens after 128)

| model | Q8_0 linears | tok/s before | tok/s after | ctx 512 before/after |
|---|---|---|---|---|
| Qwen3.8-27B-Q4_K_M (hybrid, hidden 5120) | 288 | 2.95 | 18.96 | 2.93 / 18.50 |
| Qwen3-0.6B-Q8_0 (hidden 1024, tied Q8_0 lm_head) | 197 | 150.56 | 362.26 | 115.59 / 211.14 |
| Qwen3-8B-Q4_K_M (no Q8_0) | 0 | 241.46 | 241.73 | 240.24 / 240.72 |

0.6B both arms with `TINYGRAD_DECODE_ROUTE=fp16`: the flash decode route refuses Hq=16 and has no fallback.

27B before: the Q8_0 projections lowered by default: attn_qkv `r_2_2_8_16_3_4_160` 48 x 4.84 ms (11.5 GB/s),
attn_q `r_6_4_4_2_16_4_160` 16 x 1.09 ms (61 GB/s), ssm_out `r_40_32_4_192` 48 x 0.71 ms (47 GB/s),
attn_gate `r_320_32_4_160` 48 x 0.15 ms (229 GB/s) (attributed by which kernels the route removed).
27B after, Q8_0 kernels: 10240x5120 1,517 GB/s (90%), 12288x5120 1,557 (92%), 5120x6144 1,490 (88%),
6144x5120 1,427 (84%), 1024x5120 582 (34%), 48x5120 27 (2%, launch-bound); all Q8_0 5.89 ms/token for
7.16 GB/token = 1,216 GB/s (72%). Next 27B gap: `r_40_32_4_24...` 16 x 2.23 ms = 35.7 ms/token, the Q6_K
attn_output 5120x6144 on default lowering (Q6_K primitives install only for ffn_down/lm_head/attn_v roles).
0.6B after: lm_head 151936x1024 1,670 GB/s (99%); the 1-3 MB projections are launch-bound (2.9 to 6.7 us,
22 to 50%); all Q8_0 0.91 ms/token, 694 GB/s (41%).
8B: kernel census identical (27 distinct kernels, 418 per token, same calls per kernel).

## Logits (extra/llm_research/decode/nv_q8_0_gemv_logits.py, teacher-forced, control = rollback arm)

| model | steps | argmax equal | max abs | max abs / step max | top-5 overlap | bitwise |
|---|---|---|---|---|---|---|
| 27B (depth 64) | 8 | yes | 0.0097 | 5.7e-4 | 5/5 | no |
| 0.6B (depth 128) | 16 | yes | 0.0141 | 6.6e-4 | 5/5 | no |
| 8B (depth 128) | 16 | yes | 0 | 0 | 5/5 | yes |

Tolerance: argmax equal on every step and max |dlogit| <= 1e-3 of the step's max |logit|. Not bitwise for
Q8_0 because the kernel sums d * sum(q*x) per lane then a warp ladder, where the control graph reduces
d*q*x in its own order; both are fp32.

## Gates

- Exact control: 8B bitwise; Q8_0 models within the tolerance above (not bitwise: stated, not hidden).
- Wall bracket vs the +7% decode bar: 27B +543%, 0.6B +141% (ctx 128), +83% (ctx 512).
- Census: 27B 1,803 -> 1,851 kernels/token; 0.6B 709 -> 822; 8B unchanged.

Commit measured: 1cf0c0412.
