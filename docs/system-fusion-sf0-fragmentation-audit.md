# System Fusion SF0 — Fragmentation Audit

Date: 2026-07-01. Model: Qwen3-14B-Q4_K_M, gfx1100. Defaults: DECODE_Q4K_G3_ANYSHAPE=1, DECODE_ROUTE_ATTN_K=1, DECODE_Q6K_FFN_DOWN_LONGK=1.

Goal: resolve the ~12% "launch/activation/other" bucket from the decode loss stack into named, attributed fragments.

## Bucket summary (ctx512, current shipped state)

| bucket | % GPU | kernels | calls/step |
|--------|-------|---------|------------|
| q4k_gemv | 43.20 | 4 | 220 |
| reduce_partial | 19.05 | 15 | 267 |
| **other (E_\* elementwise)** | **12.76** | **14** | **345** |
| q6k_gemv | 11.64 | 2 | 40 |
| attention (flash) | 9.84 | 6 | 240 |
| lm_head | 3.51 | 1 | 1 |

At ctx128 flash is OFF so E_49152_32_3 does not appear; the elementwise bucket is 5.90%.

## Resolved fragments (ctx512 "other" bucket)

| kernel | % GPU | calls/step | fragment class | producer | consumer |
|--------|-------|------------|----------------|----------|----------|
| E_49152_32_3 | 6.69 | 40 | attention_elementwise | r_40_28start_pos2B129_16_8 (RoPE reduce) | E_1920_32_3 |
| E_5_2_2_16_4_4n1 | 1.46 | 40 | qk_norm_scale (hidden_elementwise) | r_8_16_8 (k_norm reduce) | r_8_8_16_2_4n1 |
| E_1920_32_3 | 0.75 | 40 | attention_elementwise | E_49152_32_3 | flash_max_40 |
| E_136_32_4 | 0.67 | 40 | silu_gate_activation | q4k_g3_lanemap_gemv_17408_5120 | E_136_32_4n1 |
| E_40_32_4 | 0.64 | 40 | rmsnorm_scale_pre_attn | r_16_320 (RMSNorm reduce) | q4k_g3_lanemap_gemv_5120_5120 |
| E_40_32_4n2 | 0.64 | 40 | rmsnorm_scale_pre_ffn | r_16_320n1 (RMSNorm reduce) | q4k_g3_lanemap_gemv_17408_5120 |
| E_136_32_4n1 | 0.59 | 40 | gate_up_multiply | E_136_32_4 (silu) | q6k_coop_partial_5120_17408 |
| E_40_32_4n1 | 0.59 | 40 | residual_add_post_attn | flash_combine_40_128 | q4k_g3_lanemap_gemv_5120_5120 |
| E_40_32_4n3 | 0.31 | 20 | residual_add_post_ffn | q4k_g3_lanemap_gemv_5120_17408 | r_16_320 |
| E_20_4_2_8_16_2_4_4 | 0.14 | 1 | attention_elementwise (init) | graph_start | E_2n7 |
| TracingKey(AMD→TINY) | 0.15 | 1 | graph_boundary | — | E_20... |
| E_1187_32_4 | 0.07 | 1 | lm_head_elementwise | lm_head reduce | — |
| E_2n7 | 0.05 | 1 | graph_boundary (init) | E_20... | reduce |
| E_40_32_4n4 | 0.01 | 1 | hidden_elementwise (one-off) | — | — |

**unknown = 0.0%** at ctx512.
**Verdict: SF0_PASS_FRAGMENTATION_RESOLVED**

## Fragment classes used

- **attention_elementwise**: elementwise ops adjacent to the flash attention path (KV cache write, RoPE apply output, flash-adjacent init). Absent at ctx128 (flash OFF).
- **silu_gate_activation**: SiLU gate activation for SwiGLU FFN (E_136_32_4) and gate×up multiply (E_136_32_4n1). These are two kernels because model.py:1017 has `.silu().contiguous()`, forcing intermediate materialization.
- **hidden_elementwise**: per-layer hidden-dim elementwise (5120 elements): RMSNorm scale (E_40_32_4, E_40_32_4n2), residual adds (E_40_32_4n1, E_40_32_4n3).
- **qk_norm_scale**: per-head Q/K norm scale after QK norm reduce (E_5_2_2_16_4_4n1, 5120 elements).
- **lm_head_elementwise**: post-lm_head elementwise or sampling (E_1187_32_4, 1 call/step).
- **graph_boundary**: tinygrad graph-boundary sync or tiny init kernel (TracingKey, E_2n7).

## Root cause: silu_gate split

`E_136_32_4` and `E_136_32_4n1` are two separate kernels because `tinygrad/llm/model.py:1017` has:

```python
return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))
```

The `.contiguous()` between `.silu()` and `* self.ffn_up(x)` forces materialization of silu(gate) as an intermediate buffer. This produces TWO kernels instead of one fused silu×up kernel. The comment at line 1010 confirms this is a known issue: `# TODO: remove the need for this contiguous`.

## SF0 → SF1 hand-off

All 14 kernels in the "other" bucket are attributed. Next phase: SF1 fuseability classification for each named fragment.
