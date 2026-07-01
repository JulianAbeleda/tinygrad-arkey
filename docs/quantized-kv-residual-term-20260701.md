# Quantized KV residual term

Status: source-audited, measurement-grounded.

This note explains why the long-context speed slope for llama.cpp does not follow KV-cache storage bytes once the KV cache is quantized.

## Source path

On the local llama.cpp ROCm build, GPU FlashAttention dispatch is selected by K/V dtype in `ggml/src/ggml-cuda/fattn.cu`. The default compiled fast cases include same-type `f16/f16`, `q4_0/q4_0`, `q8_0/q8_0`, and `bf16/bf16`. Wider mixed K/V dtype support is guarded behind `GGML_CUDA_FA_ALL_QUANTS`.

The quantized path is not just a smaller f16 read:

- K-side attention score uses dtype-specific `vec_dot_KQ` helpers selected in `ggml/src/ggml-cuda/fattn-common.cuh`.
- `q4_0` K uses nibble unpack, scale load, integer dot, and scale correction.
- `q8_0` K uses q8 block load plus q8 dot against a quantized Q representation.
- V-side output accumulation uses dtype-specific `dequantize_V` helpers selected in the same file.

So quantized KV introduces a per-context-token residual term:

```text
T_decode(ctx, K, V)
  = A
  + ctx * bytes_kv(K,V) / BW_f16_attention
  + ctx * R_quant(K,V)
```

where:

```text
R_quant(K,V)
  = R_K_dot(K) + R_V_dequant(V) + R_layout_pair(K,V)
```

With the current default build, mixed K/V runs are not the right way to estimate `R_K_dot` and `R_V_dequant` separately, because mixed types can miss the default same-type fast cases. A diagnostic `q8_0 K / f16 V` run on 0.6B fell to an extremely slow path, confirming that mixed dtype support is not equivalent to the same-type fast path in this build.

## Measured 8B slopes

All rows are llama.cpp `llama-bench` text-generation depth sweeps on Qwen3-8B-Q4_K_M, flash attention on, fitted as:

```text
ms/token = A + B * ctx
```

| KV type | KV bytes / context token | A ms | B ms / ctx | R2 | implied storage BW |
|---|---:|---:|---:|---:|---:|
| f16/f16 | 147456 | 10.129 | 0.000171794 | 0.9996 | 858 GB/s |
| q8_0/q8_0 | 78336 | 10.997 | 0.000228343 | 0.9997 | 343 GB/s |
| q4_0/q4_0 | 41472 | 11.223 | 0.000191623 | 0.9997 | 216 GB/s |

Using the f16 attention bandwidth as the storage baseline:

```text
BW_f16_attention = 858 GB/s
B_storage_q8 = 78336 / 858e9 * 1000 = 0.000091 ms/ctx
B_storage_q4 = 41472 / 858e9 * 1000 = 0.000048 ms/ctx
```

Residual:

| KV type | measured B | storage-only B | residual B | residual share |
|---|---:|---:|---:|---:|
| q8_0/q8_0 | 0.000228 | 0.000091 | 0.000137 | 60% |
| q4_0/q4_0 | 0.000192 | 0.000048 | 0.000143 | 75% |

So q4_0 reduces stored bytes versus q8_0, but it does not proportionally reduce time. The saved bytes are partly replaced by unpack/dequant/dot overhead.

## Practical formula

For planning speed:

```text
B(K,V) = bytes_kv(K,V) / BW_f16_attention + R_quant_pair(K,V)
```

For the measured 8B ROCm llama.cpp path:

```text
BW_f16_attention ~= 858 GB/s
R_quant_pair(q8_0,q8_0) ~= 0.000137 ms/context-token
R_quant_pair(q4_0,q4_0) ~= 0.000143 ms/context-token
```

Then:

```text
tok/s(ctx,K,V) ~= 1000 / (A(K,V) + B(K,V) * ctx)
```

This is why storage formula remains useful for VRAM planning, but speed needs the residual term.

## Literature check

This matches the broader KV-cache quantization literature. KIVI argues that KV cache size dominates long-context memory and that a hardware-friendly implementation is needed for throughput gains, not only smaller bytes. It also distinguishes key and value quantization structure: keys prefer per-channel grouping, values per-token grouping. Recent KV-quantization work similarly describes attention kernels that process packed quantized cache entries by dequantizing them during attention, while keeping some residual/full-precision cache entries for correctness or streaming.

References:

- KIVI: https://arxiv.org/html/2402.02750v2
- KIVI proceedings page: https://proceedings.mlr.press/v235/liu24bz.html
- Occam's Razor for Extreme KV Cache Quantization: https://arxiv.org/html/2605.19660v1
- llama.cpp KV cache dtype flags: `/home/ubuntu/env/llama.cpp/tools/llama-bench/README.md`
- llama.cpp FlashAttention dtype dispatch: `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn.cu`
- llama.cpp K/V quantized attention helpers: `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh`

## Next benchmark

To split `R_K_dot` and `R_V_dequant` exactly, rebuild or configure llama.cpp with all quantized FlashAttention mixed dtype cases enabled, then run:

```text
f16/f16
q8_0/f16
f16/q8_0
q8_0/q8_0
q4_0/f16
f16/q4_0
q4_0/q4_0
```

Without all mixed fast cases enabled, mixed measurements can fall off the same-type fast path and overstate the residual.
