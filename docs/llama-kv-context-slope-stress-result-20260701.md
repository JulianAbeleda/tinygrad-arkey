# llama.cpp KV context-slope stress result

## Summary

The long-context stress test confirms both parts of the hypothesis:

1. For f16 KV cache, llama.cpp decode slowdown is almost perfectly linear in context and matches the KV-read formula with plausible HBM bandwidth.
2. The simple storage formula is **not complete for speed** once KV cache is quantized. q8/q4 KV uses fewer storage bytes, but measured decode gets slower than f16 at the same depths, so quant/dequant or quantized-attention kernel overhead becomes the missing term.

## Method

Tool:

```text
extra/llama_kv_ctx_slope_bench.py
```

Command shape:

```bash
llama-bench -m <model> -ngl 99 -n 128 -p 0 -d <ctx> -r <reps> -ctk <type> -ctv <type> -o json
```

The tool fits:

```text
ms/token = A + B * ctx
```

and computes:

```text
implied_kv_bandwidth = kv_bytes_per_ctx_token / B
```

Raw artifacts are local under:

```text
bench/llama-kv-ctx-slope/
```

Those JSON files are benchmark outputs and remain ignored by git policy.

## Results

| model/cache | KV bytes per ctx token | A ms | B ms/ctx | R2 | implied KV BW | tok/s @512 | tok/s @32768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B f16 KV | 114,688 | 3.661 | 0.000138369 | 0.9961 | 829 GB/s | 272.5 | 121.4 |
| Qwen3-8B f16 KV | 147,456 | 10.129 | 0.000171794 | 0.9996 | 858 GB/s | 98.2 | 63.4 |
| Qwen3-8B q8_0 KV | 78,336 | 10.997 | 0.000228343 | 0.9997 | 343 GB/s | 89.9 | 54.0 |
| Qwen3-8B q4_0 KV | 41,472 | 11.223 | 0.000191623 | 0.9997 | 216 GB/s | 88.4 | 57.0 |

## Interpretation

For f16 KV, the formula is strong:

```text
KV bytes/context token = layers * kv_heads * head_dim * (bytes_K + bytes_V)
```

Qwen3-8B:

```text
36 layers * 8 KV heads * 128 head_dim * (2 + 2) bytes = 147,456 bytes/context-token
```

At 32K context:

```text
147,456 * 32,768 = 4.83 GB KV read per generated token
```

The fitted bandwidth is ~858 GB/s, which is plausible on the RX 7900 XTX. So for f16 KV, long-context decline is basically the KV-read slope.

For quantized KV, storage bytes alone fail:

```text
q8_0 stores ~53% of f16 KV bytes, but slope is worse: 0.000228 > 0.000172 ms/ctx.
q4_0 stores ~28% of f16 KV bytes, but slope is still worse than f16.
```

That means the speed formula needs extra terms:

```text
ms/token(ctx) =
  weight_ms
  + KV_read_ms(ctx)
  + KV_quant_dequant_or_quant_kernel_ms(ctx)
  + layout/metadata_ms(ctx)
  + launch/scheduler_ms
```

## Conclusion

The storage formula is reliable for VRAM planning.

The speed formula is:

```text
f16 KV:
  mostly complete as KV-read bandwidth slope

quantized KV:
  incomplete unless it includes quant/dequant or quantized-attention overhead
```

For positional memory, this supports keeping active context small and relevant even when long context fits in VRAM. At 8B f16 KV, 32K context drops decode from ~98 tok/s to ~63 tok/s. Quantized KV may fit more context, but this llama.cpp run shows it can cost speed unless the backend computes efficiently in the quantized domain.

