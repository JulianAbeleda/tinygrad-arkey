# Llama KV context-slope formula scope

## Goal

Build a benchmark and analysis pass that explains whether llama.cpp decode speed declines with context exactly according to the KV-cache formula, or whether long-context performance has additional terms.

The goal is not just:

```text
How much VRAM does KV cache need?
```

The real question is:

```text
At each context depth, what part of ms/token is explained by KV reads, and what residual remains?
```

## Source baseline

The simple KV storage formula is widely used and is a good starting point:

```text
KV_bytes =
  batch_or_slots
  * layers
  * ctx_tokens
  * kv_heads
  * head_dim
  * (bytes_K + bytes_V)
```

External references:

- llama.cpp `llama-bench` supports `-d <n>` to prefill the KV cache to a fixed context depth before measuring generation. It also states benchmark numbers exclude tokenization and sampling. Source: <https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md>
- llama.cpp exposes separate K/V cache dtypes through `--cache-type-k` and `--cache-type-v`; default is `f16`, with quantized options such as `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, and `iq4_nl`. Source: <https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md>
- vLLM PagedAttention documents that KV cache is split into fixed token blocks per head, and that decode kernels process heads/sequences/partitions rather than a single flat contiguous tensor. Source: <https://docs.vllm.ai/en/latest/design/paged_attention/>
- vLLM quantized KV cache docs note that KV quantization reduces memory footprint, and that with some FlashAttention backends attention can operate directly in the quantized domain. Source: <https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/>
- vAttention argues that dynamic/paged KV layouts can add attention-kernel overhead compared with virtually contiguous cache layouts, while also reducing memory fragmentation. Source: <https://arxiv.org/html/2405.04437v2>
- KVQuant motivates KV compression because KV cache becomes the dominant memory bottleneck at long contexts. Source: <https://arxiv.org/html/2401.18079v4>

## Key hypothesis

The basic KV formula is close to complete for **storage capacity**.

It is incomplete for **decode speed**.

The decode-speed model needs at least:

```text
ms_per_token(ctx) =
  weight_ms
  + attention_kv_read_ms(ctx)
  + attention_compute_ms(ctx)
  + kv_write_ms
  + layout_indirection_ms(ctx)
  + kv_quant_dequant_ms(ctx)
  + launch_or_scheduler_ms
  + offload_or_spill_ms(ctx)
```

The linear term should dominate only when KV read bandwidth is the bottleneck and cache is fully resident on the executing device.

## Formula layer 1: storage capacity

For one active sequence:

```text
KV_storage_bytes(ctx) =
  layers
  * round_up(ctx, allocation_granularity_tokens)
  * kv_heads
  * head_dim
  * (bytes_per_K + bytes_per_V)
  + scale_metadata_bytes
  + block_table_metadata_bytes
  + padding_fragmentation_bytes
```

For multiple slots/sequences:

```text
KV_storage_total =
  sum(KV_storage_bytes(ctx_i) for each active sequence)
```

For llama.cpp server/unified-cache modes, the allocated context budget can be different from the actual used context. So measure both:

```text
ctx_allocated
ctx_used_for_decode_depth
```

## Formula layer 2: decode traffic

A first-order decode traffic model:

```text
KV_read_bytes_per_token(ctx) =
  layers
  * ctx_used
  * kv_heads
  * head_dim
  * (
      bytes_per_K_read * K_read_multiplier
    + bytes_per_V_read * V_read_multiplier
    + scale_or_metadata_bytes_per_token
    )
```

The multipliers are the important missing part.

They depend on:

- MHA vs MQA vs GQA;
- whether a kernel reuses one KV head across multiple query heads in a group;
- whether attention is paged/blocked/partitioned;
- whether K/V are quantized and dequantized before use;
- whether the attention kernel computes directly on quantized K/V;
- whether cache is on GPU, CPU, or split/offloaded;
- whether the long context causes cache/TLB/page-table/layout overhead.

## Formula layer 3: compute

For decode attention, per token:

```text
QK_score_ops(ctx) ~= 2 * layers * query_heads * head_dim * ctx
PV_ops(ctx)       ~= 2 * layers * query_heads * head_dim * ctx
softmax_ops(ctx)  ~= O(layers * query_heads * ctx)
```

This is also linear in context, but the constant differs from memory traffic. At short context or with very compressed KV, compute and softmax/reduction can become visible.

## What a fitted model should test

Collect llama.cpp `tg128` at fixed context depths:

```text
ctx = 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
```

Use `llama-bench -d <ctx>` so each generation test starts at the requested KV depth.

Fit:

```text
ms_per_token = A + B * ctx
```

Where:

- `A` is the context-independent decode cost: weight read, base kernels, launch/scheduler.
- `B` is the context-dependent slope.

Then derive:

```text
implied_kv_bandwidth =
  KV_read_bytes_per_ctx_token / B
```

If implied bandwidth is plausible and stable, the simple KV-read model explains the decline.

If implied bandwidth changes with context, or residuals bend upward, the simple formula is missing terms.

## Required measurements

For each model and cache setting:

| field | source |
|---|---|
| model path | input |
| model id | input |
| layers | GGUF metadata |
| query heads | GGUF metadata |
| KV heads | GGUF metadata |
| head dim | derived or metadata |
| max context | GGUF metadata |
| weight quant | GGUF metadata |
| K cache dtype | llama.cpp CLI arg |
| V cache dtype | llama.cpp CLI arg |
| flash attention on/off | llama.cpp CLI arg |
| offload mode | llama.cpp CLI arg |
| ctx depth | `llama-bench -d` |
| tok/s mean/stddev | llama-bench JSON |
| ms/token | derived |
| allocated/resident memory if available | rocm-smi or llama log |

## Benchmark matrix

Minimum matrix:

```text
cache K/V dtype:
  f16/f16
  q8_0/q8_0
  q4_0/q4_0, if supported and quality caveat accepted

flash attention:
  on
  off, only if it fits and is not absurdly slow

depth:
  512 -> max fit
```

For local AMD RX 7900 XTX, start with:

```text
Qwen3-8B-Q4_K_M
Qwen3-14B-Q4_K_M
Qwen3-32B-Q4_K_M, only to the contexts that fit
```

## Output

Tool:

```text
extra/llama_kv_ctx_slope_bench.py
```

Artifacts:

```text
bench/llama-kv-ctx-slope/<model-id>/latest.json
bench/llama-kv-ctx-slope/<model-id>/summary.md
```

JSON schema:

```text
tinygrad.llama_kv_ctx_slope.v1
```

Required report rows:

```text
ctx
tok_s
stddev
ms_per_token
kv_storage_bytes
kv_read_bytes_per_token_est
fit_ms_per_token
residual_ms
implied_kv_bandwidth_gb_s
```

## Interpretation rules

### Case A: KV-read explained

Criteria:

```text
linear fit R^2 high
implied_kv_bandwidth stable
residuals small
no offload/spill
```

Verdict:

```text
LLAMA_CTX_DECLINE_KV_READ_EXPLAINED
```

### Case B: quant/dequant overhead visible

Criteria:

```text
q8/q4 storage lower but ms/token slope does not fall proportionally
or q4 is slower at long context despite lower bytes
```

Verdict:

```text
LLAMA_CTX_DECLINE_KV_QUANT_OVERHEAD
```

This is plausible because quantized KV can reduce memory but add dequant or scale-load work unless the attention kernel computes directly in the quantized domain.

### Case C: layout/indirection overhead visible

Criteria:

```text
residual grows with depth even after bytes are modeled
or paged/block metadata explains extra work
```

Verdict:

```text
LLAMA_CTX_DECLINE_LAYOUT_OVERHEAD
```

### Case D: offload/spill

Criteria:

```text
sharp knee in tok/s
memory near VRAM limit
logs show CPU/KV offload or partial GPU residency
```

Verdict:

```text
LLAMA_CTX_DECLINE_OFFLOAD_OR_SPILL
```

### Case E: not enough evidence

Criteria:

```text
too few depths
high noise
missing metadata
```

Verdict:

```text
LLAMA_CTX_DECLINE_INCONCLUSIVE
```

## Acceptance

- The tool must separate storage capacity from decode-speed slope.
- The tool must not claim the KV formula is complete for speed.
- The report must fit `ms/token = A + B*ctx` and show residuals.
- The report must run at least one cache dtype setting end-to-end before any broader conclusion.
- The report must record llama.cpp command lines and build commit.
- If a context fails/OOMs, record the failure row rather than dropping it.
- No tinygrad route conclusions are made from llama results alone; this is a reference/diagnostic benchmark.

## Why this matters for positional memory

If the slope is KV-read explained, then long-context raw prompting has a predictable linear decode tax. Position-aware memory should keep the active model context small and relevant.

If the slope has quant/dequant or layout residuals, then long-context systems also need runtime-aware policies:

- choose KV dtype by depth;
- avoid over-allocating context;
- keep stable prefixes cached;
- retrieve relevant memory instead of expanding every prompt;
- prefer providers/runtimes with efficient long-context KV management when the session truly needs it.

