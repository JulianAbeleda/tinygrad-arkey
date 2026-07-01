# Prefix/cache reuse benchmark scope

## Decision

Prefix/cache reuse is a **third benchmark category**, separate from raw prefill and raw decode.

- **Prefill benchmark:** cold prompt processing throughput for uncached input tokens.
- **Decode benchmark:** autoregressive output throughput after context exists.
- **Prefix/cache benchmark:** session/runtime reuse, measuring how much prefill is skipped when requests share a stable prompt prefix.

This belongs in **tinygrad** first because tinygrad owns the loaded model, KV cache, prefix cache, and `Transformer.get_start_pos()` reuse policy. BoltBeam can ingest the resulting JSON later, but BoltBeam should not execute the runtime benchmark.

## External baseline

The benchmark follows the same separation used by serving systems and benchmark terminology:

- vLLM Automatic Prefix Caching caches KV-cache blocks and reuses them when a new request has the same prefix: <https://docs.vllm.ai/en/stable/design/prefix_caching/>
- BentoML describes prefix caching as skipping recomputation of shared prompt prefixes by reusing KV cache: <https://bentoml.com/llm/inference-optimization/prefix-caching>
- OpenAI Prompt Caching reduces latency and cost by reusing recently processed prompt prefixes: <https://developers.openai.com/api/docs/guides/prompt-caching>
- Anthropic Prompt Caching supports automatic and explicit cache breakpoints: <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>
- The IETF LLM benchmarking terminology draft states that prefix caching affects input throughput and that tests must specify whether input throughput counts all input tokens or only cache misses: <https://datatracker.ietf.org/doc/html/draft-gaikwad-llm-benchmarking-terminology-00>

## Equation

For one request:

```text
prompt_tokens = cached_prefix_tokens + uncached_prefill_tokens
output_tokens = decode_tokens

request_time ~= cache_lookup/reuse
              + prefill(uncached_prefill_tokens)
              + decode(output_tokens)
```

This is why cached-token performance must not be reported as raw prefill speed. It is a runtime/session reuse result.

## Tool

Implemented:

```text
extra/qk_prefix_cache_bench.py
```

Output:

```text
bench/qk-prefix-cache-reuse/latest.json
bench/qk-prefix-cache-reuse/summary.md
```

Schema:

```text
tinygrad.prefix_cache_reuse_bench.v1
```

## Cases

The benchmark runs three required cases:

| case | purpose |
|---|---|
| `cold_full` | no reusable prefix; full prompt is uncached prefill |
| `warm_same_prefix_changed_suffix` | stable prefix reused, changed suffix prefills |
| `prefix_broken_changed_front` | early-token change breaks prefix reuse |

## Metrics

Required metrics:

| metric | meaning |
|---|---|
| `prompt_tokens` | total input tokens for the request |
| `cached_prefix_tokens` | tokens skipped by `Transformer.get_start_pos()` |
| `uncached_prefill_tokens` | `prompt_tokens - cached_prefix_tokens` |
| `cache_hit_ratio` | `cached_prefix_tokens / prompt_tokens` |
| `ttft_ms` | time to first generated token; includes uncached prefill plus first decode |
| `total_ms` | full request time for the benchmarked completion length |
| `effective_input_tok_s_all_tokens` | user-visible apparent input throughput |
| `effective_input_tok_s_cache_miss_only` | compute-accounting throughput for uncached tokens only |

The tool pre-warms the benchmark prompt shapes before recording rows. That keeps cold rows from including one-time JIT compile cost; cold means "no reusable prompt prefix," not "uncached compiled kernels."

## Acceptance

- The benchmark is not presented as raw prefill or decode.
- The JSON records both all-token and cache-miss-only effective input throughput.
- `warm_same_prefix_changed_suffix` must report nonzero `cached_prefix_tokens`.
- `prefix_broken_changed_front` must report a materially lower hit ratio than the warm same-prefix case.
- The benchmark does not change runtime defaults.
- BoltBeam ingestion is future work over the emitted JSON schema.

## Example

```bash
DEV=AMD JIT=1 PYTHONPATH=. python extra/qk_prefix_cache_bench.py \
  --model /home/ubuntu/models/Qwen3-0.6B-Q8_0.gguf \
  --max-context 2048 \
  --stable-prefix-tokens 1024 \
  --suffix-tokens 128 \
  --decode-tokens 16
```
