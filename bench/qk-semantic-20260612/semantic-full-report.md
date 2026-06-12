# QK Semantic Full-Shape Stop-Gated Search

| model | tensor | format | shape | research winner | runtime policy | winner GB/s | best q8 GB/s | stopped vdot | verdict |
|---|---|---|---:|---|---|---:|---:|---:|---|
| `Qwen3-8B-Q4_K_M.gguf` | `blk.0.ffn_gate.weight` | Q4_K | 12288x4096 | `q4_local64_p1` | `q4_local64_p1` | 422.31 | 243.07 | 4 | keep current/fused |
| `Qwen3-8B-Q4_K_M.gguf` | `blk.4.ffn_down.weight` | Q4_K | 4096x12288 | `q4_local32_p4` | `q4_local32_p4` | 268.74 | 257.77 | 4 | keep current/fused |
| `Qwen3-8B-Q4_K_M.gguf` | `blk.0.attn_q.weight` | Q4_K | 4096x4096 | `q4_local64_p1` | `q4_local64_p1` | 183.39 | 84.55 | 4 | keep current/fused |
| `Qwen3-8B-Q4_K_M.gguf` | `blk.0.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph` | `fused_graph` | 103.21 | 37.98 | 4 | keep current/fused |
| `Qwen3-8B-Q4_K_M.gguf` | `output.weight` | Q6_K | 151936x4096 | `fused_graph` | `fused_graph` | 119.16 | n/a | 0 | keep current/fused |
| `Qwen3-8B-Q4_K_M.gguf` | `blk.0.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph` | `fused_graph` | 87.37 | n/a | 0 | keep current/fused |
| `Qwen3-8B-Q4_K_M.gguf` | `blk.0.ffn_down.weight` | Q6_K | 4096x12288 | `q6_local64_p2` | `q6_local64_p2` | 198.19 | n/a | 0 | keep current/fused |
| `Qwen3-14B-Q4_K_M.gguf` | `blk.0.ffn_gate.weight` | Q4_K | 17408x5120 | `q4_local32_p1` | `q4_local32_p1` | 366.16 | 318.69 | 4 | keep current/fused |
| `Qwen3-14B-Q4_K_M.gguf` | `blk.5.ffn_down.weight` | Q4_K | 5120x17408 | `q8_1_q4_intdot` | `q4_local32_p2` | 328.89 | 328.89 | 4 | q8 research win; not runtime-supported |
| `Qwen3-14B-Q4_K_M.gguf` | `blk.0.attn_q.weight` | Q4_K | 5120x5120 | `q4_local64_p1` | `q4_local64_p1` | 240.23 | 105.27 | 4 | keep current/fused |
| `Qwen3-14B-Q4_K_M.gguf` | `blk.0.attn_k.weight` | Q4_K | 1024x5120 | `v1_q4_packed` | `v1_q4_packed` | 64.05 | 47.86 | 4 | keep current/fused |
| `Qwen3-14B-Q4_K_M.gguf` | `output.weight` | Q6_K | 151936x5120 | `fused_graph` | `fused_graph` | 120.59 | n/a | 0 | keep current/fused |
| `Qwen3-14B-Q4_K_M.gguf` | `blk.0.attn_v.weight` | Q6_K | 1024x5120 | `q6_local64_p2` | `q6_local64_p2` | 52.74 | n/a | 0 | keep current/fused |
| `Qwen3-14B-Q4_K_M.gguf` | `blk.0.ffn_down.weight` | Q6_K | 5120x17408 | `q6_local64_p2` | `q6_local64_p2` | 212.32 | n/a | 0 | keep current/fused |

## Stop-Gate Result

The semantic stop gate fired for isolated packed-dot candidates. They remain available for explicit experiments, but are not default generated-search work.

At least one q8_1 candidate won a descriptor as a research result. It is intentionally not emitted as the runtime policy because no q8_1 wrapper/full-decode gate exists.

## Interpretation

This is the machine-readable version of the current hypothesis: packed dot is not rejected as a hardware capability, but isolated packed-dot work is rejected as the next default task. A future candidate has to be a broader semantic layout/schedule/codegen package and must beat these generated gates before touching runtime policy.
