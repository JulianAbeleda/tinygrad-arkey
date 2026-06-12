# QK Policy Pipeline Search: Qwen3-8B-Q4_K_M.gguf

| model | tensor | format | shape | research winner | runtime policy | winner GB/s | best q8 GB/s | stopped vdot | verdict |
|---|---|---|---:|---|---|---:|---:|---:|---|
| `Qwen3-8B-Q4_K_M.gguf` | `blk.0.ffn_gate.weight` | Q4_K | 12288x4096 | `v1_q4_packed` | `v1_q4_packed` | 420.55 | 242.03 | 4 | keep current/fused |
| `Qwen3-8B-Q4_K_M.gguf` | `blk.4.ffn_down.weight` | Q4_K | 4096x12288 | `v1_q4_packed` | `v1_q4_packed` | 266.88 | 231.58 | 4 | keep current/fused |
| `Qwen3-8B-Q4_K_M.gguf` | `blk.0.attn_q.weight` | Q4_K | 4096x4096 | `q4_local64_p1` | `q4_local64_p1` | 184.61 | 83.26 | 4 | keep current/fused |
| `Qwen3-8B-Q4_K_M.gguf` | `blk.0.attn_k.weight` | Q4_K | 1024x4096 | `fused_graph` | `fused_graph` | 100.05 | 37.94 | 4 | keep current/fused |
| `Qwen3-8B-Q4_K_M.gguf` | `output.weight` | Q6_K | 151936x4096 | `fused_graph` | `fused_graph` | 123.85 | n/a | 0 | keep current/fused |
| `Qwen3-8B-Q4_K_M.gguf` | `blk.0.attn_v.weight` | Q6_K | 1024x4096 | `fused_graph` | `fused_graph` | 86.06 | n/a | 0 | keep current/fused |
| `Qwen3-8B-Q4_K_M.gguf` | `blk.0.ffn_down.weight` | Q6_K | 4096x12288 | `q6_local64_p2` | `q6_local64_p2` | 197.68 | n/a | 0 | keep current/fused |

## Stop-Gate Result

The semantic stop gate fired for isolated packed-dot candidates. They remain available for explicit experiments, but are not default generated-search work.

No q8_1 candidate won the representative descriptors. The generated search therefore stops at the current v1/fused policy.

## Interpretation

This is the machine-readable version of the current hypothesis: packed dot is not rejected as a hardware capability, but isolated packed-dot work is rejected as the next default task. A future candidate has to be a broader semantic layout/schedule/codegen package and must beat these generated gates before touching runtime policy.
