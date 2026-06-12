# QK Policy Pipeline Search: Qwen3-14B-Q4_K_M.gguf

| model | tensor | format | shape | research winner | runtime policy | winner GB/s | best q8 GB/s | stopped vdot | verdict |
|---|---|---|---:|---|---|---:|---:|---:|---|
| `Qwen3-14B-Q4_K_M.gguf` | `blk.0.ffn_gate.weight` | Q4_K | 17408x5120 | `q4_local32_p1` | `q4_local32_p1` | 382.30 | 323.49 | 4 | keep current/fused |
| `Qwen3-14B-Q4_K_M.gguf` | `blk.5.ffn_down.weight` | Q4_K | 5120x17408 | `q4_local32_p2` | `q4_local32_p2` | 320.35 | 317.83 | 4 | keep current/fused |
| `Qwen3-14B-Q4_K_M.gguf` | `blk.0.attn_q.weight` | Q4_K | 5120x5120 | `v1_q4_packed` | `v1_q4_packed` | 240.31 | 104.34 | 4 | keep current/fused |
| `Qwen3-14B-Q4_K_M.gguf` | `blk.0.attn_k.weight` | Q4_K | 1024x5120 | `v1_q4_packed` | `v1_q4_packed` | 64.01 | 43.10 | 4 | keep current/fused |
| `Qwen3-14B-Q4_K_M.gguf` | `output.weight` | Q6_K | 151936x5120 | `fused_graph` | `fused_graph` | 120.85 | n/a | 0 | keep current/fused |
| `Qwen3-14B-Q4_K_M.gguf` | `blk.0.attn_v.weight` | Q6_K | 1024x5120 | `q6_local64_p2` | `q6_local64_p2` | 49.57 | n/a | 0 | keep current/fused |
| `Qwen3-14B-Q4_K_M.gguf` | `blk.0.ffn_down.weight` | Q6_K | 5120x17408 | `q6_local64_p2` | `q6_local64_p2` | 210.11 | n/a | 0 | keep current/fused |

## Stop-Gate Result

The semantic stop gate fired for isolated packed-dot candidates. They remain available for explicit experiments, but are not default generated-search work.

No q8_1 candidate won the representative descriptors. The generated search therefore stops at the current v1/fused policy.

## Interpretation

This is the machine-readable version of the current hypothesis: packed dot is not rejected as a hardware capability, but isolated packed-dot work is rejected as the next default task. A future candidate has to be a broader semantic layout/schedule/codegen package and must beat these generated gates before touching runtime policy.
