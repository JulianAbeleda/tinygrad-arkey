# QK Policy Pipeline Search: Qwen3-32B-Q4_K_M.gguf

| model | tensor | format | shape | research winner | runtime policy | winner GB/s | best q8 GB/s | stopped vdot | verdict |
|---|---|---|---:|---|---|---:|---:|---:|---|
| `Qwen3-32B-Q4_K_M.gguf` | `blk.0.ffn_gate.weight` | Q4_K | 25600x5120 | `q8_1_q4_intdot` | `v1_q4_packed` | 392.81 | 392.81 | 4 | q8 research win; not runtime-supported |
| `Qwen3-32B-Q4_K_M.gguf` | `blk.8.ffn_down.weight` | Q4_K | 5120x25600 | `q8_1_q4_intdot` | `q4_local32_p2` | 368.54 | 368.54 | 4 | q8 research win; not runtime-supported |
| `Qwen3-32B-Q4_K_M.gguf` | `blk.0.attn_q.weight` | Q4_K | 8192x5120 | `q4_local32_p1` | `q4_local32_p1` | 317.45 | 126.28 | 4 | keep current/fused |
| `Qwen3-32B-Q4_K_M.gguf` | `blk.0.attn_output.weight` | Q4_K | 5120x8192 | `q4_local64_p1` | `q4_local64_p1` | 256.50 | 228.13 | 4 | keep current/fused |
| `Qwen3-32B-Q4_K_M.gguf` | `blk.0.attn_k.weight` | Q4_K | 1024x5120 | `q4_local32_p4` | `q4_local32_p4` | 62.92 | 42.87 | 4 | keep current/fused |
| `Qwen3-32B-Q4_K_M.gguf` | `output.weight` | Q6_K | 151936x5120 | `fused_graph` | `fused_graph` | 121.53 | n/a | 0 | keep current/fused |
| `Qwen3-32B-Q4_K_M.gguf` | `blk.0.attn_v.weight` | Q6_K | 1024x5120 | `q6_local64_p2` | `q6_local64_p2` | 49.41 | n/a | 0 | keep current/fused |
| `Qwen3-32B-Q4_K_M.gguf` | `blk.0.ffn_down.weight` | Q6_K | 5120x25600 | `q6_local64_p2` | `q6_local64_p2` | 212.71 | n/a | 0 | keep current/fused |

## Stop-Gate Result

The semantic stop gate fired for isolated packed-dot candidates. They remain available for explicit experiments, but are not default generated-search work.

At least one q8_1 candidate won a descriptor as a research result. It is intentionally not emitted as the runtime policy because no q8_1 wrapper/full-decode gate exists.

## Interpretation

This is the machine-readable version of the current hypothesis: packed dot is not rejected as a hardware capability, but isolated packed-dot work is rejected as the next default task. A future candidate has to be a broader semantic layout/schedule/codegen package and must beat these generated gates before touching runtime policy.
