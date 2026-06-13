# Packed QK Tile Lowering Analysis

Date: 2026-06-13

## Verdict

`diagnose_only_not_promoted`

The raw custom `PackedQKTile` Q4_K lowering now has repeated 8B microbench data
against the v1 partial kernel. It does not clear the full-decode promotion gate.

Source-shape evidence:

- v1 partial DEBUG=4 source: `u32_scalar`.
- `tile_custom` DEBUG=4 source: `vector_u32x4`.
- report: `source/load-width-report.md`.

Repeated microbench evidence:

- model: `~/models/Qwen3-8B-Q4_K_M.gguf`
- runs: `5`
- timed iterations per run: `3`
- tensors: `ffn_gate`, `ffn_up`, `attn_output`, `attn_q`, `attn_k`

Decision table: `analysis.md`.
Raw run JSON: `raw/`.

## Result

Gain range is `-2.04%` to `+7.51%`, with median gain `-0.36%`.

Only `blk.0.ffn_up.weight` shows a material positive signal. The other four
Q4_K tensors tie or regress:

| tensor | gain |
|---|---:|
| `blk.0.attn_k.weight` | `-2.04%` |
| `blk.0.attn_output.weight` | `-1.10%` |
| `blk.0.attn_q.weight` | `-0.36%` |
| `blk.0.ffn_gate.weight` | `+0.76%` |
| `blk.0.ffn_up.weight` | `+7.51%` |

The pre-registered gate was `>=10%` median improvement across measured Q4_K
tensors before any full-decode promotion. This result fails that gate.

## Interpretation

The custom source does change the emitted source shape from scalar `u32` to
`vector_u32x4`, but that alone is not a general bandwidth win. The next useful
compiler-research step is not runtime integration. It is either:

- source/assembly/counter analysis to explain why vector-source loads do not
  translate into broad bandwidth gains, or
- a true core renderer/PatternMatcher semantic op that can expose packed QK
  load/decode structure to tinygrad instead of leaving it inside raw
  `Ops.CUSTOM`.
