# Packed QK Tile Lowering Analysis

Repeated 8B Q4_K microbench comparison of the current v1 partial kernel
against the raw custom `PackedQKTile` consumer. This is a diagnostic gate,
not a runtime-promotion artifact.

## Decision

- decision: `diagnose_only_not_promoted`
- promotion gate: median tile_custom gain >=10% on every measured Q4_K tensor before full decode
- measured tensors: `5`
- gain range: `-2.04%` to `7.51%`

## Comparison

| tensor | v1 median Q4 GB/s | tile median Q4 GB/s | gain % | v1 median ms | tile median ms | tile max_abs median |
|---|---:|---:|---:|---:|---:|---:|
| `blk.0.attn_k.weight` | 17.55 | 17.20 | -2.04 | 0.134409 | 0.137208 | 0.00199986 |
| `blk.0.attn_output.weight` | 69.27 | 68.51 | -1.10 | 0.136243 | 0.137756 | 0.00320601 |
| `blk.0.attn_q.weight` | 69.11 | 68.86 | -0.36 | 0.136553 | 0.137041 | 0.00238299 |
| `blk.0.ffn_gate.weight` | 205.95 | 207.52 | 0.76 | 0.137465 | 0.136430 | 0.00179291 |
| `blk.0.ffn_up.weight` | 198.00 | 212.86 | 7.51 | 0.142989 | 0.133007 | 0.00340462 |

## Interpretation

The raw custom tile path has a positive signal on at least one tensor, but
does not clear the full-decode promotion bar across the measured Q4_K set.
Do not integrate it into runtime from this result. The next step is source,
assembly, or counter analysis to explain why vector-source loads produce
only a partial bandwidth improvement.
