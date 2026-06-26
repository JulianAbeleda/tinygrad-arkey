# Decode tile: bottleneck confirmation + block-tile probe — result (2026-06-26)

Both actions from `docs/decode-block-tile-codegen-scope.md`. Net conclusion: the W==D wall is the
**renderer's ISA quality** (scalar, non-pipelined code for the generated loop), not the kernel structure
— kernel-level tweaks recover only ~15%.

## Part A — the tile is the dominant kernel (confirmed)

Per-kernel isolation timing (single eager `custom_kernel` launch under DEBUG=2, ctx 4096, S=43; the JIT
graph-replay path reports 0, so isolation is the working method):

| kernel | one launch | share |
|---|---:|---|
| `flash_fused_xlane_score_pv_tile_whole_cache_32_128` | **17,170 µs** | **100%** |
| `flash_state_gmax_32_128` | 6.3 µs | — |
| `flash_state_combine_32_128` | 19.1 µs | — |

The tile is **676× the gmax+combine path** and ≈100% of attention time (17 ms × 36 layers ≈ 618 ms ≈ the
W==D whole-step). So the combine is irrelevant; the **tile** is the target. (The combine, the corpus's
usual split-KV-economics worry, is a non-issue here.)

## Part B — block-tile probe: the barrier is not the cost

The tile at 17 ms for one launch is ~800× the 17.5 µs HBM roofline — extreme stalls from the per-token
serial chain (cross-lane reduce + online-state recurrence + scalar loads) with only ~4 waves/CU to hide
latency. Highest-leverage, lowest-risk kernel change: drop the per-token LDS K-stage + **barrier** (each
lane uses its own e-slice once — no cross-wave reuse to amortize, so the barrier was pure overhead), read
K straight from global into the fdot2.

Result: **17.2 ms → 14.6 ms (~15%)**, numerically equivalent. The barrier is **not** the dominant cost.
(Probe reverted; the validated LDS kernel is kept.)

## Conclusion — it's the renderer, not the kernel

The ~15% ceiling on a kernel-level change isolates the remaining ~800× to the **renderer**: tinygrad's AMD
renderer emits **scalar, non-software-pipelined ISA** for the generated loop (scalar fp32 loads, no
vectorization, no prefetch/pipeline, per-token cross-lane on the critical path), while hipcc emits
vectorized + LDS-block-tiled + pipelined ISA for the owned kernel. The generated/searchable path can
**express** the correct, occupancy-aware, fused tile (proven: microgate + route gate + S=48), but the
renderer cannot **lower** it to competitive ISA.

Label: `SEARCH_BLOCKED_BY_CODEGEN` at the **renderer scheduling/vectorization** level (quantified: ~800×
ISA-quality gap on the decode tile; ~15% recoverable by kernel tweaks, the rest renderer-bound). This is
the north-star `v_dot2` + cross-lane lowering frontier, now with a number.

## What this means for the next move

- **Do NOT** write another attention layout, and do not keep micro-tweaking the kernel — marginal returns.
- The lever is the **renderer**: auto-vectorize global loads (fp16 wide / `b128`), LDS-block-tile the token
  loop, and software-pipeline (prefetch next-tile loads across the compute) — so a *generated* loop lowers
  the way hipcc lowers the owned one. The ISA-diff gate (`qk_decode_attention_isa_diff_gate.py`) is the
  acceptance harness: re-diff after each renderer change; target LDS↑, fp16-vec loads >0, cross-lane/token↓.
- Reusable assets from this turn: the **per-kernel isolation-timing method** (Part A — works where the JIT
  DEBUG=2 path returns 0) and the confirmation that the **tile, not the combine,** is the sole target.
