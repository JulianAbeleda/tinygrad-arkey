# Decode tile: confirm bottleneck + block-tile the generated codegen — scope (2026-06-26)

Continues `docs/decode-isa-diff-gate-result.md`. The ISA diff pinned the W==D wall to a codegen-strategy
difference (owned **block-processes** tokens; the generated tile is **per-token**). Two actions:

## Part A — confirm the tile is the dominant attention kernel (dynamic per-kernel ms)

The ISA diff is static; it cannot say whether the time is in the **tile**, the **48-split combine**, or the
gmax. The DEBUG=2 timing path returns 0 under JIT graph replay (the per-program `*** … arg N mem … Xms`
lines only appear in eager execution). Plan:
- Diagnose the real DEBUG=2 line format in this fork, then capture an **eager** decode step (concrete
  `start_pos`, no TinyJit) under DEBUG=2 and parse per-program GPU ms for the xlane route (tile / gmax /
  combine) and the owned route (tile / combine). Fall back to `PROFILE=1` if eager lines are unavailable.
- Gate: if the **tile** dominates (expected from the ISA diff), Part B targets the tile. If the **combine**
  dominates, retarget Part B to the 48-split LSE combine instead. Artifact:
  `bench/qk-decode-attention-kernel-timing/latest.json`.

## Part B — block-tile the generated tile, validate, re-diff

The minimal high-leverage change, from the three pinned bleeders (LDS 256→8192, fp16-vec loads, amortized
cross-lane). Restructure `flash_fused_xlane_score_pv_tile_whole_cache_kernel` from per-token to block-tiled:

1. **K-block LDS staging.** Stage a block of `B` tokens of K into LDS once (B·Hd halfs), with **one barrier
   per block** instead of one per token. Inner loop reads K from LDS.
2. **Vectorized loads.** Load K/V as fp16 (and wider where the renderer allows) rather than scalar fp32.
3. **Amortized cross-lane.** Keep the e-shard fdot2 + cross-lane reduce, but issue it from LDS-resident K
   across the block so the warp_reduce is not gated behind a per-token global load + barrier.

Pick `B` so the K-block fits LDS comfortably (owned uses 8 KB ⇒ B·Hd·2 ≤ ~8 KB ⇒ B ≈ 32 at Hd=128). Keep
the d-sharded PV and the split count S=48 (occupancy baseline). Process:
- (a) prototype the block-tiled kernel in the microgate (`...fused_xlane_score_pv_microgate.py`), validate
  numerically against the scalar oracle (scalar fp32 1e-7; fp16 ~2e-5) across the existing shapes;
- (b) port into `qk_flash_decode.py` (raw `cache_kv` 5D), run the route gate (token-match + clean);
- (c) **re-diff with the ISA-diff gate** — expect LDS ↑ to multi-KB, fp16-vec loads > 0, cross-lane/token ↓;
- (d) only if the ISA materially improves, re-run W==D vs baseline.

## Acceptance / forks

- Part A: a per-kernel ms table with a clear dominant kernel. If the tile is <50% of attention time,
  retarget.
- Part B (a): if the block-tiled UOp kernel cannot be expressed/lowered correctly (UOp-verify or numeric
  fail), that is `SEARCH_BLOCKED_BY_CODEGEN` at the **block-tile-expressibility** level — record it; the next
  step would be a renderer auto-tiling/vectorization pass, not more kernel authoring.
- Part B (c): if the UOp is block-tiled but the renderer still emits scalar/per-token ISA (LDS/vec
  unchanged in the re-diff), that isolates the gap to the **renderer** (it does not honor the block
  structure) — a precise, different finding than "the kernel wasn't block-tiled."
- Do NOT write another attention *layout*; this is strictly about codegen strategy (block vs per-token).
