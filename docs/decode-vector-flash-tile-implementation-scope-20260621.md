# Decode Vector Flash-Decode Tile — Implementation Scope

Date: 2026-06-21

Owner: Claude

Scope upstream: `docs/llama-decode-primitive-difference-audit-result-20260621.md` (decision `VECTOR_FLASH_DECODE_TILE`)

## Objective

Build a **non-WMMA vector flash-decode tile** matching llama's T=1 occupancy strategy, and gate it against the
current winner `gqa_coop_vec`. **Not WMMA. Not MMVQ. Launch count is not the metric** (wall time vs `gqa_coop_vec`).

## Target design (from the llama `flash_attn_tile` audit)

- **Many KV parallel blocks / splits scaling with context** — llama used parallel_blocks ≈ 48/80/144 @ctx
  512/1024/4096; tinygrad `gqa_coop_vec` uses a FIXED 8 (FLASH_L=128) → ~64 blocks → occupancy-starved at T=1.
  This is the primary lever: split the KV into many blocks so 8 kv-heads × P splits fills the GPU.
- **LDS K/V staging, ~10KB-class tile** — stage this (kv-head, split)'s key range once in LDS, reuse across the
  query-head columns and all threads (llama: 10.7KB).
- **GQA query-head packing into columns, not serial G** — process the G=4 query heads of a kv-head together
  (column-packed), preserving query-head parallelism; do NOT serialize G (the prior fused-tile mistake).
- **register online-softmax** — running max/sum in registers, exp-rescale flash style.
- **separate combine/fixup only if needed** — combine the P split partials (reuse the existing flash combine);
  add stream-k fixup only if the split imbalance costs measurably.

## First gate (decisive)

Standalone decode-shape tile (Hq=32, Hkv=8, Hd=128) must **beat `gqa_coop_vec` by ≥1.05× @ctx1024** (warm,
clock-pinned, byte-exact vs reference). Compare against the **current `gqa_coop_vec`**, not the old raw-fused /
global-reread paths.

- If it passes: continue to one-layer in-model + W==D (≥5%@1024 / ≥7%@4096, no ctx512 regress >1%), then a UOp
  port for integration behind a default-off `FLASH_VARIANT`.
- **If it fails: classify the blocker (occupancy / cooperative-dot cost / combine overhead / LDS / register) and
  return `REST_DECODE`.** Do not tune blindly.

## Build order (cheapest lever first)

1. **Split-count sweep** — the cheapest, highest-value test: take the existing tile and crank the split count
   (parallel_blocks 8 → 32 → 64 → 96), measure vs `gqa_coop_vec`. Many splits + LDS may already move it (my prior
   fused-LDS tile failed at a FIXED 8 splits — the wrong setting).
2. If split-count alone is insufficient, add **warp/cooperative q·k** (avoid the per-thread 128× dot redundancy) +
   **query-head column packing**.
3. Combine over P splits (existing `flash_reduce`/combine); add stream-k fixup only if needed.

## Constraints

No model.py / default change; default-off only; clock-pinned local diagnostic restored to `auto`; correctness
byte-exact vs reference; artifacts under `bench/qk-decode-vector-flash-tile/`. No WMMA, no MMVQ reopen, no
launch-count metric.

## Artifacts

- `extra/qk_decode_vector_flash_tile_ab.py`, `bench/qk-decode-vector-flash-tile/*.json`
- `docs/decode-vector-flash-tile-result-20260621.md` (gate result + classification)
