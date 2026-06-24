# Decode Vector Flash-Decode Tile — Build Result (first gate)

Date: 2026-06-21

Scope: `docs/decode-vector-flash-tile-implementation-scope-20260621.md`

Verdict: **FIRST GATE FAILED → `REST_DECODE`.** Two bounded vector-tile levers (many KV-splits + warp-cooperative
q·k) were built, byte-exact, and measured against the current winner `gqa_coop_vec`. Best result is **0.60×
@ctx1024** (and 0.96× @ctx4096) — it does **not** clear ≥1.05×. Classified blocker: the tile is now
**work-bound on the per-key serial warp-reduce**, and matching `gqa_coop_vec` (let alone llama) requires llama's
full `flash_attn_tile` engineering (K-tile batching, vectorized fp16, register blocking) — a multi-day faithful
port, not a bounded build. Default decode behavior NOT changed.

## Measurements (clock-pinned warm, byte-exact vs reference, vs `gqa_coop_vec`)

### Lever 1 — split-count sweep (redundant per-thread dot + LDS), `extra/qk_decode_vector_flash_tile_ab.py`

| ctx1024 splits S | workgroups | tile µs | vs gqa_coop_vec (105µs) |
|---:|---:|---:|---:|
| 8 (the prior failed setting) | 64 | 507.7 | 0.21× |
| 32 | 256 | 225.4 | 0.47× |
| 64 | 512 | 219.5 | 0.48× |
| 96 | 768 | 206.8 | 0.51× |
| 128 | 1024 | 206.5 | **0.51× (plateau)** |

More KV-splits helped (0.21→0.51×, occupancy) but **plateaus at 0.51×** — the per-thread q·k redundancy (128
d-threads each recompute the full 128-dim dot) is the wall.

### Lever 2 — warp-cooperative q·k (llama structure), `extra/qk_decode_warp_flash_tile_ab.py`

Warp-per-query-head, each lane holds 4 head-dims, q·k via `ds_bpermute` warp-butterfly (no redundancy), LDS K/V
staging, register online-softmax.

| ctx | best tile µs | gqa_coop_vec µs | best vs coop | err |
|---:|---:|---:|---:|---:|
| 1024 | 177.7 (S=16) | 105.8 | **0.60×** | 0.000 |
| 4096 | 181.2 (S=96) | 174.5 | **0.96×** | 0.000 |

Cooperative dot improved 0.51→0.60× and is **byte-exact**. But the time is **flat across split count**
(177µs at S=16 through S=128) → no longer occupancy-bound; it is **work-bound** on the per-key cooperative-dot
loop.

## Classification (into the scope's buckets)

- **NOT occupancy** — lever 1 fixed it (more splits → more workgroups); lever 2's time is flat across 128→1024
  workgroups.
- **NOT correctness / V-reuse** — both byte-exact (err 0.000); LDS K/V staging works.
- **The blocker is per-key compute efficiency.** My tile's q·k is a serial per-key warp-reduce
  (latency-bound `ds_bpermute` chain + scalar fp32, one key at a time). `gqa_coop_vec` computes q·k as a **matmul**
  (efficient GEMM over all heads/keys), which tinygrad lowers far better than a hand-rolled per-key warp loop —
  so the hand-tile (177µs) cannot even beat the matmul-based split path (105µs).
- **llama's `flash_attn_tile` (9.2µs/layer, 11× faster than `gqa_coop_vec`)** achieves its speed with engineering
  this prototype does not have: `nbatch_fa=64` K-tile batching (accumulate many keys' partial dots in registers,
  reduce once per tile — not per key), vectorized fp16 throughput, and register blocking. Reproducing that is a
  **multi-day faithful kernel port** (the deep-codegen / north-star lane), not a bounded tile.

## Decision: `REST_DECODE`

The audit-named lever (non-WMMA vector flash-decode tile) is **real and proven by llama**, and the bounded
versions (split-count + warp-cooperative dot) **measurably improve** but do not clear the ≥1.05× gate vs
`gqa_coop_vec`. The remaining distance requires llama's full `flash_attn_tile` engineering = a multi-day deep
kernel project. Per the gate, **rest decode** at the current route (~86 tok/s @ctx0, 68/66/61 @ctx512/1024/4096,
~67% llama; q8 opt-in +~7%).

The lever is **not refuted as impossible** (llama proves a vector tile can be 11× faster than `gqa_coop_vec`); it
is **refuted as a bounded build** — only a full llama-class `flash_attn_tile` port (K-tile-batched, vectorized
fp16, register-blocked) would clear the gate, and that is the explicit multi-week north-star codegen effort, to be
funded separately if at all.

## Gates status

| gate | result |
|---|---|
| correctness byte-exact vs ref | PASS (err 0.000, both levers) |
| ≥1.05× vs gqa_coop_vec @ctx1024 | **FAIL** (best 0.60×) → REST_DECODE |
| one-layer / W==D | not run (first gate failed) |

## Artifacts

- `extra/qk_decode_vector_flash_tile_ab.py` (split sweep), `extra/qk_decode_warp_flash_tile_ab.py` (warp tile)
- `bench/qk-decode-vector-flash-tile/{split_sweep_ab,warp_tile_ab}.json`
- lifecycle ledger updated (vector-tile bounded build refuted; full-engineering port remains the only lever).

## Boundary

No decode default changed (`tinygrad/llm/model.py` untouched; tiles are research harnesses in `extra/`). Clock
pinned for diagnostics; `auto` restored (verified). No WMMA, no MMVQ reopen, no launch-count metric.
