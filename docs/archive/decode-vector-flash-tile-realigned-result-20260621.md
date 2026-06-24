# Decode Vector Flash-Decode Tile — Realigned Result (split-count on gqa_coop_vec)

Date: 2026-06-21

Scope: `docs/decode-vector-flash-tile-implementation-scope-20260621.md` (realigned to the corrected decode
principle: *manufacture parallel work via KV-splits + GQA columns while preserving reuse; never reduce
workgroups*).

Verdict: **First gate PASSED, W==D gate FAILED → `REST_DECODE`.** Applying the principle to the **existing
winner** `gqa_coop_vec` (more KV-splits via FLASH_L downward — the untested direction, on the path that already
has the matmul-q·k + GQA-coop reuse) gives a real **1.08× standalone attention win @ctx1024** (clears ≥1.05×), but
the whole-decode **W==D win is only +1.8%@1024 / +2.8%@512 and regresses −1.2%@4096** — below the ≥5% promotion
gate. Default decode behavior NOT changed.

## The realignment

The prior attempt built *new* hand-tiles (`qk_decode_{vector,warp}_flash_tile_ab.py`) whose hand-rolled per-key
dots cannot beat `gqa_coop_vec`'s **matmul** q·k. The corrected principle points instead at the existing winner:
`gqa_coop_vec` already preserves reuse + the matmul; it was just **under-split** (FLASH_L=128 → S=8 → 64
workgroups). Prior FLASH_L tests went the *wrong* way (256/512 = fewer splits, regressed). **FLASH_L downward
(more splits) was untested** — the exact principle lever, no new kernel, no reuse loss, no workgroup reduction.

## First gate — standalone attention vs current `gqa_coop_vec` (FLASH_L=128), clock-pinned, byte-exact

| ctx1024, FLASH_L | splits S | workgroups | µs | vs L=128 |
|---:|---:|---:|---:|---:|
| 128 (default) | 8 | 64 | 106.4 | 1.00× |
| **64** | 16 | 128 | **98.3** | **1.08×** ✓ |
| 32 | 32 | 256 | 98.9 | 1.08× |
| 16 | 64 | 512 | 112.7 | 0.94× (regress) |
| 8 | 128 | 1024 | 142.8 | 0.75× (regress) |

**Principle confirmed:** more splits (64→128 workgroups) speeds attention 1.08× @ctx1024 — clears the ≥1.05×
first gate. But there is a **sweet spot at L=64**: finer (L≤16) regresses because tinygrad's combine-over-splits
(`flash_gmax/den/combine`) costs more than llama's stream-k fixup. @ctx4096 the sweep is flat at best (L=64 ≈
1.00×, finer regresses).

## W==D whole-decode gate (clean wall, PROFILE=0, auto clock, FLASH_L=64 vs default 128)

| ctx | default 128 tok/s | FLASH_L=64 tok/s | Δ | gate |
|---:|---:|---:|---:|---|
| 512 | 68.1 | 70.0 | **+2.8%** | (improvement, not a regression — ok) |
| 1024 | 66.4 | 67.6 | **+1.8%** | need ≥5% → **FAIL** |
| 4096 | 60.6 | 59.9 | **−1.2%** | need ≥7% → **FAIL (regresses)** |

host-sync 0% throughout; byte-exact (flash exact-vs-SDPA up to fp reassociation; same variant, greedy-stable).

**W==D gate FAILED:** +1.8%@1024 (< 5%) and −1.2%@4096 (a regression, not the ≥7% win). The attention win is real
but (a) attention is only ~23% of decode so 1.08× → ~1.8% whole-decode, and (b) a fixed L=64 over-splits at
ctx4096 (S=64) where the combine overhead regresses it.

## Classification (why it misses the W==D gate)

- **Principle is correct and confirmed** — more KV-splits (64→128 workgroups) does speed `gqa_coop_vec` attention
  at short ctx, on the path that preserves reuse + the matmul. This is the *right* lever, validated.
- **But two ceilings keep it sub-gate:** (1) attention is only ~23% of decode @ctx1024, so even a solid attention
  win is ~1.8% whole-decode; (2) tinygrad's **combine-over-splits cost** caps the useful split count (~16-32)
  well below llama's many-split regime (48-144) — finer splits regress, so tinygrad cannot reach llama's
  occupancy without a **stream-k / efficient many-split combine**, which is the deeper engineering gap.
- The −1.2%@4096 is the over-split combine overhead at a fixed L; a ctx-adaptive split target (~16 splits) would
  avoid the regression but still nets < 5% (the ~23% attention ceiling), so it does not change the gate outcome.

## Decision: `REST_DECODE`

The realigned, principle-correct lever (more splits on `gqa_coop_vec`) **passes the standalone first gate** but
**does not clear the ≥5% W==D promotion gate** and regresses at ctx4096. Per the directive (gate miss → stop,
classify, REST; do not tune blindly), **rest decode** at the current route (~86 tok/s @ctx0, 68/66/61
@ctx512/1024/4096, ~67% llama; q8 opt-in +~7%).

The only remaining decode lever is the full llama-class `flash_attn_tile` (many splits + **efficient stream-k
combine** + K-tile-batched vectorized body) — the multi-week north-star codegen project, to be funded separately.

**Owner note (marginal, optional):** FLASH_L=64 is a *measured, byte-exact* small win at short context
(+2.8%@512, +1.8%@1024) that regresses −1.2%@4096. A ctx-gated `FLASH_L` (≈64 below ctx~2048, 128 above) could
ship ~2% at short context as a default-off opt-in — but it is sub-gate and below auto-clock W==D noise, so it is
**not** promoted here; surfaced only for the owner's call.

## Gates status

| gate | result |
|---|---|
| correctness byte-exact | PASS (err 0.000) |
| standalone ≥1.05× vs gqa_coop_vec @ctx1024 | **PASS (1.08×, FLASH_L=64)** |
| W==D ≥5%@1024 or ≥7%@4096, no ctx512 regress | **FAIL (+1.8%@1024, −1.2%@4096)** → REST_DECODE |
| tok0/dNLL | clean (same variant, greedy-stable) |

## Artifacts

- inline sweep (this doc); `bench/qk-decode-vector-flash-tile/warp_tile_ab.json` (prior hand-tile levers)
- W==D logs: FLASH_L=128 vs 64 (ctx128/512/1024/4096), clean wall auto clock
- lifecycle ledger updated.

## Boundary

No decode default changed (`model.py` untouched; FLASH_L is an existing env, default 128). Clock pinned only for
the standalone diagnostic; `auto` restored (verified). No WMMA, no MMVQ, no launch-count metric, no
workgroup-reducing tile, compared only against the current `gqa_coop_vec`.
