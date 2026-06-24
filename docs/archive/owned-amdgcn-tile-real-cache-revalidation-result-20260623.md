# Owned AMDGCN Tile Real-Cache Revalidation — Result (2026-06-23)

## Verdict: **OWNED_TILE_REAL_CACHE_CTX1024_BLOCKED**

The owned AMDGCN tile was **broken for real in-model decode by a dtype-contract bug** — it reads `__half` K/V but
the canonical `cache_kv` is **fp32**, so it read fp32 bytes as fp16 → NaN K → garbage tokens from decode step 1.
**Fixed** by casting Q/K/V to fp16 before the tile. Revalidated: **byte-identical to gqa for 64 tokens on two
prompts**, default gqa decode **unchanged**, and **W==D +11.5%@ctx2048 / +16.0%@ctx4096**. It stays
`default_eligible=false` only because the route is **ctx-restricted to ≥2048** (cannot fire at ctx1024). Source
fix **shipped** in the default-off owned-tile route; default decode byte-identical.

## 1. Verdict
`OWNED_TILE_REAL_CACHE_CTX1024_BLOCKED` — correctness PASS, W==D positive where the route fires (ctx≥2048),
blocked at ctx1024 by the route's ctx restriction (not a correctness or dtype issue).

## 2. Why this revalidation was required
The prior task (`RUNTIME_KV_ARG_PATCH_VALUES_CORRECT_DATA_STALE`) proved GraphRunner arg patching is correct and
that the owned-tile route (B4, fp32 cache, no RUNTIME_KV) bakes in real decode while gqa works — pointing at the
owned tile's data contract, not KV-cache machinery.

## 3. What prior B4/B5 validation missed
B4/B5 "byte-identical W==D" used the W==D harness with a **degenerate/uninitialized (zero) cache**: zero fp32
bytes read as fp16 are still zero, so the dtype bug never manifested. Real K (~222 magnitude) read as fp16 = NaN.
**Never claim owned-tile correctness from a degenerate cache.**

## 4. Reproduction of the real-cache failure — `OWNED_TILE_REAL_CACHE_FAIL_REPRODUCED`
`extra/qk_owned_tile_real_cache_repro.py`, real chunked-prefill (T=512×4 → ctx2048) + 8 decode steps, token
correctness:
- gqa baseline: `[38835, 34208, 13, 279, 3974, 13876, 38835, 34208]` ✓
- owned tile **fp32 cache** (no cast): `[151936 ×8]` — **broken**
- owned tile **+ fp16 cast**: byte-identical to gqa ✓
`cache_kv.dtype = dtypes.float` (fp32) confirmed.

## 5. Dtype / layout contract
- `self.cache_kv` in canonical decode is **fp32** (`dtypes.float`).
- `owned_flash_tile_gqa` (`extra/qk_owned_flash_decode.hip`) reads `const __half* K/V/Q` and Q·K via
  `__builtin_amdgcn_fdot2` (fp32 accumulate). The kernel is correct **standalone** (finite + rel_rmse e-7 at K
  magnitude 0.5…200) — the failure is purely feeding it fp32 bytes as `__half`.
- Q was already fp16 in-model (read fine); only the cache-sourced K/V were fp32 → NaN.

## 6. Fix candidates tested
- **Candidate C (chosen) — fp16 cast before the tile**: `OWNED_TILE_STAGING_DIAGNOSTIC_PASS` → promoted to the
  mandatory contract. The owned-tile route now unconditionally does `q.cast(fp16)`, `assigned_kv[0,0].cast(fp16)`,
  `assigned_kv[1,0].cast(fp16)`. fp16→fp16 is a no-op, so it is safe for any future fp16-cache route. Despite the
  cast materializing the cache prefix in fp16, the route is **still faster** than gqa (§8).
- Candidate A (native fp16 cache) and B (fp32-aware kernel) not needed for correctness; A is the path to drop the
  cast copy for a larger W==D win (follow-on).

## 7. Correctness result — `OWNED_TILE_REAL_CACHE_CORRECTNESS_PASS`
Real chunked-prefill + decode, token authority: **64-token greedy byte-identical to gqa on two prompts**, no
`151936` collapse, no NaN. Default gqa decode **byte-identical** (`[279, 1156, 22148, …]`) — the cast is inside
the default-off owned-tile branch only. (`bench/qk-owned-amdgcn-tile-real-cache/correctness.json`.)

## 8. W==D result
W==D (fixed bound start_pos, `.item()` in window, median-of-40, warm-8):

| ctx | gqa tok/s | owned+fp16cast tok/s | Δ |
|---|---|---|---|
| 2048 | 71.5 | **79.7** | **+11.5%** |
| 4096 | 67.1 | **77.8** | **+16.0%** |

ctx512/1024: the route is ctx-restricted (`DECODE_ATTN_AMDGCN_MIN_CTX=2048`) → does not fire → no change → the
**+5%@ctx1024 gate is BLOCKED**. ctx4096 clears the +7% bar. (`bench/qk-owned-amdgcn-tile-real-cache/wd.json`.)
Caveat: single-process median-40 (not in-process A/B); the delta is large and consistent with the B3 owned-tile
2.35× GPU-busy advantage.

## 9. Registry / default decision
`candidates.json` `decode_attention_llama_flash_tile_owned_amdgcn_b4.real_cache_status` updated
`BROKEN_REAL_CACHE` → **`FIXED_REAL_CACHE`** with the fix, revalidation, and W==D numbers. **`default_eligible`
stays false** — the ctx≥2048 restriction means it cannot serve ctx1024; a short-ctx-correct variant is the
prerequisite for universal promotion. No default flip.

## 10. Runtime-KV implication
The owned tile is now **real-cache-correct**, so runtime-KV is **unblocked on the owned-tile/dtype side**. But:
(a) runtime-KV's fp16 cache already satisfies the contract, yet its **opaque append wrote NaN** in-model (a
separate, still-open bug); (b) **B4+fp16cast already wins +11.5%/+16% *with* the materialization copy**, so
runtime-KV's copy-elimination is now only an **incremental** gain on top — its priority drops. Recommend: if
runtime-KV resumes, reuse this fp16 contract and debug the opaque-append NaN; otherwise the bigger near-term win
is a **short-ctx-correct owned tile** to unblock ctx1024 (and a native fp16 cache to drop the cast copy).

## 11. Remaining blockers
1. **ctx restriction**: owned tile only fires ≥2048 (short-ctx over-split). Needs a ctx-scaled-split variant to
   reach ctx1024 (the promotion gate).
2. **Cast copy**: the fp16 cast materializes the cache prefix; a native fp16 owned-tile cache (Candidate A) would
   drop it for a larger win.
3. **Runtime-KV opaque append NaN** (separate lane), only if copy-elimination is pursued.

## 12. Artifacts and commands
- `extra/qk_owned_tile_real_cache_repro.py` → `bench/qk-owned-amdgcn-tile-real-cache/repro.json`;
  `.../correctness.json`; `.../wd.json`.
- Repro: `DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 PYTHONPATH=. .venv/bin/python extra/qk_owned_tile_real_cache_repro.py`.
- Route: `DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 DECODE_ATTN_AMDGCN_TILE=1` (fires ctx≥2048; fp16 cast now built-in).

## 13. Files changed
- `tinygrad/llm/model.py`: owned-tile route now **unconditionally casts Q/K/V to fp16** (mandatory dtype contract;
  default-off route, default gqa path untouched). **Shipped** (the route was broken without it).
- `bench/qk-decode-eval/candidates.json`: `real_cache_status` → `FIXED_REAL_CACHE`.
- New: `extra/qk_owned_tile_real_cache_repro.py`, `docs/owned-amdgcn-tile-real-cache-revalidation-result-20260623.md`,
  `bench/qk-owned-amdgcn-tile-real-cache/{repro,correctness,wd}.json`.

## 14. Working tree status
`model.py` carries the shipped fp16-cast fix (default decode byte-identical, default-off route only). No default
change; no 14B/32B; no paged KV; no new attention tile (a 1-line dtype cast, not a kernel); no
activation/norm/GEMV work; no native codegen.
