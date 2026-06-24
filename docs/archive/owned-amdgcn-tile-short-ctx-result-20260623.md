# Owned AMDGCN Tile Short-Context Correctness / Promotion — Result (2026-06-23)

## Verdict: **OWNED_TILE_SHORT_CTX_WD_PASS** → candidate promoted to `default_eligible=true`

The ctx1024 block was **not** a short-context correctness bug — it was an **overly-conservative route guard**
(`DECODE_ATTN_AMDGCN_MIN_CTX=2048`). The standalone tile is **empty-split-safe**, and the prior dtype fp16-cast
fix already resolved the only real short-ctx failure. Lowering the guard to 512 makes the owned tile
**byte-identical to gqa at ctx512/1024** and **W==D-positive at every ctx** (`+6.1/+8.4/+11.5/+15.5%` @
512/1024/2048/4096), clearing the promotion gates. Candidate promoted `default_eligible=true`, `default_on=false`
(no flip). Default gqa decode **unchanged**.

## 1. Verdict
`OWNED_TILE_SHORT_CTX_WD_PASS`: real-cache token-correct + W==D ≥+5%@ctx1024 and ≥+7%@ctx4096 with no ctx512
regression, byte-identical at all ctx.

## 2. Why short-ctx was blocked
Two stacked issues, both now resolved:
- The **dtype-contract bug** (fp32 cache read as fp16) — fixed in the prior task (mandatory fp16 cast). This was
  the *actual* source of the historical "short-ctx 151936" failures (e.g. ctx~600), misattributed to over-split.
- An **overly-conservative `MIN_CTX=2048` guard** added defensively against a *theorized* over-split NaN. With
  the dtype fixed, the guard was the only thing blocking ctx1024.

## 3. Failure matrix (`bench/qk-owned-amdgcn-tile-short-ctx/failure_matrix.json`) — `SHORT_CTX_FAILURE_NOT_REPRODUCED`
Standalone tile vs numpy at real K magnitude (~200), S=48 (default) and a ctx-scaled S:

| ctx | S=48 empty splits | S=48 finite | S=48 rel_rmse | ctx-scaled S | finite | rel_rmse |
|---|---|---|---|---|---|---|
| 512 | 1 | ✓ | 4.2e-8 | 11 | ✓ | 4.2e-8 |
| 1024 | 1 | ✓ | 4.9e-5 | 23 | ✓ | 4.9e-5 |
| 2048 | 0 | ✓ | 4.9e-5 | 47 | ✓ | 4.9e-5 |
| 4096 | 0 | ✓ | 9.6e-6 | 48 | ✓ | 9.6e-6 |

The tile is **finite + correct even with empty splits** — the kernel writes neutral `m=-1e30, l=0, part=0` for
empty splits (unconditionally, after the loop) and the combine skips `l=0` splits. So no empty-split NaN, no
uninitialized meta, no overread. **The over-split theory does not hold; the block was the guard.**

## 4. Split policy / kernel fix — `SHORT_CTX_ROUTE_GUARD_ONLY`
No kernel change and no ctx-scaled-S policy were needed. The single change: **`DECODE_ATTN_AMDGCN_MIN_CTX`
default `2048` → `512`** in `model.py` (env-overridable). The route now fires at ctx≥512 (aligned with the flash
branch). (Candidate B/C empty-split kernel guards were unnecessary; ctx-scaled S is available as a future W==D
micro-optimization but not required for correctness.)

## 5. Correctness result (`.../short-ctx/correctness.json`) — `SHORT_CTX_TOKEN_CORRECTNESS_PASS`
Real chunked-prefill multi-step decode, token authority:
- **ctx512**: 16-token byte-identical to gqa ✓
- **ctx1024**: **64-token byte-identical** to gqa ✓
- ctx2048: 64-token / 2-prompt byte-identical (prior real-cache result) ✓
- default gqa decode **unchanged** (`[279, 1156, 22148, …]`); no `151936` collapse.

## 6. W==D result (`.../short-ctx/wd.json`) — gates cleared
W==D (fixed bound start_pos, `.item()` in window, median-of-40, warm-8, repeated separate-process A/B):

| ctx | gqa tok/s | owned tok/s | Δ | reps |
|---|---|---|---|---|
| 512 | 76.4 | 81.1 | **+6.1%** | |
| 1024 | 74.4 | 80.6 | **+8.4%** | +8.2 / +8.5 / +8.4 |
| 2048 | 71.7 | 79.9 | **+11.5%** | |
| 4096 | 67.3 | 77.7 | **+15.5%** | +15.6 / +15.5 / +15.5 |

Gate: ≥+5%@ctx1024 ✓, ≥+7%@ctx4096 ✓, no ctx512 regression (a gain) ✓, byte-identical ✓. Deltas grow smoothly
with ctx (the owned tile's compute advantage dominates as attention grows) — physically consistent and tight on
repeats.

## 7. Registry / default decision
`candidates.json` `decode_attention_llama_flash_tile_owned_amdgcn_b4`: `real_cache_status` →
**`PROMOTABLE_ALLCTX`**, `contexts` → [512,1024,2048,4096], `historical_expected_verdict` → `PROMOTABLE`, and
**`default_eligible: true`, `default_on: false`**. The route stays gated behind `DECODE_ATTN_AMDGCN_TILE` and
guarded to gfx1100 / Qwen3-8B / B=1 / T=1 with a gqa fallback. **No default flip** — recommend an in-process A/B
confirmation before flipping `default_on`.

## 8. Runtime-KV implication
Runtime-KV stays deferred. The owned tile is now a fully promotable decode-attention win **without** runtime-KV.
Runtime-KV (full copy elimination) would only be **incremental** on top: B4+fp16cast already wins +6–15% *with*
the materialization + cast copies. The remaining copy-tax follow-on is a **native fp16 owned-tile cache** (FO2,
drops the cast) — bigger win, lower priority.

## 9. Remaining blockers
None for promotion eligibility. Optional optimizations: (a) native fp16 cache to drop the cast copy (FO2);
(b) ctx-scaled S for marginal short-ctx occupancy; (c) cheaper/fused combine (split-kv economics). An in-process
A/B is recommended before an actual `default_on` flip.

## 10. Artifacts and commands
- `extra/qk_owned_tile_short_ctx_probe.py` → `bench/qk-owned-amdgcn-tile-short-ctx/failure_matrix.json`;
  `.../correctness.json`; `.../wd.json`.
- Failure matrix: `DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_owned_tile_short_ctx_probe.py`.
- Route (now fires ctx≥512): `DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 DECODE_ATTN_AMDGCN_TILE=1`.

## 11. Files changed
- `tinygrad/llm/model.py`: `DECODE_ATTN_AMDGCN_MIN_CTX` default `2048`→`512` (one line; default-off route only).
- `bench/qk-decode-eval/candidates.json`: B4 candidate → `default_eligible=true`/`default_on=false`,
  `PROMOTABLE_ALLCTX`, contexts [512,1024,2048,4096].
- New: `extra/qk_owned_tile_short_ctx_probe.py`, `docs/owned-amdgcn-tile-short-ctx-{scope,result}-20260623.md`
  (+ `docs/owned-amdgcn-tile-two-followons-scope-20260623.md`),
  `bench/qk-owned-amdgcn-tile-short-ctx/{failure_matrix,correctness,wd}.json`.

## 12. Working tree status
`model.py` carries the one-line guard change (default decode byte-identical, default-off route only). No default
flip; no 14B/32B; no runtime-KV; no paged KV; no new attention tile (a guard threshold, not a kernel); no
codegen/renderer; no activation/norm/GEMV.
