# Owned AMDGCN Tile Post-Promotion — Four-Step Result (2026-06-23)

## Verdict: **POST_PROMOTION_KEEP_DEFAULT_OFF_SYNTHESIS_UPDATED** (+ `POST_PROMOTION_FP16_CACHE_WD_PASS`)

All four steps executed. The owned route is **owner-default-ready** (`OWNER_DEFAULT_READY`) but **kept default-off**
(no flip authorized). **FO2 (native fp16 cache) shipped** — byte-identical, +5–8% over the cast route. Runtime-KV
**deferred (incremental)**. Project synthesis **updated**; stale narratives superseded. Default decode unchanged.

## 1. Step 1 — Owner default decision hardening → `OWNER_DEFAULT_READY`
`extra/qk_owned_tile_default_hardening.py` (true **in-process A/B**: one model, two jits toggled via
`getenv.cache_clear`, interleaved 60 samples/mode):
- ctx1024 **+8.1%**, ctx4096 **+15.0%** (cast version; **1% spread** — tightest measurement).
- **route fired** when on (owned_flash nodes off=0 / on=2), **fell back** correctly at ctx256 (below MIN_CTX).
- default decode byte-identical with flag off (verified separately).
Gates met. **Default kept off** (`default_on=false`) — the scope authorizes a flip only on explicit user request,
which was not given. Recommend a clean `qk_decode_runtime_overhead.py` A/B before flipping.
(`bench/qk-owned-amdgcn-tile-post-promotion/default_decision.json`.)

## 2. Step 2 — FO2 native fp16 owned-tile cache → `FP16_CACHE_WD_PASS` (shipped)
Decision table answered (`.../fp16_cache_probe.json`): the cast copies the materialized fp32 cache half to fp16
per layer/token; a native fp16 cache makes the route's mandatory cast a **no-op**, dropping that copy; the default
gqa path does not need it; it is route-local and the prefill→decode handoff is byte-identical.

**Design B-coupled**: `DECODE_ATTN_AMDGCN_TILE` now implies the fp16 cache (`_init_state`,
`DECODE_ATTN_AMDGCN_FP16CACHE` override). The tile requires fp16 anyway, so this is the correct contract.

- **Correctness**: byte-identical to default gqa at **ctx512/1024/2048**; default decode unchanged
  (`cache_dtype=fp32` with flag off).
- **W==D** (gqa vs owned-fp16, interleaved repeats — clock-controlled):

  | ctx | gqa | owned fp16 | Δ vs gqa | Δ vs cast route |
  |---|---|---|---|---|
  | 512 | 76.5 | **86.6** | +13.1% (12.8/13.3/13.2) | +7.0 |
  | 1024 | 74.3 | **86.2** | +16.0% (15.8/16.2) | +6.7 |
  | 2048 | 71.4 | **84.8** | +18.8% | +7.3 |
  | 4096 | 67.3 | **82.9** | +23.2% (23.0/22.8/23.7) | +7.1 |

  Ships (gate met: byte-identical, route-gated, beats the cast route, no default change).
  Caveat: a separate-process **single** sweep was clock-noisy (ctx512 −3.6% / ctx4096 +5.5% outliers); interleaved
  repeats resolve to the tight values above. (`.../fp16_cache_wd.json`.)

## 3. Step 3 — Runtime-KV status → `RUNTIME_KV_DEFER_INCREMENTAL`
| question | answer |
|---|---|
| needed for promotion? | no |
| still valuable? | marginally (residual materialization, also paid by gqa) |
| blocked by owned-tile correctness? | no (tile fixed) |
| remaining blocker | opaque decode-append writes NaN in-model (separate, open) |
| resume before FO2 / default decision? | no |

**Defer.** FO2 already removed the cast copy byte-identically; runtime-KV would only remove the residual
materialization (incremental) and still has an open opaque-append-NaN bug. If resumed, scope from the post-fix
state — do **not** reuse the refuted GraphRunner/cache-identity theories.
(`.../runtime_kv_status.json`.)

## 4. Step 4 — Project synthesis update → `PROJECT_SYNTHESIS_UPDATED`
- New: `docs/post-owned-attention-promotion-synthesis-20260623.md` (lane table + tok/s picture + corrected
  narrative).
- `structure/Development/session-handoff.md`: superseding banner (owned attention promoted; Route-B "B4 W==D
  fail" superseded; runtime-KV deferred).
- `docs/README.md`: ⭐ bullet for this follow-on. Historical docs not rewritten (superseded by pointers).

## 5. Candidate / default metadata state
`decode_attention_llama_flash_tile_owned_amdgcn_b4`: `default_eligible=true`, `default_on=false`,
`real_cache_status=PROMOTABLE_ALLCTX`, `fo2_fp16_cache=SHIPPED` (+13/+16/+19/+23%), env
`DECODE_ATTN_AMDGCN_TILE=1` (couples fp16 cache) + `DECODE_ATTN_AMDGCN_S=48` + `FLASH_DECODE_THRESHOLD=512`,
contexts [512,1024,2048,4096]. **No default flip.**

## 6. Remaining blockers
None for eligibility. Before an actual `default_on` flip: a clean `qk_decode_runtime_overhead.py` (or CLI
`--benchmark`) A/B to confirm the absolute deltas (separate-process sweeps are clock-noisy). Optional future work:
runtime-KV residual-copy elimination (incremental, open append-NaN); cheaper/fused split-KV combine.

## 7. Artifacts and commands
- `extra/qk_owned_tile_default_hardening.py` → `bench/qk-owned-amdgcn-tile-post-promotion/default_decision.json`.
- `bench/qk-owned-amdgcn-tile-post-promotion/{fp16_cache_probe,fp16_cache_wd,runtime_kv_status}.json`.
- Owned route (now fp16 cache): `DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 DECODE_ATTN_AMDGCN_TILE=1`.

## 8. Files changed
- `tinygrad/llm/model.py`: `_init_state` allocates fp16 cache when the owned route flag (or
  `DECODE_ATTN_AMDGCN_FP16CACHE`) is set; default fp32 unchanged.
- `bench/qk-decode-eval/candidates.json`: b4 env `FLASH_DECODE_THRESHOLD 2048→512`, `fo2_fp16_cache=SHIPPED`.
- New: `extra/qk_owned_tile_default_hardening.py`, this result doc + synthesis doc, 4 post-promotion bench
  artifacts; session-handoff banner; README bullet.

## 9. Working tree status
`model.py` carries the FO2 fp16-cache allocation (default-off route only; default decode byte-identical,
cache_dtype fp32 when flag off). No default flip; no 14B/32B; no runtime-KV implementation; no new attention
tile; no codegen/renderer; no paged KV.
