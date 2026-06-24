# Owned AMDGCN Tile — Two Follow-Ons Scope (2026-06-23)

## Mission

From `OWNED_TILE_REAL_CACHE_CTX1024_BLOCKED` (owned tile fixed for real cache via fp16 cast; byte-identical to
gqa; W==D **+11.5%@ctx2048 / +16%@ctx4096**; `default_eligible=false` only because ctx-restricted ≥2048), execute
the two named follow-ons to push the owned-tile decode-attention route toward universal (all-ctx) promotion:

- **FO1 — short-ctx-correct owned tile (ctx-scaled splits).** Unblock ctx1024 (and ctx512), the promotion gate.
- **FO2 — native fp16 owned-tile cache.** Drop the fp32→fp16 cast copy for a bigger W==D win.

Target end state: owned tile **real-cache-correct and W==D-positive at ctx 512/1024/2048/4096**, a defensible
candidate for `default_eligible=true` (no default flip in this task).

Boundaries: no default change; no 14B/32B; no paged KV; no RoPE/activation/norm/GEMV work; no native tinygrad
codegen/renderer work; keep the canonical `gqa_coop_vec` default **byte-identical**; token correctness is the
authority (never positions-written; never a degenerate/zero cache); revert unsafe changes on failure.

## Required reading

1. `docs/owned-amdgcn-tile-real-cache-revalidation-result-20260623.md`
2. `docs/runtime-kv-graphrunner-arg-patch-result-20260623.md`
3. `docs/decode-attention-route-b-b4-external-graph-node-result-20260621.md`
4. `docs/b4-cheaper-combine-result-20260622.md`
5. `docs/split-kv-economics-audit-result-20260621.md`
6. `bench/qk-owned-amdgcn-tile-real-cache/{repro,correctness,wd}.json`
7. `structure/Development/performance-primitive-research-principles.md`

Inspect: `tinygrad/llm/model.py` (owned-tile route, `_init_state`), `extra/qk_owned_flash_decode.hip`,
`extra/qk_owned_flash_decode_graph_node.py`, `extra/qk_owned_tile_real_cache_repro.py`,
`bench/qk-decode-eval/candidates.json`.

---

## FO1 — Short-Ctx-Correct Owned Tile (ctx-scaled splits)

### Root cause (hypothesis to confirm)
The owned tile uses a fixed `S` (default 48) KV-splits; grid `(Hkv, S)`; each split covers `per=ceil(n_valid/S)`
positions, `t0=s*per`, `t1=min(t0+per, n_valid)`. At short ctx the **trailing splits are empty** (`t0 ≥ n_valid`)
→ `m=-1e30, l=0` → the combine NaNs (it does not safely skip empty splits). e.g. ctx1024 (n_valid≈1025), S=48 →
per=22, split 47 `t0=1034 > 1025` = empty → NaN. The `DECODE_ATTN_AMDGCN_MIN_CTX=2048` gate exists to dodge this.

### Fix candidates (try in order; prefer no-kernel-change first)
- **A. ctx-scaled S (preferred, no .hip change).** Compute `S` from the bound `start_pos+T` at trace time so
  every split is non-empty and `per` stays near the validated sweet spot (~43 at ctx2048):
  `S = clamp(round((start_pos+T)/PER_TARGET), S_MIN, 48)`, `PER_TARGET≈43`, `S_MIN≈4`. This guarantees the last
  split start `(S-1)·per < n_valid` (no empty splits) and scales parallelism with ctx. Pass `S` to
  `amdgcn_flash_decode` (it already sizes part/meta + grid + combine by `S`). JIT captures one `S` per ctx bucket.
- **B. combine guards empty splits (.hip change).** Make the combine treat `l==0` splits as zero-weight
  (`exp(m-gm)` with `m=-1e30` already → 0; guard the final divide by total `l>0`). More robust (any `S`), but a
  kernel edit. Use only if A is insufficient.

### Phase 1.0 — Reproduce
Confirm S=48 over-splits at ctx1024: standalone `amdgcn_flash_decode` at n_valid≈1025 with S=48 → NaN/garbage;
with S=24 → finite + matches numpy. Record. Artifact `bench/qk-owned-amdgcn-tile-short-ctx/repro.json`.

### Phase 1.1 — Implement ctx-scaled S (Candidate A)
In `model.py` owned-tile route, replace the fixed `getenv("DECODE_ATTN_AMDGCN_S",48)` with a ctx-derived `S`
(env-overridable: `DECODE_ATTN_AMDGCN_S` forces fixed; default = ctx-scaled). Lower `DECODE_ATTN_AMDGCN_MIN_CTX`
to the smallest ctx where the scaled tile is correct (target ≤512, or the proven floor). Keep the route default-off.

### Phase 1.2 — Correctness gate (real cache, token authority)
For ctx 512 and 1024 (real chunked-prefill + ≥32 decode steps): owned tile **byte-identical to gqa**; no `151936`
collapse; no NaN at any sampled layer; default gqa decode unchanged. Eager + TinyJit. Artifact
`.../short-ctx/correctness.json`. Verdicts: `SHORT_CTX_OWNED_TILE_CORRECT` / `SHORT_CTX_OWNED_TILE_NAN_PERSISTS`
/ `SHORT_CTX_REQUIRES_COMBINE_FIX` (→ try Candidate B) / `SHORT_CTX_NOT_EXPRESSIBLE`.

### Phase 1.3 — W==D gate
ctx 512/1024 (now firing) + re-confirm 2048/4096 (S-scaling must not regress). Gate: **≥+5%@ctx1024**, no ctx512
regression, byte-identical, tight spread, route fires. Artifact `.../short-ctx/wd.json`. Verdicts:
`SHORT_CTX_WD_PASS` / `SHORT_CTX_WD_FAIL` / `SHORT_CTX_NO_WIN_AT_1024`.

---

## FO2 — Native fp16 Owned-Tile Cache (drop the cast copy)

### Rationale
The route currently materializes `assigned_kv` (fp32) **and** casts it to fp16 — the cast is the *only* extra
cost over gqa (both pay the materialization). A native fp16 cache for the owned-tile route removes the cast.
(Full copy elimination would need the runtime-KV opaque append, which is separately broken — out of scope here.)

### Fix
`_init_state`: when the owned-tile route is active (`DECODE_ATTN_AMDGCN_TILE` and/or a new
`DECODE_ATTN_AMDGCN_FP16CACHE`), allocate `cache_kv` as **fp16**. The canonical store writes fp16 (k,v are fp16 →
fp16 store, no overflow at K~222). The owned tile reads fp16 directly (the `.cast` becomes a no-op / removed).
Strictly isolated: when the owned-tile flag is OFF, cache stays fp32 → **gqa default byte-identical**.

### Risks / must-verify
- prefill (ctx<2048) uses gqa over the fp16 cache — must stay correct;
- the owned-tile route output must remain byte-identical to the fp32-cache+cast version (both are fp16 precision
  for the tile, so they should match);
- the default (no owned-tile flag) path must be untouched (fp32 cache).

### Phase 2.0 — Implement
Add the fp16-cache allocation under the route flag in `_init_state`; make the route's cast conditional on cache
dtype (no-op when already fp16). Keep default fp32.

### Phase 2.1 — Correctness gate
Owned tile (fp16 cache, no cast) **byte-identical to gqa** at ctx 512/1024/2048/4096 (real cache, ≥32 tokens);
default gqa decode unchanged. Artifact `bench/qk-owned-amdgcn-tile-fp16cache/correctness.json`. Verdicts:
`FP16CACHE_CORRECT` / `FP16CACHE_DECODE_DIVERGES` / `FP16CACHE_BREAKS_DEFAULT`.

### Phase 2.2 — W==D gate + cast comparison
W==D ctx 2048/4096 (and 512/1024 if FO1 landed): fp16-cache route vs (a) gqa baseline, (b) the fp32+cast route.
Expect the fp16 cache to **beat the cast version** (cast removed). Confirm `E_49152` is fp16 or smaller; route
fires; byte-identical. Artifact `.../fp16cache/wd.json`. Verdicts: `FP16CACHE_WD_BEATS_CAST` /
`FP16CACHE_WD_NEUTRAL` / `FP16CACHE_WD_REGRESSES`.

---

## Phase 3 — Combined + Registry / Default Decision

- Compose FO1+FO2: owned tile firing at all ctx with no cast. Final W==D table ctx 512/1024/2048/4096.
- Update `candidates.json` `decode_attention_llama_flash_tile_owned_amdgcn_b4`:
  - `real_cache_status` → reflect ctx coverage + cast status;
  - `default_eligible=true` **only if** byte-identical at all measured ctx **and** W==D ≥+5%@1024 / ≥+7%@4096 /
    no regression **and** fallback-safe; else keep `false` with the precise blocker.
- **No default flip** in this task (record the recommendation + the exact promotion criteria met/unmet).
- If runtime-KV becomes the only remaining lever (full copy elimination), note it as a separate follow-on.

## Phase 4 — Result Doc

`docs/owned-amdgcn-tile-two-followons-result-20260623.md` with: verdict; FO1 (root cause, fix, correctness, W==D);
FO2 (fix, correctness, W==D, cast comparison); combined all-ctx W==D table; registry/default decision +
criteria; remaining blockers; artifacts/commands; files changed; working-tree status. Add a ⭐ README bullet.

### Allowed final verdicts
- `OWNED_TILE_ALLCTX_WD_PASS_PROMOTABLE` (all-ctx correct + W==D gates met; recommend default_eligible=true)
- `OWNED_TILE_CTX1024_UNBLOCKED_WD_PASS` (FO1 lands, FO2 optional)
- `OWNED_TILE_FP16CACHE_WIN_CTX1024_STILL_BLOCKED` (FO2 lands, FO1 fails)
- `OWNED_TILE_SHORT_CTX_NAN_UNFIXED`
- `OWNED_TILE_FOLLOWONS_WD_FAIL`
- `OWNED_TILE_FOLLOWONS_PARTIAL` (mixed)

## Boundaries (repeat)
No default flip; no 14B/32B; no paged KV; no RoPE/activation/norm/GEMV; no codegen/renderer; gqa default must stay
byte-identical; token-correctness authority; revert unsafe changes on failure; do not claim from zero-cache or
positions-written.
