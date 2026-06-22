# Post Owned-Attention-Promotion — Project Synthesis (2026-06-23)

## Verdict: `PROJECT_SYNTHESIS_UPDATED`

This supersedes the stale "decode attention exhausted / B4-B5 sub-bar / runtime-KV is the next promotion-critical
lane / owned tile not real-cache-correct / ctx1024 blocked" narratives. The owned AMDGCN decode-attention route is
now a **promotable, all-ctx, byte-identical decode-attention win** — the first promotable decode-*attention*
primitive (prior promotable wins were GEMV).

## Current state — decode primitive/lane synthesis

| primitive / lane | latest status | default eligibility | next action |
|---|---|---|---|
| `Q4K_GEMV_WARP` (FFN GEMV) | W==D pass, byte-identical (+8.5–9.8%) | `default_eligible=true`, `default_on=false` | owner default decision (parallel) |
| **owned AMDGCN decode attention** (b4) | **W==D pass ALL ctx + FO2 fp16 cache: +13.1/+16.0/+18.8/+23.2% @512/1024/2048/4096, byte-identical** | **`default_eligible=true`, `default_on=false`** | owner default decision (in-process A/B first) |
| FO2 native fp16 owned-tile cache | **shipped (coupled to the route flag)**; drops the cast copy, +5–8% over cast | n/a (part of the route) | done |
| runtime-KV (copy elimination) | **deferred — incremental** (FO2 already removed the cast copy; open opaque-append-NaN) | n/a | resume only if a future audit shows the materialization is the dominant tax |
| Route B older B4/B5 "combine tax / W==D fail / sub-bar" narrative | **superseded** | n/a | historical only (do not propagate) |

## What changed (2026-06-23 arc)
1. **Owned tile dtype-contract bug fixed** — it read the fp32 `cache_kv` as `__half` → NaN K → garbage from decode
   step 1; masked because prior W==D used a degenerate/zero cache. Fix: mandatory fp16 cast (→ now fp16 cache).
2. **ctx1024 unblocked** — the `MIN_CTX=2048` guard was over-conservative; the tile is empty-split-safe. Fix:
   default `2048→512`. Byte-identical all-ctx, W==D clears the gates.
3. **FO2 native fp16 cache** — the route flag now implies an fp16 cache (cast becomes a no-op), dropping the
   fp32→fp16 copy for +5–8% over the cast route, byte-identical.
4. **Runtime-KV saga resolved/closed as the root-cause for the owned route** — the long "RUNTIME_KV baking" was the
   owned-tile dtype bug, not GraphRunner arg patching (proven correct) or cache identity (refuted). Runtime-KV is
   now an optional incremental lever, not a blocker.

## Decode tok/s picture (W==D, gfx1100, Qwen3-8B-Q4_K_M)
| ctx | default gqa | owned route (fp16 cache) |
|---|---|---|
| 512 | ~76.5 | **~86.6 (+13.1%)** |
| 1024 | ~74.3 | **~86.2 (+16.0%)** |
| 2048 | ~71.4 | **~84.8 (+18.8%)** |
| 4096 | ~67.3 | **~82.9 (+23.2%)** |

vs llama.cpp ~97–100 tok/s @ctx512–1024: the owned route lifts tinygrad from ~76% to **~87–89% of llama @ctx1024**,
and the gap closes further at long context.

## Stale references corrected
- `structure/Development/session-handoff.md`: superseding banner added (owned attention promoted; runtime-KV
  deferred).
- `docs/README.md`: ⭐ bullets for the real-cache fix, short-ctx promotion, and this four-step follow-on.
- Historical docs (B3/B4/B5 combine-tax, runtime-managed-KV, graphrunner-arg-patch) are **not rewritten** — they
  carry their own corrected verdicts and point forward.

## Notes / caveats
- All deltas are W==D (fixed-ctx steady-state decode), correctness validated separately on a real cache
  (byte-identical). Separate-process single sweeps are clock-noisy; the numbers above use in-process A/B (cast) and
  interleaved repeats (fp16 cache). A clean `qk_decode_runtime_overhead.py` confirmation is recommended before an
  actual `default_on` flip.
- The owned route stays gated behind `DECODE_ATTN_AMDGCN_TILE`, guarded to gfx1100 / Qwen3-8B / B=1 / T=1 with a
  gqa fallback; **no default flip** was made.
