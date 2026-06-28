# Scope — route-bind the generated block tile into the model W==D path (2026-06-27)

For Codex audit. Fixes the foundational gap found by the parity-closure run
(`docs/pure-search-loop-parity-run-route-binding-finding-20260627.md`): the generated block-tile route is **never
exercised by the model W==D path**, so every in-model number we have for it is a phantom.

## 1. The problem (exact — corrected per Codex audit)

The in-model generated whole-cache branch **already exists**: `tinygrad/llm/model.py:1042` (gated by
`DECODE_ATTN_GENERATED_WHOLECACHE`) calls `flash_decode_attention_whole_cache` at `:1047` with the known-good
signature. Inside that function the block tile is selected only when **the fused-tile branch is active** —
`qk_flash_decode.py:1333` (`DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE`) → `:1352`
(`tile_builder = flash_block_tiled_... if DECODE_ATTN_BLOCK_TILE`).

So the bug is **a flag contract, NOT a missing branch.** `DECODE_ATTN_BLOCK_TILE=1` alone does not bind the route —
the whole-cache branch (`GENERATED_WHOLECACHE`) and the fused-tile branch (`FUSED_XLANE_SCORE_PV_TILE`) were never
set, so in-model the selection **fell through** to owned (`AMDGCN_TILE=1`) or gqa_coop_vec (`AMDGCN_TILE=0`).
Verified by `DEBUG=2`: `owned_flash_tile_gqa_whole` (×36) fired for the "block-tile" run; never `flash_block_tiled`.

The block tile binds in-model **only** with the full stack:
```
DECODE_ATTN_GENERATED_WHOLECACHE=1  DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1  DECODE_ATTN_BLOCK_TILE=1  DECODE_ATTN_AMDGCN_TILE=0
```
Consequence: the transfer snapshot's `block_tile_route_full_stack` W==D (32.8/6.2; "~1.75× in-model") is **unverified**
(`wd_authority=session_reported`). The search loop targeted a route that does not run where promotion is decided.

## 2. The fix — flag contract, NOT a new branch (corrected per Codex)

**Do NOT add a second branch.** Reuse the existing in-model path at `model.py:1042-1048` (known-good `assigned_kv`,
`start_pos+T`, `vsp+T`, reshape contract). Two acceptable shapes:

- **(a) Flag implication (preferred for the search):** make `DECODE_ATTN_BLOCK_TILE=1` imply the route — i.e. when
  it is set, treat `GENERATED_WHOLECACHE` and `FUSED_XLANE_SCORE_PV_TILE` as on (a single documented place that
  ORs them into the existing branch conditions at `model.py:1042` / `qk_flash_decode.py:1333`). One flag, no
  divergence.
- **(b) Explicit stack (already applied to the search):** require the candidate/W==D stack to carry the full set.
  `bench/qk-search-spaces/decode_attention_loop_search_space.json` `baseline_stack` now includes
  `DECODE_ATTN_GENERATED_WHOLECACHE=1 DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1 DECODE_ATTN_AMDGCN_TILE=0`, and a
  `route_binding_contract` field documents it; the generator emits the full stack. No `model.py` change.

Either way: **default route is unchanged** (with the flags unset, owned ships byte-identical); the only requirement
is that a W==D candidate carries the full stack and proves attribution.

### 2b. dtype/cache contract (the correctness trap to audit)
The owned route forces an **fp16 cache** (`model.py:1129`: `_kv_dtype = fp16 if DECODE_ATTN_AMDGCN_TILE`). The block
tile reads fp32 cache and casts to half internally (microgate uses fp32 cache). Decide explicitly: either
(a) extend the fp16-cache gate to include `DECODE_ATTN_BLOCK_TILE` and have the tile read fp16, or (b) keep fp32
cache for the block-tile route and ensure the kernel's internal cast is correct. Mismatch = NaN K = garbage tokens
(the exact bug the owned dtype-contract comment at `model.py:1065-1067` warns about). **Token-match is the gate.**

### 2c. `route_bound` precheck in the parity matrix — IMPLEMENTED (this commit)
`extra/qk_owned_oracle_parity_audit.py` now has a `wd_token.route_bound` row and gates `wd_tok_s`: both are
`UNKNOWN` unless `bench/qk-owned-oracle-parity/route_attribution.json` proves `route_bound && token_match` AND the
transfer snapshot is harness-measured (not `session_reported`). Verified: with no attribution artifact, `wd_tok_s`
is now `UNKNOWN` (was MISMATCH from the phantom 32.8/6.2). The loop can no longer reason from phantom W==D.
**Remaining for the harness (2d):** produce that `route_attribution.json`.

### 2d. Route-attributed W==D harness — BUILT (this commit): `extra/qk_decode_route_attribution_wd.py`
Built to the "Harnesses Are Performance Primitives Too" bar (a *valid benchmark artifact*, not a bare timer). It
measures the owned comparator and the candidate **in the same session** (fresh model load per route — the KV-cache
dtype gate depends on `AMDGCN_TILE`), and emits all required fields: workload (ckpts/maxc/nmeas/ntok/warmup),
candidate id + primitive class + exact env flags, comparator id + why (owned = shipped default/oracle), git
commit/dirty, hardware, repeats + median + **stdev (noise band)**, a **correctness gate (in-model token-match** vs
the comparator), the **in-model W==D timing authority**, **route attribution** (the decode-attention kernel that
fired → `route_bound`), the `WD_PROMOTION_PCT` threshold, and a verdict + stop reason. Outputs:
`bench/qk-owned-oracle-parity/route_attribution.json` (consumed by 2c) and a harness-measured
`transfer_snapshot_<ts>.json` (`authority=harness_measured`, replacing the session-reported one the generator globs).
Verdicts: `NOT_ROUTE_BOUND…` / `ROUTE_BOUND_BUT_TOKEN_MISMATCH` / `ROUTE_BOUND__TOKEN_MATCH__WD_{AT,BELOW}_THRESHOLD`.
Smoke: `QK_NMEAS=8 QK_NTOK=8 QK_CKPTS=512`.

### 2e. Hybrid fail-loud preflight guard — IMPLEMENTED (this commit)
Keeps (b)'s explicit flags but makes the partial-stack mistake **impossible to pass silently**. `tinygrad/llm/model.py`
(right after `out = None` in the decode-attention selection) now checks: if `DECODE_ATTN_BLOCK_TILE=1` on the
supported shape but the enabling flags (`GENERATED_WHOLECACHE` + `FUSED_XLANE_SCORE_PV_TILE`) are not set, it
**raises** (`DECODE_ATTN_BLOCK_TILE_STRICT=1`, default) or **warns** (`=0`) — catching the phantom *before* a W==D run,
where the `route_bound` precheck only catches it after. Default-inert: `BLOCK_TILE` unset ⇒ guard skipped ⇒
byte-identical owned default. **Verified:** `BLOCK_TILE=1` alone raises with an actionable message; the full stack
passes the guard and **`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` fires in-model** (×36) — the route
binds. (Remaining: harness `route_attribution.json` + token-match + harness-measured snapshot, §2c/§2d.)

## 3. Big picture — how this solves the issue

The pure-search loop's only promotion authority is **in-model W==D + token-match**. Today that authority is
measuring the wrong kernel, so the entire search — parity rows, candidate gates, "transfers in-model" claims — sits
on an unverified premise. Route-binding the block tile makes the W==D harness exercise the **actual** generated
route, which:
- converts the `wd_tok_s` and `split_kv_combine` parity rows from phantom to **ground truth**;
- lets every candidate the loop runs (FIXED_S topology, future work-removal) be **honestly** W==D-gated;
- closes the false-positive class entirely via the `route_bound` precheck (the loop refuses to trust W==D until the
  route fires);
- and re-anchors the diagnostic truths (occupancy, slope, combine) to a route that is real in the path that ships.

In one line: **it gives the search a true north.** Everything downstream is only meaningful once the thing we
measure is the thing that would ship.

## 4. Expected outcome

- **Most likely:** the honest in-model block-tile W==D is **far below** the session-reported 32.8/6.2 — i.e. the
  real gap to owned is even larger than believed (the generated route pays full in-model overhead: route dispatch,
  the separate split-KV combine kernels, dtype casts). The phantom flattered it. The `wd_tok_s` parity row gets a
  trustworthy (worse) number, and `split_kv_combine` becomes evaluable.
- **Possible:** the block-tile branch **errors or mismatches tokens** (dtype/cache or shape contract) — surfacing a
  real `PRIMITIVE_PLACEMENT_BUG`/correctness bug that the isolated harnesses never caught. That is a *good* outcome:
  it's the route-binding correctness work that must precede any speed search.
- **Either way:** the search stops optimizing a phantom. No performance is expected to improve from this change
  itself — it is **instrumentation/ground-truth work**, the prerequisite the audit flagged on day one.
- **Non-goal:** do not promote the block tile by default. `DECODE_ATTN_AMDGCN_TILE` (owned) stays the shipped route.

## 5. Acceptance gates (for the audit)

1. Flag unset ⇒ byte-identical (owned default untouched); shape guards match the owned branch exactly.
2. `DEBUG=2` shows `flash_block_tiled*` fires in `model.generate` decode with the block-tile flag.
3. **Token-match passes** vs the owned/gqa route (greedy byte-identical or explicit quality gate) — the dtype/cache
   contract is correct.
4. W==D is **harness_measured** (new snapshot), not session-reported.
5. Parity `route_bound` precheck = MATCH; `wd_tok_s` is no longer UNKNOWN-for-route-reasons.
6. No regression to the owned default route (W==D unchanged with the flag off).

Files: `tinygrad/llm/model.py` (decode-attention selection ~1050-1092, cache dtype ~1129),
`extra/qk_flash_decode.py:1276` (`flash_decode_attention_whole_cache`), `extra/qk_owned_oracle_parity_audit.py`
(route_bound), `extra/qk_decode_runtime_overhead.py` + `extra/qk_decode_token_match_check.py` (measurement).
