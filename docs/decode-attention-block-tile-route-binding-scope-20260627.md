# Scope — route-bind the generated block tile into the model W==D path (2026-06-27)

For Codex audit. Fixes the foundational gap found by the parity-closure run
(`docs/pure-search-loop-parity-run-route-binding-finding-20260627.md`): the generated block-tile route is **never
exercised by the model W==D path**, so every in-model number we have for it is a phantom.

## 1. The problem (exact)

The decode-attention route selection in `tinygrad/llm/model.py` (~lines 1050–1092) builds `out` from exactly three
branches:
1. **A1 generated skeleton** (`DECODE_ATTN_GENERATED_SKELETON...`) → `flash_decode_attention(...)`
2. **Owned AMDGCN tile** (`DECODE_ATTN_AMDGCN_TILE`, default 1) → `amdgcn_flash_decode(...)`
3. **Fallback** → `flash_decode_attention(..., variant="gqa_coop_vec")`

**No branch calls `flash_decode_attention_whole_cache` (`extra/qk_flash_decode.py:1276`)** — the function that
dispatches the block tile (`tile_builder = flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel if
getenv("DECODE_ATTN_BLOCK_TILE")`, line 1352) and the `FIXED_S` topology (line 1341). `model.py` references neither
`DECODE_ATTN_BLOCK_TILE` nor `flash_block_tiled`. So:
- `DECODE_ATTN_BLOCK_TILE=1` is a **no-op in-model**; the block tile is only reachable from isolated harnesses
  (microgate, isolated timing) that call the kernel directly.
- In the W==D harness it **silently falls back**: `AMDGCN_TILE=1` → owned fires; `AMDGCN_TILE=0` → gqa_coop_vec fires.
- Verified by `DEBUG=2`: `owned_flash_tile_gqa_whole` (×36) fired for the "block-tile" run; never `flash_block_tiled`.

Consequence: the transfer snapshot's `block_tile_route_full_stack` W==D (32.8/6.2 tok/s; "~1.75× in-model") is
**unverified** — `wd_authority=session_reported_not_harness_measured`. The search loop has been targeting a route
that does not run where promotion is decided.

## 2. The fix

### 2a. Add a default-off block-tile branch in `model.py` decode-attention selection
Between the A1 skeleton and the owned-tile branch (or right after owned, guarded so it takes precedence only when
its flag is set), add:
```
if out is None and getenv("DECODE_ATTN_BLOCK_TILE", 0) and DECODE_ATTN_AMDGCN_ARCH_OK and <same shape guards: B==1,
   Hd==128, Hq==32, Hkv==8, (Hq//Hkv)==4, _amdgcn_ctx >= min_ctx>:
    try:
        from extra.qk_flash_decode import flash_decode_attention_whole_cache
        out = flash_decode_attention_whole_cache(q.reshape(Hq,Hd)<.cast(dtype per the cache contract)>, assigned_kv,
                                                 start_pos+T, vsp+T, Hd, Hq, Hkv, MAXC, L)   # whole-cache, buffer-identity
    except Exception as e:
        if getenv("DEBUG",0): print(f"DECODE_ATTN_BLOCK_TILE fallback: {e}"); out = None
```
- **Default-off** (`DECODE_ATTN_BLOCK_TILE` default 0): the owned route stays the shipped default; byte-identical
  when the flag is unset.
- **Output contract:** must return the same shape the owned branch does (reshaped to `(B,Hq,T,Hd)` at line 1093).
- **Confirm the exact `flash_decode_attention_whole_cache` signature** (`extra/qk_flash_decode.py:1276`) and pass
  `assigned_kv` (the whole `cache_kv.after(store)` buffer — preserve buffer-identity, no slice/reshape; the same
  rule that gave owned its win, `[[owned-tile-buffer-identity]]`).

### 2b. dtype/cache contract (the correctness trap to audit)
The owned route forces an **fp16 cache** (`model.py:1129`: `_kv_dtype = fp16 if DECODE_ATTN_AMDGCN_TILE`). The block
tile reads fp32 cache and casts to half internally (microgate uses fp32 cache). Decide explicitly: either
(a) extend the fp16-cache gate to include `DECODE_ATTN_BLOCK_TILE` and have the tile read fp16, or (b) keep fp32
cache for the block-tile route and ensure the kernel's internal cast is correct. Mismatch = NaN K = garbage tokens
(the exact bug the owned dtype-contract comment at `model.py:1065-1067` warns about). **Token-match is the gate.**

### 2c. Add a `route_bound` precheck to the parity matrix
`extra/qk_owned_oracle_parity_audit.py`: add a row (or a global gate) that asserts the generated route **actually
fires in-model** before any `wd_tok_s` / `split_kv_combine` row is trusted — e.g. a captured DEBUG=2 kernel-name
attribution showing `flash_block_tiled*` present (not `owned_flash_tile`/`gqa_coop`). If not route-bound →
`wd_tok_s` status = `UNKNOWN` (blocked: route not bound), not MISMATCH/MATCH. This makes the loop structurally
incapable of repeating the false-positive.

### 2d. Regenerate a harness-measured transfer snapshot
After 2a–2c pass, run the real W==D (`extra/qk_decode_runtime_overhead.py`, `QK_CKPTS=512,4096`) with
`DECODE_ATTN_AMDGCN_TILE=0 DECODE_ATTN_BLOCK_TILE=1 <stack>`, confirm `flash_block_tiled` fires + token-match, and
write a new `transfer_snapshot_*.json` with `authority=harness_measured` (replacing the session-reported one the
generator picks up by glob).

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
