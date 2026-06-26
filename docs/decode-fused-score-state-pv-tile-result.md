# Fused score+state+PV decode tile — result (2026-06-26, continuation)

Candidate: `decode_attention_fused_score_state_pv_tile`
Continues `docs/decode-score-broadcast-lifecycle-resolution-plan.md` (the cross-lane reducer dependency).
Loop: follow the wall → isolate → identify → solve → rerun. No new attention detour was added.

## What this turn established (all verified by live gate runs)

1. **The "5D indexing wall" was a false alarm — a stale string-marker short-circuit, not a lowering bug.**
   The fused gate pre-check counted literal `"0 * Hkv"` / `"1 * Hkv"`, which the new 5D builder
   (`cache[0, 0, kvh, t, e]` / `cache[1, 0, kvh, …]`) no longer contains, so it returned
   `BLOCKED__INCOMPLETE_QKV_LIFECYCLE` *before the numeric ever ran*. Fix in
   `extra/qk_decode_attention_fused_score_state_pv_tile_gate.py`: update the two cache-load markers to
   the 5D access pattern, and pass a 5D `(2, 1, Hkv, MAXC, Hd)` cache to the standalone kernel
   (`cache5 = cache.reshape(2, 1, Hkv, MAXC, Hd)`).

2. **After the fix the fused tile is numerically correct AND route-clean.**
   - Standalone numeric: PASS (max_abs 5.3e-05, rel_rmse 1.95e-07). The 5D `cache[i,0,kvh,t,e]` indexing
     lowers and computes correctly — there is no 5D lowering wall.
   - Route gate: `FUSED_SCORE_STATE_PV_TILE_ROUTE_CLEAN__WD_REQUIRED`. Exactly 3 generated programs —
     `flash_fused_score_state_pv_tile_whole_cache_32_128` + `flash_state_gmax_32_128` +
     `flash_state_combine_32_128` — owned absent, no materialization, token-match. Bounded kernel count.

3. **Economics pre-gate REJECTS the fused tile — it is physically bad.**
   `extra/qk_decode_attention_fused_score_state_pv_attribution.py` →
   `FUSED_SCORE_STATE_PV_ATTRIBUTED__GENERATED_TILE_PHYSICALLY_BAD`. Blockers: no v_dot2/native packed
   dot; no LDS/tile-staged K/V reuse; no cross-lane q.k sharing; q.k dot repeated across local output
   columns. → terminal **`REJECTED_BY_ECONOMICS`** for the scalar fused tile.

4. **The missing physical primitives are NOT missing from codegen — they already exist in generated
   kernels.** In `extra/qk_flash_decode.py`: LDS staging (`klds = UOp.placeholder(..., AddrSpace.LOCAL)`,
   lines 205/244), fdot2/v_dot2 (`__builtin_amdgcn_fdot2(...)`, lines 215/257), cross-lane
   (`_warp_reduce_sum_staged` / `warp_reduce_max`, lines 184/217/274/564-567/609-611).
   `flash_pall_score_state_pv_lifecycle_whole_cache_kernel` already composes all three. So the label is
   **`SEARCH_BLOCKED_BY_FUSED_COMPOSITION`**, not missing codegen and not a true wall.

5. **The cross-lane reducer fix retroactively repaired the physically-fast tiles' correctness.**
   - P10 (`…p10_xlane_output.py`): now `ONLINE_STATE_PV_P10_XLANE_OUTPUT_PASS` (~1e-8 vs NumPy and vs the
     scalar oracle). Was a ~1.0 error before the reducer fix.
   - P7 in-model (`…online_state_pv_xlane_gate.py`): `token_match: true` now (was `TOKEN_MISMATCH`). The
     xlane route is numerically correct in the full model.

6. **The remaining wall on the physically-fast tiles is materialization, not math.**
   P7 fails only on `route_clean: false` → `full_maxc_copy_kernels: ["E_98304_32_3"]`,
   `selected_route_buffer_identity: false`. The xlane route passes `cache_f = cache_kv.reshape(2·Hkv·MAXC·Hd)`
   (`qk_flash_decode.py:1106`), which materializes a full K+V copy because the model `cache_kv` is a
   non-contiguous assigned view. The **fused** route alone (line 1147) passes raw `cache_kv` (5D) and is
   therefore buffer-identity clean; every other route (score-broadcast 1125-1133, pall-lifecycle 1140,
   the separate score kernel 1159, all tile branches) still passes `cache_f` → `E_98304`.

   (The audit's H3 gate-hardening has landed: `qk_decode_search_gate.py:41` now flags any
   `_copy_elems(n) >= 49152` and ties `buffer_identity_inputs` to "no full-MAXC copy", so `E_98304` is no
   longer invisible.)

## Net state

| Property | Scalar fused tile (line 780) | Physically-fast tiles (xlane / pall-lifecycle) |
|---|---|---|
| Numerically correct | yes (5e-5; tokens match) | yes now (P10 1e-8; P7 token-match) |
| Route-clean / buffer-identity | yes (raw `cache_kv`) | no — `E_98304` from `cache_f` reshape |
| Bounded kernel count | yes (3) | xlane is 4 (separate score) |
| Physically fast (fdot2/LDS/cross-lane) | no (scalar) | yes |

No single route is yet correct + clean + bounded + fast. The two families have **complementary**
deficiencies, and both deficiencies are now engineering, not walls.

## The solve (next step — unblocked, in progress)

Converge the two: **make the buffer-identity-clean fused tile (`flash_fused_score_state_pv_tile_whole_cache_kernel`,
line 780) physically fast** by replacing its scalar `qv*kvv` reduction with LDS-staged K + fdot2 packed
dot + cross-lane q.k reduction (port the now-correct machinery from the xlane / lds-crosslane kernels),
while keeping its raw-`cache_kv` 5D access and single-tile structure. Validate against the existing
standalone numeric gate (the scalar tile is the oracle; revert the kernel edit if numeric regresses),
then re-run the attribution economics pre-gate (expect v_dot2/LDS now present), then W==D.

Alternative (larger blast radius): convert the xlane route's cache access to raw `cache_kv` + 5D
indexing to kill `E_98304` — but its score and tile kernels are shared flat-cache kernels used by other
routes, so this touches multiple callers. The fused-tile port is the smaller, isolated, validatable
change.

**Difficulty note (the real crux).** This is not a drop-in substitution. In the fused tile, the LOCAL
lane axis is `d` (the output column, W=Hd+2 lanes), and the q.k dot is a reduction over `e` (head dim) —
so the dot is recomputed independently in every d-lane (blocker 4). To compute it once and share it,
`e` would have to own the lane (for a cross-lane reduction over e), but `d` already does. Resolving that
is a tile-layout / lane-ownership re-architecture (cooperative d-ownership + phased e-reduction with LDS
staging), which is exactly the owned route's "fused tile economics" and the represented-search-space gap
that FutureSight/BubbleBeam is meant to close. Treat it as a codegen/tiling design step, not a quick
port: prototype the lane layout in an isolated microgate (synthetic q, K-tile) against the scalar
oracle before editing the in-model kernel.

## Labels

- Scalar fused tile: **`REJECTED_BY_ECONOMICS`** (correct + clean + bounded, physically bad).
- Lane: **NOT** `SEARCH_BLOCKED` and **NOT** missing codegen. The primitives exist, the reducer is fixed,
  and the physically-fast tiles are now numerically correct. The remaining gap is
  `SEARCH_BLOCKED_BY_FUSED_COMPOSITION` reframed as a concrete, validatable kernel-composition task
  (fast primitives into the clean fused tile) — in progress, not walled.

Do not revive score-broadcast (economically refuted). Do not add another attention route before the
fused-tile composition is attempted.
