# Fused x-lane score+PV tile — lane-layout microgate scope (2026-06-26)

Continues `docs/decode-fused-score-state-pv-tile-result.md`. The economics pre-gate refuted the scalar
fused tile (`REJECTED_BY_ECONOMICS`); the deficiency is one specific, code-named gap.

## The exact gap (stated by the code itself)

- `flash_fused_score_state_pv_tile_whole_cache_kernel` (line 780): `d` is a LOCAL lane axis, scalar dot →
  the q.k dot is recomputed in every output-column lane.
- `flash_pall_score_state_pv_lifecycle_whole_cache_kernel` (line 226 docstring): "still recomputes q.k
  per output column because **score reuse across the PV column axis is not expressible yet**"; `d` GLOBAL.
- `flash_online_state_pv_tile_xlane_whole_cache_kernel` (line 515, P10-correct): token-sharded, `d`
  GLOBAL, and the score is **precomputed by a separate kernel** (4-kernel route, not fused).

None compute the score **once per token and reuse it across all PV output columns** inside one tile —
which is what the owned route does and what makes decode attention fast. That single lane-layout is the
unexpressed primitive blocking a physically-fast fused tile.

## The layout to prove (canonical flash-decode warp tile)

One 32-lane warp owns one `(kvh, split)` tile. `R = Hd / 32` (= 4 for Hd=128). Per token `j` in the split:

1. **Score (e-sharded, computed once):** each lane owns an e-slice `e = lane*R + r`. Lanes cooperatively
   form the q.k dot (LDS-staged K + `fdot2`), then `_warp_reduce_sum_staged` reduces across lanes →
   `sc` (one scalar per token per query head, broadcast to all lanes). The score is computed **once**.
2. **Online softmax (scalar, identical on all lanes):** `new_m = max(old_m, sc)`, `corr = exp(old_m-new_m)`,
   `p = exp(sc-new_m)`; update `l`.
3. **PV (d-sharded, reuses the one score):** the **same** lane now owns d-slice `d = lane*R + dd`
   (4 output columns). `acc[dd] = acc[dd]*corr + p * V[t, d]`. No cross-lane here — embarrassingly parallel
   across d. The single `p` is reused for all d (no recompute).

The lane meaning transitions e-slice → d-slice within one token loop; the cross-lane reduce (now fixed,
P11/P12 pass) sits between. Output per `(kvh, g, s)`: `acc[Hd]` (PV partial) + `l` + `m` (W=Hd+2), then
the existing `flash_state_gmax` + `flash_state_combine` merge splits — drop-in with the line-780 tile.

## Microgate contract

`extra/qk_decode_attention_fused_xlane_score_pv_microgate.py`, artifact
`bench/qk-decode-attention-fused-xlane-score-pv-microgate/latest.json`.

- Self-contained generated kernel (not yet wired into `qk_flash_decode.py`), validated against a NumPy
  per-split-partial oracle — the same oracle the line-780 standalone numeric uses (per-split `m`, `l`, PV).
- Two modes, to separate *layout* correctness from *fast-primitive* composition:
  - `scalar` — e-shard dot via scalar mul-add + cross-lane reduce, d-shard PV. Proves the **layout**.
  - `fdot2` — same layout with LDS-staged K + `__builtin_amdgcn_fdot2`. Proves the fast primitives compose
    with the layout (this is the version that must later pass the economics pre-gate).
- Shapes: `Tc=128,L=64` (exact, S=2), `Tc=130,L=64` (tail), `Tc=32,L=64` (single split), `Tc=256,L=64`
  (multi split), at `Hq=32,Hkv=8,Hd=128` (real decode shape, G=4).
- Verdicts (specific):
  - `FUSED_XLANE_SCORE_PV_MICROGATE_PASS` — both modes match the oracle (max_abs ≤ 1e-3, rel_rmse ≤ 1e-5).
  - `FUSED_XLANE_SCORE_PV_LAYOUT_FAIL` — scalar mode wrong (the lane layout itself is incorrect).
  - `FUSED_XLANE_SCORE_PV_FDOT2_COMPOSE_FAIL` — scalar passes but fdot2 mode wrong/excepts.
  - `FUSED_XLANE_SCORE_PV_BLOCKED__UOP_VERIFY` — a codegen/UOp-verification exception (isolate the op).

## What pass / fail unlocks

- **PASS (scalar+fdot2):** the score-reuse-across-PV-columns primitive is expressible and correct. Port it
  into `qk_flash_decode.py` as the fused tile body, keep the buffer-identity raw-`cache_kv` 5D access, then
  re-run the attribution economics pre-gate (expect `has_v_dot2`/`has_lds` true) → W==D. This is the path
  to `PROMOTED_GENERATED_FUSED_DECODE` or a clean `REJECTED_BY_ECONOMICS` if the tile economics still lose.
- **LAYOUT_FAIL / BLOCKED__UOP_VERIFY:** the gap is genuinely a codegen/representation limit on the
  e-shard→reduce→d-shard composition → `SEARCH_BLOCKED_BY_CODEGEN` on that specific store/reduce pattern;
  hand off to renderer/lowering work rather than another attention route.
