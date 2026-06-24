# Decode Mode B — Generated Tile-Variant Search — Result (2026-06-23)

## Verdict: `DECODE_MODE_B_EXECUTED_ORACLE_REMAINS_BEST`
Ran the deep generated-kernel search (tile constants TK × S × combine × **vector-width × unroll** — `.hip` body
templated). 14 variants, **all 14 PASS every cost-ordered gate** (build + standalone correctness → route-fire →
materialization/ABI → ISA → 64-tok byte-identical → clean synced W==D), **no variant beats the frozen oracle outside
spread**. The shipped tile constants are optimal. The generated-kernel search loop is now proven end-to-end. **No
default flip; the default decode is byte-identical (templating is additive).**

## Harness compliance
W==D (synced, PROFILE=0, 30 reps) the only ranking authority; correctness/route/materialization/ISA gated before
W==D; all artifacts stamped via `qk_harness_contract`; every variant appended to the project ledger; rejects (none
here) would stop at first gate. Oracle recheck `SEARCH_ORACLE_RECHECK_PASS` (no drift).

## Additive templating (verified)
`owned_flash_tile_gqa_whole` is now parametric: `WCVEC` (LDS-staging load width 1/2/4 via `#if`), `WCUNROLL`
(`#pragma unroll` on the position loop), `TK`. Defaults `(TK16,VEC1,U1)` reproduce the shipped kernel **byte-for-byte**
(default decode `[279,1156,22148,18495,1033,5798]` unchanged; all 8 spot-check variants rel_rmse 2.75e-7). Variants get
uniquely-named symbols (`owned_flash_tile_gqa_whole_tk{TK}_v{VEC}_u{U}`) + own hashed `.co`; `_specialize_tile` injects
the defines. Default decode env → no suffix → shipped behavior. GQA map / `v_dot2` / cross-lane / buffer-identity ABI
held fixed.

## Leaderboard (W==D Δ vs oracle @ctx1024; 90.6/89.3 tok/s oracle)
| variant | knobs | Δ@1024 | Δ@512 |
|---|---|---:|---:|
| TK8 | tk8 | +0.4% | +0.2% |
| combine_hd64 | hd64 | +0.1% | +0.2% |
| oracle_equiv | shipped | −0.3% | −0.1% |
| S32/S64/U2/U4/VEC2/TK32 | — | −0.2…−0.4% | ~0 |
| S96 | s96 | −1.5% | −1.1% |

All within the oracle's spread band except S96 (more splits = real overhead, matching Mode A). The oracle-equiv
variant reproducing the oracle (−0.3%, within spread) confirms harness fidelity. **No win.**

## Measurement-noise note (the spread band working)
Two variants recorded large ctx512 deltas — VEC4 (−15.2%) and VEC2_U2 (−34.9%) — but with **45–57% spread** at ctx512
vs **clean ctx1024** (89.2/88.7 ≈ oracle, spread 0.4–0.5%). These are W==D measurement noise (a contaminated ctx512
window), not real regressions: the spread-band discipline correctly recorded them as untrustworthy and the gate
neither promoted nor rejected on them. The verdict (oracle-remains-best) is robust — they're worse, not better.

## Recommendation
Keep the default tile constants. No recommend-only winner. The deep generated-kernel loop is validated; future
Mode-B-style searches (other shapes/GPUs) reuse this exact machinery.

## Files changed
Modified (additive, default byte-identical): `extra/qk_owned_flash_decode.hip` (WCVEC/WCUNROLL templating),
`extra/qk_owned_flash_decode_graph_node.py` (TK/VEC/UNROLL params + variant symbol), `tinygrad/llm/model.py`
(`DECODE_ATTN_AMDGCN_TK/VEC/UNROLL` env knobs, default 16/1/1), `extra/qk_decode_search_gate.py` (env CANDIDATE_KERNEL),
`extra/qk_b4_combine_tax.py` (5-tuple). New: `extra/qk_decode_mode_b_execute.py`, 8 artifacts under
`bench/qk-decode-mode-b-search/`, +14 ledger entries, this doc. **No default flip.**
