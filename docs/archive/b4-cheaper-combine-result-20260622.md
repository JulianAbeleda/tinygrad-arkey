# B5-lite v2: Cheaper Split-KV Combine (tiered targets) — Result

Date: 2026-06-22

Executes `docs/b4-cheaper-combine-scope-20260622.md`: push the B4 split-KV combine (`owned_flash_combine`) to the
**preferred/stretch** tier (≤6–7µs / ~5µs) and **re-measure W==D** to resolve whether the combine is on the decode
critical path or overlaps. Combine-only; no new tile, no Route-A codegen, no default change.

## Decision: **`B5_COMBINE_LOCAL_PASS_WD_FAIL`** — a stretch-tier combine (`hw128`, ~2.4× cheaper) PASSES local but **W==D SATURATES at ~+5.7%@ctx4096** (base +5.41 → hd64 +5.71 → hw128 +5.66). **DECISIVE: the combine OVERLAPS in the JIT graph — it is NOT the W==D lever.** Even a free combine projects ~+5.7% (< +7%). The combine-tax Amdahl projection is refuted. **Rest Route B attention.**

## C1 — baseline reproduced
combine 12.6µs @S48 / 16.2µs @S64; correct; split-KV audit `COMBINE_TAX_DOMINATES`. ✅

## C2/C3 — the new lever (`hw`) + local A/B PASS (stretch tier)
The `hd` variant recomputed `exp(m_s−gm)` **per output dim** (CWD× redundant). **`owned_flash_combine_hw`** precomputes
the S weights **once** into LDS (cooperatively) so the main loop is pure FMA over `part`. Local A/B
(`extra/qk_b4_combine_ab.py`, launch-floor-corrected compute; correctness `rel_rmse ≤ 5e-7`):

| combine | compute @S48 | compute @S64 | speedup vs base |
|---|---|---|---|
| base | ~6.2µs | ~9.7µs | 1.0× |
| hd64 | ~4.5µs | ~5.8µs | ~1.7× |
| **hw128 / hw64** | **~3.5µs** | **~4.0µs** | **~2.4× (S64), up to 3.3×** |

`hw` reaches the **stretch tier** (combine compute well under 5µs at the operative S), correct, no tile regression →
local gate PASS. (Caveat: the standalone launch floor is clock-state-sensitive — 6.4µs vs 3.2µs across runs — so the
absolute compute is noisy; the **~2× ratio and the raw ~5µs combine are robust**, and W==D is the arbiter.) `hd32/64/128`
differ only in noise → not workgroup-count-bound; the win is LDS-meta + thread-per-dim + **weight precompute**. `sr`
(split-reduction) refuted earlier (sync/LDS overhead).

## C5 — W==D, the decisive 3-point saturation curve (`bench/qk-decode-attention-route-b-b5-combine/wd.json`)
`extra/qk_b4_decode_eval.py --policy adaptive --splits 48 64 --ckpts 512 1024 2048 4096`, routed best-S Δ vs
`gqa_coop_vec`, tokens byte-identical throughout:

| ctx | base combine | hd64 (~1.7×) | hw128 (~2.4×) |
|---|---|---|---|
| 512 (off) | −0.14% | −0.15% | +0.43%* |
| 1024 | +0.20% | +0.25% | +0.23% |
| 2048 | +1.84% | +2.07% | +1.98% |
| **4096** | **+5.41%** | **+5.71%** | **+5.66%** |

(*ctx512 is route-off control; ±0.4% is clock noise.) **W==D @ctx4096 saturates at ~+5.7%** while combine compute
drops **9.7 → 5.8 → 4.0µs**. A 2.4× cheaper combine adds only **+0.25%** whole-decode; **hw128 vs hd64 (another ~1.8µs
cheaper) adds ~0**.

## Resolution: the combine OVERLAPS (refutes the projection)
The combine-tax doc projected **half-combine → ~+7.0%**, **free → +8.58%**, from an Amdahl model treating the combine's
**standalone GPU time** as fully serial. The 3-point measurement **refutes** it: cutting the combine **2.4×** moves
whole-decode **+0.25%**, and further cuts move it **nothing** → in the JIT graph the combine **overlaps** other work and
is **off the critical path**. **Extrapolating the saturation, even a free combine yields ~+5.7%@ctx4096 — below +7%.**
This is the canonical "isolated kernel wins don't transfer to in-model integration" finding, now nailed with a clean
saturation curve.

## Verdict & recommendation: `B5_COMBINE_LOCAL_PASS_WD_FAIL` → **REST Route B attention**
The combine can be made stretch-tier cheap (`hw`, ~2.4×), but **no combine-only optimization makes B4 promotable** —
W==D is saturated ~+5.7%@4096, structurally below +7%. The B4 owned-AMDGCN attention route is a proven, default-off
capability with a real but sub-bar long-context gain (owner knob). **The remaining whole-decode W==D lever is the
non-attention FFN/GEMV share of the decode step, not the attention primitive.** `hw128` is banked as the best combine
variant (`DECODE_ATTN_AMDGCN_COMBINE=hw128`, default `base`, `default_eligible=false`).

## Deliverables
`docs/b4-cheaper-combine-scope-20260622.md` · `extra/qk_b4_combine_ab.py` (hw variants + bandwidth/wg/floor fields) ·
`owned_flash_combine_hw`/`_hd`/`_sr` in `extra/qk_owned_flash_decode.hip` + variant registry in
`extra/qk_owned_flash_decode_graph_node.py` · `bench/qk-decode-attention-route-b-b5-combine/{latest,wd}.json` · this doc.

## Boundaries honored
Only the combine changed. No new tile, no Route-A codegen, no KV repack/transpose, no default change, no closed-lane
reopen. `gqa_coop_vec` comparator SSOT. W==D is the gate (local GPU-busy launch-corrected diagnostic only). Unrelated
dirty work untouched.
