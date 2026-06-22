# B4 Split-KV Combine-Tax Attribution + Policy — Result

Date: 2026-06-21

Follow-on to `docs/decode-attention-route-b-b4-external-graph-node-result-20260621.md` (`B4_WD_FAIL_INTEGRATION`:
graph-node capability solved, whole-decode W==D +5.6–5.85%@ctx4096 / ~0%@ctx1024, below the +7%/+5% bar). This task
**attributes the split-KV cost** to decide whether a cheaper combine or a ctx-aware split policy can make B4
promotable. No new tile, no Route-A codegen, no default change.

## Decision: **`COMBINE_TAX_DOMINATES`** (the actionable lever) **+ `NO_POLICY_CLEARS_GATE`** (as-is). The split-KV combine is a **fixable latency-bound flat floor**; no ctx-aware split policy makes B4 promotable as-is, but a cheaper/fused combine is projected to clear +7%@ctx4096. Amdahl (attention ≈17% of decode) co-limits the long-context ceiling.

## Attribution (Phase 1 — standalone per-kernel GPU-busy, `extra/qk_b4_combine_tax.py`, no model)
`owned_flash_tile_gqa` vs `owned_flash_combine`, median-of-40 `wait=True`, ctx × S
(`bench/qk-decode-attention-route-b-b4-combine-tax/latest.json`):

| ctx | opt S (min total) | tile µs | combine µs | total µs | **combine %** |
|---|---|---|---|---|---|
| 512 | 48 | 16.0 | 12.7 | 28.7 | **44%** |
| 1024 | 48 | 23.4 | 12.6 | 36.0 | **35%** |
| 2048 | 48 | 36.8 | 12.6 | 49.4 | **26%** |
| 4096 | 64 | 56.5 | 16.2 | 72.7 | **22%** |

Three structural facts:
1. **The combine is a FLAT floor in ctx, scaling only with S:** ~8µs@S8 → 12.6µs@S48 → 16µs@S64, **identical at ctx512
   and ctx4096** (it merges `S` partials per head — independent of KV length). So a fixed ~12–16µs tax is paid every
   token regardless of context.
2. **The combine is LATENCY-bound, not bandwidth-bound — i.e. FIXABLE:** it moves ~0.8 MB in ~12.6µs = **64 GB/s = 6.7%
   of HBM peak** (960 GB/s), running only **Hq=32 workgroups × 32 threads** (~1024 threads under-occupying 96 CUs).
   There is ~3–5× headroom (more workgroups / cooperative reduction / fuse the merge into the tile).
3. **Fewer splits CANNOT dodge the combine** — the tile needs many splits for T=1 occupancy: tile µs @ctx1024 is
   65→23→21 for S=8→48→64. Cutting S to shrink the combine (S8 combine 8µs) costs far more in the tile (S8 tile 65µs).
   The per-ctx optimum is S=48 (≤2048) / 64 (4096); **no ctx-adaptive split policy beats it** (Phase 2 — see below).

## Why this explains the W==D curve
- **ctx512 loses (−2.9%):** the flat combine floor is **44%** of an already-small attention; coop (fewer splits, no
  separate combine floor) is cheaper there → amdgcn can't win → the route is **ctx-gated** (`DECODE_ATTN_AMDGCN_MIN_CTX`).
- **ctx4096 wins +5.6% but caps below +7%:** combine is 22% of attention; the tile's q·k/PV win is real but ~16µs is
  given back to the combine every layer.

## Amdahl projection (combine is the lever for ctx4096)
Per-token @ctx4096 ≈16.4 ms, 36 layers, combine 16.2µs (S=64) → combine costs 0.58 ms/token. Measured W==D +5.6%:

| scenario | ctx4096 W==D (projected) |
|---|---|
| measured (combine 16.2µs) | **+5.6%** |
| combine **halved** (~8µs) | **+7.4%** → clears +7% |
| combine **free** (fused) | **+9.2%** |

So a cheaper/fused combine **plausibly makes B4 promotable at ctx≥2048**. The projection is grounded in the measured
attention split + the Amdahl share; it is an estimate, not a measured W==D.

## Phase 2 — policy sweep (measured: `NO_POLICY_CLEARS_GATE`)
A real interleaved-W==D adaptive sweep across **all four ctx** (now including **ctx2048**) at the attribution-optimal
splits {48,64} (`extra/qk_b4_decode_eval.py --policy adaptive --splits 48 64 --ckpts 512 1024 2048 4096`) gives the
**routed best-S delta per ctx** — monotonic, exactly tracking the shrinking combine fraction:

| ctx | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| routed W==D Δ (best S) | — (off) | **+0.20%** | **+1.84%** | **+5.41%** |
| combine % of attention (Phase 1) | 44% | 35% | 26% | 22% |

`extra/qk_b4_policy_sweep.py` derives the four ctx-aware policies from these (off-ctxs use the shipped `gqa_coop_vec`
default = 0%) and evaluates each against the gate (`bench/qk-decode-attention-route-b-b4-combine-tax/policy_sweep.json`):

| policy | 512 | 1024 | 2048 | 4096 | gate |
|---|---|---|---|---|---|
| `off_below_4096` | 0 | 0 | 0 | +5.41% | **FAIL** (4096 < +7%) |
| `off_below_2048` | 0 | 0 | +1.84% | +5.41% | **FAIL** |
| `adaptive_bestS` (route ≥1024) | 0 | +0.20% | +1.84% | +5.41% | **FAIL** |
| `no_regression` (route where Δ > noise) | 0 | 0 | +1.84% | +5.41% | **FAIL** |

**`NO_POLICY_CLEARS_GATE`.** Smaller S cannot help either (Phase 1: tile starved 38–65µs at S≤24). So **no ctx-aware
split policy makes B4 promotable** — the ceiling is the combine floor + Amdahl, not the split count. The ctx-gated
route (`DECODE_ATTN_AMDGCN_MIN_CTX=2048`, default-off) stays the right *deployment* shape, but it does not clear the
bar as-is.

## Classification & recommendation
**`COMBINE_TAX_DOMINATES`** (short/mid ctx) **+ AMDAHL** (long-ctx ceiling). The combine is the single fixable lever:
- **Recommended next (bounded):** scope a **cheaper combine** for the B4 route — a more-parallel reduction (combine
  over `Hq × Hd`-lane workgroups instead of 32×32, ~3–5× the threads), or **fuse the merge into a second tile pass /
  one persistent kernel** (removes the partial write→read round-trip + the second launch). Target: combine ≤ ~5µs →
  projected ≥+7%@4096. Gate it through `qk_b4_decode_eval.py` W==D (same bar). This is **not** a new attention tile and
  **not** Route-A codegen — it is an optimization of the existing `owned_flash_combine`.
- **If not funded:** since `NO_POLICY_CLEARS_GATE`, bank as **`B4_SPLIT_KV_TAX_REST`** — optionally retained as the
  ctx-gated **owner-call knob** (`DECODE_ATTN_AMDGCN_TILE=1` + `MIN_CTX≥2048`, default-off) for a real but sub-bar
  long-ctx gain (the `FLASH_L=64` precedent).
- **Hard ceiling note:** even a free combine caps attention's ~17% decode share; W==D gains beyond the combine fix
  require attacking the FFN/GEMV share (outside this lane).

## Deliverables
- Phase 1 attribution: `extra/qk_b4_combine_tax.py` → `bench/qk-decode-attention-route-b-b4-combine-tax/latest.json`
  (44 ctx×S rows + bandwidth/occupancy/Amdahl-projection/verdict).
- Phase 2 policy sweep: `extra/qk_b4_policy_sweep.py` (+ the measured `qk_b4_decode_eval.py --policy adaptive --splits
  48 64 --ckpts 512 1024 2048 4096` run) → `bench/qk-decode-attention-route-b-b4-combine-tax/policy_sweep.json`
  (four policies, gate eval, `NO_POLICY_CLEARS_GATE`).
- Scope `docs/b4-split-kv-combine-tax-scope-20260621.md` · this result doc.

## Boundaries honored
No new tile, no Route-A codegen, no KV repack, no default change, no closed-lane reopen. `gqa_coop_vec` comparator SSOT.
Attribution is GPU-busy diagnostic (not a headline). Bounded: Phase 1 launch-only (fast); Phase 2 reused prior W==D +
attribution (no open-ended ctx×S W==D re-sweep).
