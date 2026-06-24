# Primitive 2 — Stream-K / adaptive-split decode attention: REFUTED by audit (not built) 2026-06-17

Roadmap Primitive 2. Goal was to flatten the remaining ctx slope by splitting KV across more blocks (occupancy)
+ fixup/combine. **Verdict: gate not reachable after Primitive 1 (gqa_coop_vec) closed the slope. Not built**
(per "build only if the gate earns it"). RX 7900 XTX, Qwen3-8B-Q4_K_M.

## Why the gate is unreachable [measured]

Primitive 1 (`gqa_coop_vec`, coalesced LOCAL-d loads) already:
- **Flattened the slope to −8%** (47.7→43.9 tok/s, ctx512→4096) ≈ llama's −7%. Stream-K's whole purpose
  (slope-flatten) is essentially done.
- Shrank attention to a **small share**: eager decode breakdown @ctx4096 (gqa_coop_vec) = **GEMV 58% / other
  23.7% / attention 18.3%** (attention was 47% under the old hoisted/v2). Stream-K only speeds attention.
- **Already fills the GPU at the long-ctx gate:** `flash_partial_coop_vec` grid @ctx4096 = Hkv×S = 8×32 =
  **256 workgroups × 129 threads** → saturates 96 CUs. Stream-K's lever is KV-split *occupancy*; there is no
  occupancy deficit left to exploit where the gate lives (long ctx).

Stream-K gate = ≥5%@2048 and ≥8%@4096. Ceiling: even making attention **free** @4096 = 1/(1−0.183) = +22%,
but Stream-K (a) only helps occupancy, which is already saturated at long ctx, and (b) adds fixup/combine
kernels. Realistic gain on the 18.3% attention is **~+1–3% decode**, **below the ≥8% gate**. At short ctx
(ctx512, grid 32 wg, under-filled) Stream-K *could* add occupancy — but ctx512 isn't the gate and attention is
an even smaller share there.

## Verdict

**REFUTED (gate not earnable).** The decode-attention slope gap is closed by gqa_coop_vec; the residual decode
gap is the **base gap (GEMV 58% @ctx4096)**, which the GEMV-structural audit
(`qk-base-decode-gemv-structural-plan-20260617.md`) already found has no bounded target (dp4a +1% e2e). Building
Stream-K would add complexity (fixup/combine) for ~+1–3%.

## Decode-attention follow-ons now SETTLED
- P1 vectorized/coalesced loads (gqa_coop_vec): **SHIPPED** (+6.5…+48.8%, slope closed).
- P2 Stream-K: **REFUTED** (this doc — slope already flat, GPU filled at long ctx, attention now 18.3%).
- P3 MMVQ/GEMV: **REFUTED** earlier (no bounded target, dp4a +1%).

## Next (per roadmap, decode bounded levers exhausted)
Decode is now ~**48% of llama FLAT** with the slope gap closed; the remaining gap is base/structural. Honest
next directions: **Primitive 7 (prefill WMMA** — different phase, the revived WMMA's right home, prefill already
81% of llama), the very-high-risk **decode-block / low-sync-spec** primitives, or the **14B/32B matrix**. No
further bounded decode-attention target remains.
