# q4k_ffn_mmvq_sudot4_with_q8_lifecycle — VERDICT: C (q8 lifecycle fails; int-dot FFN stays CLOSED) 2026-06-18

Audited + **prototyped** (graph-reuse probe + fused-pack prototype, as the research principles demand). The
activation-lifecycle stage cannot clear its gate for gate/up. The only sub-break-even path is a zero-extra-kernel
RMSNorm epilogue (verdict D, deep + lossy, not pursued). Scope: `q4k-ffn-q8-lifecycle-scope-20260618.md`.
Artifacts: `bench/qk-q8-lifecycle/{reuse_map,pack_anatomy}.json`. No kernel changes, no routing, no defaults.

## 1. Activation reuse map (Phase 1)
Only **gate + up** share a Q4_K activation (post-attn FFN input) → **amortization ceiling = 2**. k/v are Q6_K
(kill attn-input reuse); attn_o and ffn_down consume unique activations; lm_head is Q6_K. No Q4_K activation is
shared by ≥3 linears. (`reuse_map.json`)

## 2. q8 pack anatomy (Phase 2)
29.7µs / **4 kernels** (2× split max-reduce ~15µs + quantize 7.9µs + signed-pack 6.9µs), all **launch/ramp-bound**
(16KB; each floors ~7µs). Layout = `uint32[IN/4]` + `f32[IN/32]`, **llama-q8_1-compatible**. Not fused across the
4, but **the pack IS auto-commoned across gate+up** by the scheduler. (`pack_anatomy.json`)

## 3. Break-even (Phase 3) — kernel saves 11.1µs/linear, reuse=2

| reuse n | pack cost | 2 sudot4 + pack | vs 2 coop (132.2µs) | verdict |
|---|---|---|---|---|
| 2 | 29.7 (current) | 139.7 | 0.95× | LOSE |
| 2 | 12.0 (fused, redundant-max) | 122.0 | 1.08× | beats coop, <1.15× |
| 2 | 8.0 (idealized 1-kernel 2-out) | 118.0 | 1.12× | <1.15× |
| 2 | **4.8** (break-even) | 114.8 | **1.15×** | threshold |
| 2 | 0 (RMSNorm epilogue) | 110.0 | 1.20× | PASS |

**Pack cost needed for reuse=2 to clear 1.15× coop: ≤4.8µs** (≤5.2µs for 1.05× opaque). reuse=3/4 don't exist
for Q4_K. So the only viable pack cost is **≤4.8µs**, and a separate kernel can't get below ~7µs.

## 4. Graph-reuse probe (Phase 4) — REFUTED
| paired gate+up | time | kernels | vs coop |
|---|---|---|---|
| 2× fp coop (baseline) | 131.9µs | 4 | 1.00× |
| 2× sudot4, 1 shared pack (lazy) | 139.8µs | 6 | **0.94×** |
| 2× sudot4, manual `.realize()` | 137.7µs | 6 | 0.96× |
| 2× sudot4, duplicated pack | 146.8µs | 7 | 0.90× |

**TinyJit auto-commons the pack** (lazy 6 kernels < dup 7; no manual `.realize()` needed) — graph reuse *works*.
But 1-pack-for-2 still **loses (0.94-0.96× coop)**. Per the Phase-4 gate: *q8 reuse over two linears is REFUTED.*
sudot4 correct, rel err 0.006.

## 5. Fused-pack feasibility (Phase 5) — prototyped, insufficient
Fused single-kernel quant+pack (range over output words, per-block max recomputed) = **12.0µs** (1 kernel,
correctness-verified) — the redundant max dominates; and it doesn't even emit scales (+~7µs). An idealized
1-kernel 2-output pack (max-once) floors at ~8µs but fights the custom_kernel multi-store plumbing. Standalone
pack floor = 6.9µs. **No complete fused pack reaches ≤8µs cleanly, and even 8µs → 1.12× coop < the 1.15× gate.**
The Phase-5 gate ("below ~8µs earns a build") is not met by a complete pack; and the break-even (≤4.8µs) is
unreachable by any *separate* kernel.

## 6. Quality requirement (Phase 6)
The path is **q8-lossy** (rel err 0.006 vs byte-identical fp coop). Before any future route: **dNLL ≤ 0.01**
(teacher-forced, decode-path, OFF=fp-coop ref vs ON, ≥2 windows, à la `qk_nll_eval`). Untested for this path
(q8_1 activation quant is what llama uses, so likely fine — but must verify, never default without it).

## 7. Verdict: C — q8 lifecycle fails; int-dot Q4_K FFN remains CLOSED
- **Reuse ceiling 2** (gate+up) loses even with the pack auto-commoned (0.94×).
- **Fused pack floor ~8-12µs** > the ≤4.8µs break-even; best case 1.12× coop, below the 1.15× gate; and lossy.
- **No build earned.**
- **D noted (not pursued):** q8 as a **zero-extra-kernel RMSNorm epilogue** (norm emits fp + qpacked + scales)
  → 1.20× coop, the only path that clears the economics — but a deep activation-lifecycle change (norm op +
  block wiring), still q8-lossy (needs the dNLL gate), best-case decode EV ~+3-4% (gate+up = 2 of 7
  linears/layer). Scope separately only with a byte-identical or higher-EV motivation.

## Durable (banked)
The full primitive boundary; reuse_map (ceiling 2); pack anatomy (~7µs/kernel floor, auto-commoned); the
break-even (≤4.8µs / zero-kernel to win); the fused-pack prototype (12µs, correctness-verified). Combined with
the banked sudot4 helper fix + 57% kernel, the int-dot Q4_K FFN line is **complete and closed**.

## Files
`[docs]` scope + this verdict; `[test]` `bench/qk-q8-lifecycle/{reuse_map,pack_anatomy}.json`. No kernel/route/
default changes.
