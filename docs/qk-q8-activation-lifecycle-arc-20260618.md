# Q8 activation lifecycle / reuse arc (ledger + anatomy + break-even) 2026-06-18

Scoping the actual next question after the sudot4 full-linear fail: **can q8 activation become cheap enough or
reusable enough that the faster sudot4 kernel wins whole-linear?** Audit/probe only — no MMVQ kernel, no routing.
RX 7900 XTX / gfx1100. **Answer: no (verdict C).** See `qk-q8-activation-lifecycle-verdict-20260618.md`.

## Phase 0 — q8 pack ledger + activation reuse map

Pack used by the sudot4 probe: `q8_1_quantize` (fp32 → int8 + per-32 scale) then `q8_signed_pack_u32_kernel`
(4 signed int8 → uint32). q8_1-compatible. **29.7µs, 4 kernels.** Output: `qpacked uint32[IN/4]` +
`scales f32[IN/32]`. The pack is **per-activation** (reusable by every linear consuming that activation).

**Activation reuse map (per transformer block, from the model inventory — 36 layers):**

| activation | linears consuming it | weight quant | Q4_K count | useful q8 reuse |
|---|---|---|---|---|
| normed attn input | q (4096→4096), k (4096→1024), v (4096→1024) | q=**Q4_K**, k/v=**Q6_K** | 1 | none (k/v Q6_K) |
| post-attn FFN input | **gate (4096→12288), up (4096→12288)** | both **Q4_K** | **2** | **gate+up only** |
| attn output | o (4096→4096) | Q4_K | 1 | none |
| FFN swiglu output | down (12288→4096) | Q4_K ×18 / Q6_K ×18 | 1 | none |

**Amortization ceiling = 2** (gate+up). k/v being Q6_K kills the attn-input reuse; o and down each consume a
unique activation. There is no Q4_K activation shared by ≥3 linears.

## Phase 1 — pack cost anatomy

| pack stage | kernels | time µs | fusible? | notes |
|---|---|---|---|---|
| abs/max reduce (per-32 block) | 2 (split reduce) | ~15.0 | partially | reduction-bound but tiny (16KB) → launch-floored |
| quantize / round / clip / cast int8 | 1 | 7.9 | yes | memory-bound, launch-floored |
| signed pack 4×int8→uint32 | 1 | 6.9 | yes | memory-bound, launch-floored |

1. **Why 4 kernels:** tinygrad emits max-reduce as 2 split kernels + a quantize elementwise + a separate pack
   custom kernel.
2. Separate ops: max-reduce, quantize+cast, layout/pack.
3. All are **launch/ramp-bound**, not bandwidth (16KB at 900GB/s = 0.02µs of real transfer; each kernel floors
   at **~7µs** fixed cost).
4. Fusible in principle (a single custom kernel doing max→scale→quant→pack), but a multi-store fused custom
   kernel hit the recurring `custom_kernel` plumbing limit (resolve_function dtype / multi-store sink). Even if
   built, it inherits the **~7µs single-kernel floor** (the standalone pack kernel alone is 6.9µs).
5. Shape is fixed/concrete (IN=4096).
6. Warmstart/TC irrelevant (no matmul).
7. Measured standalone (DEBUG2 GPU time); in-graph it can be CSE'd to one instance but still ~7µs.

## Phase 2 — break-even / Amdahl

sudot4 saves **~11µs/linear** vs fp coop (55.0 vs 66.1µs). Pack amortized over reuse count `n`:

| reuse n | pack+n·55 (29.7µs pack) | n·coop (66.1) | net vs coop | ratio |
|---|---|---|---|---|
| 2 (gate+up, real ceiling) | 139.7 | 132.2 | **−7.5µs LOSE** | 0.95× |
| 3 (hypothetical) | 194.7 | 198.3 | +3.6 | 1.02× |
| 4 (hypothetical) | 249.7 | 264.4 | +14.7 | 1.06× |

**Required pack cost to clear the gates over gate+up (n=2):** ≤**9.1µs** for 1.3× base, ≤**5.0µs** for 1.15×
coop, ≤**5.2µs** for 1.05× opaque. With a fused pack at the ~7µs floor: paired gate+up = **1.32× base / 1.13×
coop / 1.03× opaque** → clears base only, **fails coop and opaque**.

**Conclusion:** n is capped at 2 (no ≥3 reuse exists), and a separate pack kernel can't get below ~7µs > the 5µs
break-even. So no separate-kernel q8 pack clears the whole-linear gate vs the byte-identical fp coop. The only
arithmetic that works is a **zero-extra-kernel pack** (q8 produced as an epilogue of the prior op) — see the
graph audit + verdict.
