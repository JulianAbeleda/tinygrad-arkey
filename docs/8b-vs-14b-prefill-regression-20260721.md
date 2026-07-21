# Why the prefill regression was fixed at 8B but persisted at 14B

**Date:** 2026-07-21 · **Hardware:** AMD RX 7900 XTX (gfx1100), 24 GB VRAM, ~122.8 TFLOP/s fp16 WMMA peak

**One-line:** The 8B fix was *"materialize fp16 and feed the scheduler a clean GEMM."* That's not a recipe, it's an **affordance** — 8B's fp16 weights fit in 24 GB. 14B's don't, so the same fix physically cannot run, and 14B stayed on the slow path until a *different* fix (fused-dequant packed-WMMA) reached the same tensor-core regime **without** materializing fp16.

---

## 1. What the regression was

Prefill throughput on gfx1100 sat far below llama.cpp. The default route for quantized weights, `DIRECT_PACKED_FALLBACK`, dequantizes Q4_K on the fly and runs the matmul on the generic vector ALUs — **it never lights up the tensor cores (WMMA)**. Measured floor:

| Model | Route | pp512 tok/s | vs llama (~1837) |
|---|---|---|---|
| 14B | `DIRECT_PACKED_FALLBACK` | ~354 | 0.19× |

The whole effort was: get the tensor cores doing the prefill GEMM.

## 2. The roofline: why tensor cores are the entire game

Prefill is a GEMM: `M` tokens × weight `W[N,K]`. Compute cost `= 2·M·N·K` FLOPs.

The gfx1100 has two compute regimes:

- **Vector FMA (what direct-packed uses):** ~a few TFLOP/s effective for the dequant+multiply mix.
- **WMMA / tensor cores (what we want):** ~122.8 TFLOP/s fp16 peak — **~10–20× higher**.

So the ceiling is set by *which unit does the multiply*, not by clever tiling alone. Any route that leaves the multiply on the vector ALUs is capped ~10–20× below a WMMA route on the same silicon. That is the ~0.19× gap above. **Getting to WMMA is the fix; everything else is second-order.**

## 3. The 8B fix: `FULL_RESIDENT_OVERLAY` (materialize fp16)

At 8B, fp16 weights fit in VRAM, so the fix was direct:

1. Materialize a resident **contiguous fp16 copy** of each weight (`_pf16_w`).
2. The scheduler now sees a clean `fp16 × fp16` GEMM — no quant view-chain in the way.
3. A clean GEMM accepts the rich warmstart recipe `TC + UPCAST(0) + UPCAST(1) + UNROLL(0,8)`, which packs the WMMA tiles densely.

Result: 8B pp512 → **~3448 tok/s, faster than llama.** Regression closed.

### The memory math that makes this legal at 8B

```
Qwen3-8B params            ≈ 8.19 × 10^9
fp16 resident copy         = 8.19e9 × 2 B      ≈ 16.4 GB
+ KV cache + activations   ≈ a few GB
                           ────────────────
total                      < 24 GB   ✅ FITS
```

The overlay is affordable because 16.4 GB leaves headroom under the 24 GB card.

## 4. Why 14B did NOT get fixed by the same move

The 8B fix is a *"fits fp16"* solution. Run the same arithmetic at 14B:

```
Qwen3-14B params           ≈ 14.77 × 10^9
fp16 resident copy         = 14.77e9 × 2 B     ≈ 29.5 GB   ← already > 24 GB
+ Q4_K packed weights kept ≈ 8.4–9 GB
+ KV cache + activations   ≈ a few GB
                           ────────────────
total                      ≈ 38–41 GB  ❌ OOM  (fp16 alone already OOMs)
```

The fp16 copy *by itself* (29.5 GB) exceeds the whole 24 GB card. So `FULL_RESIDENT_OVERLAY` can **never activate** at 14B — the policy declines it and the route falls back to `DIRECT_PACKED_FALLBACK`, i.e. **the slow ~354 tok/s regime the fix was supposed to escape.** The regression persisted not because the fix was wrong, but because 14B can't pay its entry cost.

### The trap this created

The "ceiling" everyone chased for 14B — a projected ~1940 tok/s — was **extrapolated from the 8B overlay speed**, i.e. from a path 14B structurally cannot run. And the 8B *recipe* (rich UPCAST/UNROLL warmstart) got mistaken for the *cause* of the speed. Both are wrong:

- **Measured:** applying 8B's `UPCAST/UNROLL` warmstart to a 14B contiguous fp16 weight → **6.6 TFLOP/s vs packed-WMMA's 9.5 → 31% *slower*.**
- The speed was never in the Opt list; it's in the **tile geometry** (`PACKED_WMMA_GEOM` tm/tn/tk/waves/LDS).
- "14B is slow *because* it can't take UPCAST/UNROLL" is a **spurious correlation** — 8B was simply a smaller model that happened to fit fp16.

## 5. How 14B was actually fixed: `BOUNDED_PACKED_TILES` (packed-WMMA)

Instead of porting the 8B recipe, build a third strategy that reaches the WMMA regime **without materializing fp16**:

- Dequantize Q4_K → fp16 **in-register, fused into the WMMA**, driven by a view-chain off the packed bytes (`bitcast/reshape/pad/expand/reshape/bitcast`).
- No resident fp16 anywhere → no OOM → the entry cost 14B couldn't pay simply isn't charged.
- Reaches the tensor-core roofline the same as overlay does.

```
14B packed-WMMA VRAM  ≈  9 GB packed weights + KV + activations  <  24 GB  ✅ FITS
```

Measured:

| | 8B fp16 overlay | 14B packed-WMMA | llama.cpp 14B |
|---|---|---|---|
| pp512  | 3448 | **1829** | ~1837 |
| pp1024 | 3209 | 1726 | — |
| pp2048 | 2792 | 1510 | — |
| pp4096 | 2234 | 1255 | — |
| Weight format | contiguous fp16 (`_pf16_w`) | view-chain off Q4_K bytes | — |
| Warmstart | TC + UPCAST + UNROLL | **TC only** (geometry does the work) | — |

14B packed-WMMA: ~**5.2× over direct-packed**, at parity with llama at pp512.

## 6. The three strategies, and which fits which model

`prefill_policy.py: _EXECUTING_STRATEGIES`:

| Strategy | Uses WMMA? | Needs fp16 resident? | 8B | 14B |
|---|---|---|---|---|
| `DIRECT_PACKED_FALLBACK` | ❌ vector ALU | no | (slow floor) | (slow floor) |
| `FULL_RESIDENT_OVERLAY` | ✅ | **yes (fp16 must fit)** | ✅ activates | ❌ OOM, declines |
| `BOUNDED_PACKED_TILES` (packed-WMMA) | ✅ | **no (fused dequant)** | n/a | ✅ activates |

The policy is fail-closed: if a strategy's precondition (VRAM, role, shape) isn't met, it declines and falls through. 14B declining overlay → falling to direct-packed was, mechanically, the persistent regression.

## 7. What's left (a *different* regression, not this one)

After packed-WMMA, 14B is at parity short-context but drops to ~76% of llama at pp4096. That residual is **not** the GEMM/quant regression above — it's **attention fusion**: both prefill attention branches materialize the full `T×KV` score matrix, while llama runs a fused flash-attention kernel (`flash_attn_ext_f16`). Closing it needs a fused-flash-**with-WMMA** prefill kernel. The cheaper reuse-the-decode-kernel path was spiked and is a proven **NO-GO** (scalar-`fdot2` compute wall) — see `docs/flash-prefill-scope-20260721.md`.

---

### TL;DR math

```
8B :  fp16 = 8.19e9 × 2  = 16.4 GB  <  24 GB  → overlay legal → WMMA → fixed
14B:  fp16 = 14.77e9 × 2 = 29.5 GB  >  24 GB  → overlay illegal → direct-packed → still broken
14B fix: fused Q4→fp16-in-WMMA, 9 GB resident → WMMA without fp16 → fixed (1829 tok/s)
```

The 8B fix and the 14B regression are the same coin: the 8B solution's precondition (fp16 fits) is exactly what 14B violates.
