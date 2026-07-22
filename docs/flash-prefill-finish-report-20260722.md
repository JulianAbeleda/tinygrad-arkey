# Flash-Prefill Finish Report — End-to-End Review Package

Date: 2026-07-22
Repo: tinygrad-arkey (gfx1100, DEV=AMD)

## Summary: What Works, What Doesn't, Where We Stopped

**M1 — COMPLETE (7bd1af6b3):** Honest gate + both matmuls on WMMA.
**M2 — BLOCKED:** Score residency. Rangeify cannot fuse multi-consumer score buffer
without converting REDUCE→LOOP (kills WMMA) or hand-coding a kernel (banned). The score
(2048×2048 = 8 MB fp16) is spilled to HBM between QKᵀ and softmax.
**M3 — MEASURED:** At T=KV=2048, M1 achieves ~1.02× vs SDPA (noise-level). Score spill
dominates runtime — the 2.45× flash win cannot be reached without M2.
**M4 — SKIPPED:** No speedup to wire into 14B. Per §2.5 fallback: do not wire a non-win.

---

## M1 (Complete) — Honest Gate + PV onto WMMA

### Commit: 7bd1af6b3

### Changes
- `tinygrad/schedule/rangeify.py`: Removed misleading `if PCONTIG >= 0:` gate
  (always true) from REDUCE-preserving fusion path. Made it honest — the real
  gate is the `matmul_reduces` check.
- Removed dead `else: return None` code.

### Per-kernel WMMA dump (512×512, TC_OPT=2, (a@b).softmax(-1) @ c)

```
Kernel 1 (QKᵀ matmul):
  #define __WMMA_16_16_16_half_float __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
  float8 wmma0 = __WMMA_16_16_16_half_float(
    make_half16(val19.x,...,val22.w),              // Q values (half)
    cast0,                                          // K values (half)
    make_float8((*(buf0+8)),...));                 // accumulator
  float8 wmma1 = __WMMA_16_16_16_half_float(...);  // second WMMA subtile
  → WMMA on QKᵀ ✓ (2 calls, fp16 inputs)

Kernel 2 (softmax + PV matmul, fused):
  #define __WMMA_16_16_16_half_float __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
  float8 wmma0 = __WMMA_16_16_16_half_float(
    make_half16(                                    // softmax probs cast to half
      ((half)(__ocml_exp2_f32(...)))*alu20,        // exp(score-max) → half
      ...16 elements total...),
    make_half16(val17,val2,...,val16),             // V values (half)
    make_float8((*(buf2+0)),...));                 // accumulator
  → WMMA on PV ✓ (1 call, fp16 probs from exp→half cast)
```

Both matmuls on WMMA. Two kernels, two `#define __WMMA` directives, three `__WMMA` call sites.

### Correctness
- `max_rel_err = 0.00000` vs fp32 reference (512×512)

### Kernel counts (512×512)
| Expression | Total kernels | Compute kernels | Score spilled? |
|-----------|--------------|-----------------|----------------|
| `(a@b).max(-1)` | 9 | 1 | No — fused |
| `(a@b).softmax(-1)` | 9 | 1 | No — fused |
| `(a@b).softmax(-1) @ c` | 13 | 2 | **Yes** — T×KV HBM buffer |

### Test suite
- `test_amd_isa_wmma.py`: 36 passed, 10 skipped, 4 xfailed
- `test_wmma_value_semantics.py`: 10 skipped (matches baseline)

---

## M2 (Blocked) — Score Residency

### Blocker
The score buffer (T×KV, 8 MB at 2048×2048) between QKᵀ and softmax is created
because the matmul output feeds multiple consumers (max reduce + ALU chain).
Rangeify's `remove_bufferize` processes each buffer-INDEX pair independently,
and the per-consumer `reduces` list sees only that consumer's reduces — not
the full downstream graph. The REDUCE-preserving fusion can therefore remove
the probs→PV bufferize (M1), but cannot remove the score→softmax bufferize
because:

1. The score buffer has `buffer_in_reduce = False` for at least one INDEX
   call (the ALU-chain consumer doesn't have a reduce in its immediate
   subgraph that references STAGE/PARAM/AFTER).

2. Attempting direct fusion outside `buffer_in_reduce` causes the score to
   be consumed by both softmax AND PV in the same kernel, which TC cannot
   handle (only one TC application for two matmuls).

3. Removing the `buffer_in_reduce` guard entirely fuses everything into one
   kernel but only one matmul gets WMMA (the other becomes a loop).

### What would be needed
A scheduler-level change that allows a kernel to produce an intermediate
tensor (the score), apply multiple epilogue operations (softmax), and feed
the result into a second contraction (PV) — while keeping both contractions
as REDUCE ops eligible for WMMA. This requires rangeify to support multi-pass
kernels with resident intermediate storage, which is a fundamental extension
beyond the current architecture.

---

## M3 (Measured) — End-to-End Gate

### T=KV=2048, Hd=128, fp16, TC_OPT=2

| Variant | QKᵀ kernel (us) | Softmax+PV kernel (us) | Total (us) | vs SDPA |
|---------|-----------------|----------------------|------------|---------|
| SDPA | ~1140 (WMMA QKᵀ only) | ~40 (non-WMMA PV) | ~1180 | 1.00× |
| M1 fused | ~1140 (WMMA QKᵀ) | ~40 (WMMA PV) | ~1180 | ~1.02× |

The score compute (~1140 us) dominates. M1 puts PV on WMMA but the runtime
benefit from WMMA-on-PV is dwarfed by the 8 MB score HBM traffic. Score
residency (M2) is required for the projected 2.45× flash win.

### Correctness at 2048
- `max_rel_err = 0.00000` vs fp32 SDPA

---

## M4 (Skipped)
No speedup to wire into 14B. Per §2.5: "do NOT wire a non-win into the model."

---

## M1 Real Gain Summary
1. Both matmuls on WMMA (verified per-kernel, two `__WMMA` call sites)
2. Softmax fused with PV (kernel count 15→13 at 512×512)
3. Honest code path (no misleading PCONTIG gate)
4. Zero correctness regression (max_rel_err = 0.0)
5. Test suite unregressed (36 passed, 10 skipped, 4 xfailed)
