# Hand-asm crack — the v_dot4 int-dot GEMV NEARLY SATURATES (49.5% peak, 91% of readraw)

Date: 2026-06-15. `extra/qk_prefetch_gemv.py`. The crack at the dequant inner loop: llama.cpp's actual
approach -- int-dot via v_dot4 (q8 int8 activation, 4 MACs/instruction, integer accumulate) + wide uint4
loads + independent accumulators. RESULT: it closes the bandwidth gap the fp dequant could not.

## Result (degraded-GPU session; relative valid, scale by readraw)
| variant | Q4-GB/s | % peak | % of readraw | vs fp |
|---|---|---|---|---|
| readraw (saturation ceiling) | 466.7 | 54.3% | 100% | -- |
| fp (naive dequant) | 140.3 | 16.3% | 30% | 1.00x |
| fp_prefetch / fp_acc8 (all fp/load/ILP tricks) | ~200 | ~24% | ~44% | 1.45x |
| **vdot (v_dot4 int-dot + wide loads)** | **403.0** | **46.9%** | **86%** | **2.87x** |
| **vdot_acc4 (+ 4 int accumulators)** | **425.1** | **49.5%** | **91%** | **3.03x** |

## The finding
- The fp dequant caps at ~24% because of the per-weight fp-CONVERT + serial fp-ACCUMULATE -- no load/ILP
  trick moves it (we tried: prefetch, wide, 8 accumulators -- all ~24%).
- **The int-dot (v_dot4) avoids the fp convert entirely** (integer multiply-accumulate, 4 MACs/instr) and
  with wide loads + 4 accumulators reaches **49.5% of peak = 91% of the readraw saturation ceiling.**
- Scaled to a healthy GPU (readraw -> 85%), vdot_acc4 would be ~77% of peak -- EXCEEDING llama.cpp's 54%.

So the decode dequant bottleneck (fp dequant capping bandwidth at ~half of pure-read) is SOLVED at the
kernel level by the int-dot approach. This is the first decode kernel that near-saturates memory. The
"42->54% is hand-asm" boundary was right -- and the hand-asm (v_dot4 int-dot, the llama.cpp lever)
actually closes it at the kernel level.

## Honest caveat: e2e realization is the open question
Standalone the kernel near-saturates (425 GB/s). But D1 measured the v_dot4 decode path e2e = NULL
(30.2 = 30.3 tok/s) -- because the e2e was dominated by the per-layer q8 QUANTIZATION overhead (vdot e2e
read half the bytes but at 61 GB/s, slower per-byte than fp's 144), not the GEMV. The combination NOT yet
tried: AMORTIZED q8 quant (quantize x once/token, shared across the 7 linears -- A0's idea) + THIS optimized
v_dot4 builtin GEMV (A0 used the SLOW scalar int-dot v_mad, not the v_dot4 builtin at 425 GB/s). That is the
make-or-break for translating the near-saturating kernel into a real decode tok/s win.

## Net
The hand-asm crack SUCCEEDED at the kernel level: v_dot4 int-dot near-saturates (91% of readraw, ~77% of
peak scaled, exceeding llama.cpp). The decode bandwidth bottleneck is the fp-dequant, and int-dot solves it.
Remaining: wire amortized-quant + this optimized kernel and measure e2e -- the one combination the prior
nulls (D1 = unamortized; A0 = scalar int-dot) never tested together.
