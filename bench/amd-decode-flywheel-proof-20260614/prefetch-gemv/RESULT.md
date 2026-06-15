# Prefetch GEMV — MLP IS the lever (confirmed), but needs a clean implementation

Date: 2026-06-15. `extra/qk_prefetch_gemv.py`. Tests the root-cause lever: does more memory-level
parallelism (loads in flight) raise the Q4_K GEMV bandwidth? (Degraded-GPU session -- relative numbers
valid, absolute % lower than historical.)

## Result
| variant | Q4-GB/s | % peak | loads (disasm) | valu |
|---|---|---|---|---|
| readraw (no dequant) | 467.7 | 54% | 9 | 44 |
| fp (naive interleaved) | 140.8 | 16% | **1** | 78 |
| fp_wide (load block first) | 194.3 | 23% | 9 | 1293 |
| fp_prefetch (load next block) | 204.1 | 24% | 18 | 1298 |

fp_wide 1.38x fp; fp_prefetch 1.45x fp.

## What it confirms
The naive fp has **loads=1** in the ISA -- the compiler serializes the loads with the dequant (one load in
flight), pinning it at 16% of peak. Issuing more loads ahead (loads 1 -> 9 -> 18) raises bandwidth 16% ->
24% (1.45x). **More loads in flight = more bandwidth.** This directly validates the root cause: the GEMV is
memory-LATENCY-bound (low MLP), and MLP/prefetch is the lever -- not compute (readraw with 44 valu hits 54%;
the dequant compute is not what caps the naive fp).

## Honest caveats
1. The prefetch impl is VALU-bloated (78 -> 1298, register-array + full unroll) -> it becomes partly
   compute-bound and stalls at 24%, far below readraw's 54%. A register-efficient deep-prefetch (hardware
   load buffering, balanced register pressure) is needed to approach the ceiling.
2. tinygrad's REAL GEMV already captures some MLP via UNROLL/UPCAST (~42% historically), better than this
   naive prefetch's 24% -- so this beats the NAIVE baseline, not the existing kernel. The remaining headroom
   is tinygrad-42% -> readraw-54%/llama-54%.

## Net
The decode root cause (memory-level-parallelism / prefetch, not compute) is now not just diagnosed but
DEMONSTRATED: a naive GEMV is load-serialized (16%), and adding MLP gives 1.45x. The lever is real and
targets the measured bottleneck. The next step is a clean, register-efficient prefetch pipeline that beats
tinygrad's existing 42% toward the readraw/llama.cpp ceiling -- and whether that is expressible in tinygrad's
opts (UNROLL depth / wider loads / async) or needs a custom pipelined-load primitive.
