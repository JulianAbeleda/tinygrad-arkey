# HARNESS FIXED — clean per-kernel GPU decomposition; corrects host-dispatch & Tensile-transfer claims

Fixed the decomposition harness: per-kernel GPU timestamps from the official profiler (PROFILE=1 ->
ProfileGraphEvent), validated (discard durations <0 or >2x wall), ONE replay's event, GPU-busy vs wall sanity-
checked. 0 corrupt, GPU-busy=wall confirms validity. This AVOIDS the broken probes (sub-jit overhead inflation,
.sum() codegen-break, multi-replay summing). /tmp/decomp2.py.

## Clean result (one FFN block, 512 tokens)
| | wall | GPU-busy(1 replay) | GPU% of wall | breakdown |
|---|---:|---:|---:|---|
| WMMA FFN | 7.51ms | 7.54ms | **100%** | matmul 99%, glue 1% |
| Tensile FFN | 15.09ms | 4.36ms | **29%** | tensile 96% (gate_up 1440us, down 1380us), glue 4% |

## What this CORRECTS (clean data overturns earlier claims)
1. **WMMA FFN is GPU-BOUND (GPU 100% busy), matmul = 99% of the wall.** -> RETRACT "prefill FFN is host-dispatch-
   bound / matmul is a small fraction of the wall" (the D1 "matmul ~24%" and the host-dispatch framing). The FFN
   matmul GPU time IS the wall.
2. **The Tensile kernel TRANSFERS at the GPU level: Tensile GPU 4.36ms vs WMMA 7.54ms = 1.73x faster GPU** (matches
   isolated 1.56-1.66x). -> RETRACT "the kernel doesn't perform in-graph." It performs; the GPU work IS faster.
3. **The real Tensile problem is DISPATCH: the TensileRunner custom_kernel path leaves the GPU 71% IDLE** (15.09ms
   wall for 4.36ms GPU). That dispatch overhead (10.7ms of gaps) is why Tensile is 2x SLOWER e2e in the isolated
   block despite the 1.73x-faster kernel. NOT shape, NOT transposes, NOT "kernel slow in-graph" -- DISPATCH.

## Remaining real questions (now well-posed)
- **In-jit WMMA matmul runs at ~20 TF** (7.47ms for 154 GFLOP / 3 matmuls = ~2.5ms each) vs the isolated ~42 TF
  (~1.2ms). 2x in-graph slowdown -- likely the .contiguous() output write fused into the kernel, or lower in-graph
  occupancy. THIS is the real WMMA lever (close the 20->42 in-graph gap), and since the FFN is GPU-bound + matmul
  99%, it WOULD translate e2e (unlike the red-herring framing).
- **Tensile dispatch (GPU 71% idle):** the TensileRunner/custom_kernel path is dispatch-broken. If fixed to be
  GPU-bound like the WMMA path, Tensile FFN = 4.36ms vs 7.54ms = 1.73x FASTER. (In the FULL forward the gaps may
  overlap adjacent-layer compute, explaining the ~1.0x there vs 2x-slower isolated -- to verify.)

## Honest note
The harness IS fixable (validated profiler timestamps). My earlier "not cleanly measurable" + "host-dispatch-bound"
+ "matmul small fraction" were all wrong for the FFN -- corrected here by the fixed harness. Lesson: the official
profiler (validated) gives clean per-kernel GPU times; my failures were bad probes, not a stack limit.

## Files
/tmp/decomp2.py (fixed harness). Corrects why-tensile-doesnt-transfer-ANSWERED + ffn-wall-decomposition-audit.
