# E0 -- amortized quant + v_dot4 GEMV: NULL e2e. The near-saturating kernel does not translate. Loop closed.

Date: 2026-06-15. The last untested combination: amortized q8 quant + the v_dot4 builtin GEMV. Result:
definitive negative. The decode-kernel optimization program is now closed with a measured boundary.

## Result (cli --benchmark, Qwen3-8B, same degraded session)
| config | tok/s | e2e GB/s | MB/token |
|---|---|---|---|
| fp baseline | 30.4 | 144 | 4738 |
| vdot per-linear (D1) | 30.5 | 61 | 2016 |
| vdot + amortized quant (E0) | ~28-30 | 61 | 2016 |

## The finding (closes the standalone->e2e gap for the BEST kernel)
- The v_dot4 int-dot GEMV near-saturates STANDALONE (425 Q4-GB/s = 49.5% peak, 91% of readraw).
- E2E the SAME kernel runs at 61 GB/s (7% peak) -- 7x slower than standalone -- and is null vs fp (30 = 30).
- Amortizing the quant (E0) changed nothing (MB unchanged 2016, tok/s null). So the quant was NOT the wall
  (correcting nothing -- it confirms D1's null was not fixable by amortization).
- So even the BEST possible decode GEMV (near-saturating standalone) does NOT translate e2e. The e2e wall is
  STRUCTURAL -- the per-kernel pipelining/occupancy of the JIT'd 252-launch decode + the q8 quant overhead --
  NOT kernel throughput. A0's "occupancy wall" extends to the lean v_dot4 builtin, not just the scalar int-dot.

## The complete, measured decode conclusion
- KERNEL level: SOLVED. v_dot4 int-dot near-saturates memory (the dequant bottleneck is closed; 91% of readraw).
- E2E level: a 2-7x structural penalty below the standalone kernel, INDEPENDENT of kernel quality. fp GEMV
  171 standalone -> 144 e2e; vdot 425 standalone -> 61 e2e. The e2e never achieves the kernel's bandwidth.
- The structural penalty is the cross-layer / per-kernel-pipelining problem (many small launches, no megakernel,
  quant overhead) that tinygrad cannot express -- the genuine frontier, ruled in by elimination of everything else.

So decode parity with llama.cpp is NOT reachable via kernel optimization on tinygrad/RDNA3, even with a
near-saturating kernel -- because the binding constraint is the e2e per-kernel structure, not the kernel.
Every kernel lever is now measured and exhausted: instruction count, tensor cores, fusion, load-MLP,
reduction-ILP, int-dot near-saturation -- all either null e2e or capped. The decode optimum lives in the
megakernel/cross-layer space (tinygrad-inexpressible) or hand-asm full-decode kernels (the Writer), exactly
the program's thesis boundary.

## What machine search reaches
The positive result stands: machine search has its home on the schedule axis for the BATCHED-matmul regime
(N1/N2/L0/L1, 95% of oracle in ~1 trial, 42x live). Single-stream decode parity is not a kernel-search
problem -- it is a kernel-STRUCTURE (megakernel) problem, below the granularity machine search over tinygrad
operates on. This is the definitive, measured end of the decode-kernel investigation.
