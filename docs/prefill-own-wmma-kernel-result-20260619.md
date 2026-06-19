# POWN-0/1 RESULT — pure tinygrad WMMA config sweep does not break the 34% plateau

Executed the no-external-dependency prefill WMMA sweep from `prefill-own-wmma-kernel-scope-20260619.md`.
Probe: `extra/qk_prefill_wmma_sweep.py`. Artifact: `bench/qk-prefill-own-wmma/sweep.txt`. No route/default/model
change.

## Gate

Dominant ffn shape: M=512, K=4096, N=12288, fp16. Current tinygrad WMMA: ~40.8-42.0 TFLOPS. Gate:
**>=62 TFLOPS** (>=1.5× current, ~50% of assumed 122 TFLOPS WMMA peak). External control from PXB-1:
hipBLASLt **69.8 TFLOPS** on this same shape.

## Sweep result [M]

| config | threads | acc/thread | TFLOPS | % peak | verdict |
|---|---:|---:|---:|---:|---|
| B128x128x16 W2x2 | 128 | 128 | **42.02** | 34% | best / baseline |
| B128x128x16 W4x2 | 256 | 64 | 28.36 | 23% | slower |
| B128x128x16 W2x4 | 256 | 64 | 31.75 | 26% | slower |
| B128x128x16 W4x4 | 512 | 32 | 28.15 | 23% | slower |
| B128x256x16 W2x2 | 128 | 256 | 11.49 | 9% | much slower |
| B256x128x16 W2x2 | 128 | 256 | 11.68 | 10% | much slower |
| B256x256x16 W2x2 | 128 | 512 | 11.52 | 9% | much slower |
| B128x128x32 W2x2 | 128 | 128 | 37.27 | 31% | slower |
| B128x128x16 W1x1 | 32 | 512 | 11.73 | 10% | much slower |
| noLDS B128x128x16 W2x2 | 128 | 128 | 37.21 | 30% | slower |
| noLDS B128x128x16 W2x4 | 256 | 64 | 27.39 | 22% | slower |

All correct configs produced MSE ~6.7e-7 vs fp16 oracle. The best value is the current 128-thread/128-accumulator
shape at **42.0 TFLOPS**. More waves/fewer accumulators, bigger tiles, larger BLOCK_K, and removing LDS all regress.

## Interpretation

The scoped pure-tinygrad knobs do **not** expose hidden WMMA issue headroom. The original suspicion that more waves
or lower per-thread accumulator pressure would improve occupancy/latency hiding is refuted for this kernel family:
it makes the kernel slower. Dropping LDS also hurts, even though explicit LDS tiling did not help in PWLT-A2.

The remaining ~42 -> ~70 TFLOPS gap to hipBLASLt is therefore not a bounded config choice in the current
`custom_kernel` shape. It is likely in the class of Tensile-style software pipelining, instruction scheduling,
kernel selection, or lower-level WMMA codegen/assembly control.

## Verdict

**POWN-1 KILL:** no pure-tinygrad config in this bounded sweep reaches the >=62 TFLOPS isolated gate. Do not route.
Do not reopen as "try more waves", "bigger tile", "BLOCK_K", or "drop LDS" without new evidence.

The prefill matmul frontier now has only two honest outcomes:

1. accept PREFILL_V2 as the no-deps resting point, or
2. make an explicit external/raw-HIP/Tensile boundary decision if the measured external ceiling is worth the
   runtime/dependency work.
