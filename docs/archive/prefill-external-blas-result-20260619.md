# PXB-0/PXB-1 RESULT — external BLAS ceiling is real, but routing remains a boundary decision

Executed the ceiling-first part of `prefill-external-blas-scope-20260619.md` as a reference/control for the prefill
matmul frontier. No tinygrad routing, no default change, no model path change.

## PXB-0 — toolchain workaround → PASS [M]

The original split ROCm failure is real for HIP-language compilation: ROCm 7.2.4 clang appends
`/opt/rocm-7.2.4/include` after `/usr/include`, so `/usr/include/hip` (HIP 5.7) wins and conflicts with the
7.2.4 BLAS headers. The standalone probe does not define GPU kernels, so it can be compiled as host C++ instead:

```sh
g++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 \
  -I/opt/rocm-7.2.4/include -L/opt/rocm-7.2.4/lib -Wl,-rpath,/opt/rocm-7.2.4/lib \
  extra/qk_prefill_blas_ceiling.cpp -lamdhip64 -lrocblas -lhipblaslt \
  -o /tmp/qk_prefill_blas_ceiling
```

This compiles and links against `/opt/rocm-7.2.4/lib/{libamdhip64,librocblas,libhipblaslt}.so`.

## PXB-1 — ceiling measurement → GO for the ceiling, not a route [M]

Artifact: `bench/qk-prefill-external-blas/ceiling.json`. Device: RX 7900 XTX / gfx1100. Timing: HIP events,
10 warmup iterations, 30 timed iterations, fp16 inputs/output with fp32 compute. Current tinygrad reference for the
dominant ffn shape: ~40.8 TFLOPS.

| shape (M,N,K) | rocBLAS TFLOPS | hipBLASLt TFLOPS | best | best / tinygrad | % of 122 TFLOPS peak |
|---|---:|---:|---:|---:|---:|
| ffn_gate/up (512,12288,4096) | 60.96 | **69.81** | hipBLASLt | **1.71×** | 57.2% |
| ffn_down (512,4096,12288) | **70.94** | 64.09 | rocBLAS | **1.74×** | 58.1% |
| attn_q/o (512,4096,4096) | **76.71** | 61.66 | rocBLAS | **1.88×** | 62.9% |
| attn_k/v (512,1024,4096) | **51.82** | 40.87 | rocBLAS | 1.27× | 42.5% |

The dominant ffn_gate/up shape clears the scope's isolated GO gate (>=1.5× current matmul). The external BLAS
ceiling is therefore **not** shape-limited to tinygrad's ~41 TFLOPS plateau. However, it also does **not** hit the
optimistic ~80% peak (~98 TFLOPS) assumption; measured best large-shape results are ~57-63% peak.

## Consequences

- **External BLAS is a valid reference ceiling:** the no-deps tinygrad WMMA work used this measured ~70 TFLOPS
  target; POWN-1 later failed to move beyond 42.0 TFLOPS.
- **The prefill full-pp upside is moderate, not guaranteed 1.6×:** if the whole ~74% matmul bucket moved from
  ~41 TFLOPS to ~70 TFLOPS, the Amdahl upper bound is roughly 1.4-1.45× before bridge/layout overhead. The original
  >=1.5× in-model gate remains necessary and may fail.
- **Routing is still a separate authority boundary:** tinygrad `DEV=AMD` uses HCQ/HSA queues directly; rocBLAS and
  hipBLASLt use the HIP runtime. Passing PXB-1 does not solve VA-pointer interop, synchronization, fallback policy,
  dependency policy, or Tensor/TinyJit capture.
- **Policy state remains no external deps unless changed:** this result is a ceiling/control. It does not overturn
  the pure-tinygrad successor (`prefill-own-wmma-kernel-scope-20260619.md`).

## Verdict

PXB-0/PXB-1 are complete:

- toolchain workaround: **PASS** via host-only C++ compile with explicit ROCm 7.2.4 headers/libs;
- standalone ceiling: **GO** on the dominant prefill GEMM (hipBLASLt 69.8 TFLOPS, 1.71× tinygrad);
- integration: **not attempted** and still policy/runtime-bound.

After POWN-1, the remaining prefill research question is now cleanly split:

1. **no-deps route:** bounded config/shape knobs are refuted; only a deeper codegen/assembly/Tensile-class rewrite
   remains, not a scoped primitive edit.
2. **external route:** if the dependency boundary is accepted later, can the HIP-runtime library call or extracted
   Tensile kernel be bridged into the HCQ model path with enough in-model pp transfer?
