// EBT-1 Lane-A shim: run a rocBLAS fp16 GEMM directly on tinygrad-owned (HCQ/KFD) VRAM pointers, no copies.
// Host C++ (sidesteps the split-HIP header issue, like qk_prefill_blas_ceiling.cpp). Built as a .so, called via ctypes.
//   g++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 -shared -fPIC -I/opt/rocm-7.2.4/include \
//       -L/opt/rocm-7.2.4/lib -Wl,-rpath,/opt/rocm-7.2.4/lib extra/qk_prefill_bridge_shim.cpp -lamdhip64 -lrocblas \
//       -o /tmp/qk_bridge.so
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <cstdint>
#include <cstdio>

extern "C" {
// Probe HIP's view of a tinygrad VA pointer. Returns hipError code; fills memtype (-1 if attr call failed) and
// whether HIP's resolved devicePointer matches the input (i.e. usable as a device pointer without copy).
int ebt1_pointer_attr(uint64_t va, int *memtype, int *dev_ptr_matches) {
  hipPointerAttribute_t at; for (int i=0;i<(int)sizeof(at);i++) ((char*)&at)[i]=0;
  hipError_t e = hipPointerGetAttributes(&at, (const void*)va);
  *memtype = (e==hipSuccess) ? (int)at.type : -1;
  *dev_ptr_matches = (e==hipSuccess && (uint64_t)at.devicePointer==va) ? 1 : 0;
  return (int)e;
}

// Run C[M,N] (row-major) = A[M,K] * B[K,N] on the given device VA pointers (fp16, fp32 compute), no copies.
// row-major via col-major: gemm(none,none, N,M,K, B(ldN), A(ldK), C(ldN)). warm iters, then timed iters.
// Returns rocblas_status (0=success). gemm_ms = avg over timed iters. last_hip = hip error after sync.
int ebt1_gemm(uint64_t a_va, uint64_t b_va, uint64_t c_va, int M, int N, int K,
              int warm, int iters, double *gemm_ms, int *last_hip) {
  rocblas_handle h; if (rocblas_create_handle(&h)!=rocblas_status_success) return -100;
  hipStream_t s; hipStreamCreate(&s); rocblas_set_stream(h, s);
  const float alpha=1.0f, beta=0.0f;
  const __half *A=(const __half*)a_va, *B=(const __half*)b_va; __half *C=(__half*)c_va;
  auto call=[&](){ return rocblas_gemm_ex(h, rocblas_operation_none, rocblas_operation_none,
      N, M, K, &alpha, B, rocblas_datatype_f16_r, N, A, rocblas_datatype_f16_r, K,
      &beta, C, rocblas_datatype_f16_r, N, C, rocblas_datatype_f16_r, N,
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0); };
  rocblas_status st = call();
  if (st != rocblas_status_success) { rocblas_destroy_handle(h); return (int)st; }
  *last_hip = (int)hipStreamSynchronize(s);           // did the GEMM on tinygrad pointers actually complete?
  for (int i=0;i<warm;i++) call(); hipStreamSynchronize(s);
  hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1);
  hipEventRecord(e0,s); for(int i=0;i<iters;i++) call(); hipEventRecord(e1,s); hipEventSynchronize(e1);
  float ms=0; hipEventElapsedTime(&ms,e0,e1); *gemm_ms = ms/iters;
  hipEventDestroy(e0); hipEventDestroy(e1); rocblas_destroy_handle(h);
  return 0;
}
}
