// Baseline experiment (no tinygrad dependency added): time the FULL per-layer prefill matmul SEQUENCE with rocBLAS,
// to estimate the prefill tok/s ceiling if the matmuls ran at library speed. Standalone host C++ (sidesteps the
// split-HIP header issue, same as qk_prefill_blas_ceiling.cpp).
//   g++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm-7.2.4/include -L/opt/rocm-7.2.4/lib \
//       -Wl,-rpath,/opt/rocm-7.2.4/lib extra/qk_prefill_blas_sequence.cpp -lamdhip64 -lrocblas -o /tmp/seq
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <vector>
#define CK(x) do{ auto e=(x); if(e!=hipSuccess){printf("hip err %d @ %s\n",(int)e,#x); return 1;} }while(0)
struct GEMM{ const char*name; int M,N,K; };

int main(){
  const int T=512, L=36;                       // Qwen3-8B: 36 layers, prefill ubatch 512
  // per-layer prefill matmuls (M=T rows): C[M,N]=A[M,K]*B[K,N]
  std::vector<GEMM> layer = {
    {"attn_q",T,4096,4096},{"attn_k",T,1024,4096},{"attn_v",T,1024,4096},{"attn_o",T,4096,4096},
    {"ffn_gate",T,12288,4096},{"ffn_up",T,12288,4096},{"ffn_down",T,4096,12288},
  };
  size_t maxA=(size_t)T*12288, maxB=(size_t)12288*12288, maxC=(size_t)T*12288; // generous scratch (B = K*N up to 50M)
  __half *dA,*dB,*dC; CK(hipMalloc(&dA,maxA*2)); CK(hipMalloc(&dB,maxB*2)); CK(hipMalloc(&dC,maxC*2));
  rocblas_handle h; rocblas_create_handle(&h); hipStream_t s; hipStreamCreate(&s); rocblas_set_stream(h,s);
  float alpha=1.f,beta=0.f;
  auto gemm=[&](const GEMM&g){
    return rocblas_gemm_ex(h, rocblas_operation_none, rocblas_operation_none, g.M, g.N, g.K,
      &alpha, dA, rocblas_datatype_f16_r, g.M, dB, rocblas_datatype_f16_r, g.K, &beta,
      dC, rocblas_datatype_f16_r, g.M, dC, rocblas_datatype_f16_r, g.M,
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
  };
  auto run_forward=[&](){ for(int l=0;l<L;l++) for(auto&g:layer) gemm(g); };  // all matmuls of one prefill forward
  // warm
  for(int i=0;i<5;i++) run_forward(); CK(hipStreamSynchronize(s));
  hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b); int IT=20;
  hipEventRecord(a,s); for(int i=0;i<IT;i++) run_forward(); hipEventRecord(b,s); hipEventSynchronize(b);
  float ms; hipEventElapsedTime(&ms,a,b); double matmul_ms = ms/IT;
  // total FLOPs/forward
  double flop=0; for(int l=0;l<L;l++) for(auto&g:layer) flop += 2.0*g.M*g.N*g.K;
  printf("=== rocBLAS prefill MATMUL SEQUENCE (8B, %d layers, T=%d) ===\n", L, T);
  printf("rocBLAS matmul-only time/forward: %.2f ms  (%.1f TFLOPS effective over the sequence)\n", matmul_ms, flop/(matmul_ms*1e-3)/1e12);
  // PREFILL_V2 reference: 245 ms/forward warm, ~74%% matmul -> ~181 ms matmul, ~64 ms non-matmul (attn SDPA+norm)
  double pv2_forward=245.0, nonmm=0.26*pv2_forward;
  double new_forward = matmul_ms + nonmm;
  printf("\nBaseline projection (non-matmul attn/norm held at PREFILL_V2's ~%.0f ms):\n", nonmm);
  printf("  PREFILL_V2 (current):     %.0f ms/forward -> %.0f tok/s\n", pv2_forward, T/(pv2_forward/1e3));
  printf("  rocBLAS matmuls + same non-mm: %.0f ms -> %.0f tok/s  (%.2fx PREFILL_V2)\n", new_forward, T/(new_forward/1e3), pv2_forward/new_forward);
  printf("  llama.cpp prefill ~3069 tok/s for reference.\n");
  rocblas_destroy_handle(h); return 0;
}
