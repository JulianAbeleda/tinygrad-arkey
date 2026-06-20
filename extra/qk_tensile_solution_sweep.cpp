#define ROCBLAS_BETA_FEATURES_API
// Host-only rocBLAS: enumerate ALL solutions for the gateup GEMM (m=512,n=12288,k=4096, HHS) and dispatch each once,
// so the capture shim records every variant's kernarg. No __global__ kernels (avoids the split device-toolchain).
//   g++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm-7.2.4/include extra/qk_tensile_solution_sweep.cpp \
//       -L/opt/rocm-7.2.4/lib -lrocblas -lamdhip64 -o /tmp/qk_sweep
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <vector>
int main(){
  rocblas_handle h; rocblas_create_handle(&h);
  hipStream_t st; hipStreamCreate(&st); rocblas_set_stream(h,st);
  int m=512,n=12288,k=4096; float alpha=1.f,beta=0.f;
  void *a,*b,*c; hipMalloc(&a,(size_t)m*k*2); hipMalloc(&b,(size_t)k*n*2); hipMalloc(&c,(size_t)m*n*2);
  hipMemset(a,0,(size_t)m*k*2); hipMemset(b,0,(size_t)k*n*2); hipMemset(c,0,(size_t)m*n*2);
  auto args=[&](rocblas_gemm_algo algo,int32_t sol){ return rocblas_gemm_ex(h,
      rocblas_operation_none,rocblas_operation_none,m,n,k,&alpha,
      a,rocblas_datatype_f16_r,m, b,rocblas_datatype_f16_r,k, &beta,
      c,rocblas_datatype_f16_r,m, c,rocblas_datatype_f16_r,m,
      rocblas_datatype_f32_r, algo, sol, 0); };
  rocblas_int ls=0;
  rocblas_gemm_ex_get_solutions(h,rocblas_operation_none,rocblas_operation_none,m,n,k,&alpha,
      a,rocblas_datatype_f16_r,m,b,rocblas_datatype_f16_r,k,&beta,
      c,rocblas_datatype_f16_r,m,c,rocblas_datatype_f16_r,m,rocblas_datatype_f32_r,
      rocblas_gemm_algo_solution_index,0,nullptr,&ls);
  printf("num_solutions=%d\n",ls);
  std::vector<rocblas_int> idx(ls);
  rocblas_gemm_ex_get_solutions(h,rocblas_operation_none,rocblas_operation_none,m,n,k,&alpha,
      a,rocblas_datatype_f16_r,m,b,rocblas_datatype_f16_r,k,&beta,
      c,rocblas_datatype_f16_r,m,c,rocblas_datatype_f16_r,m,rocblas_datatype_f32_r,
      rocblas_gemm_algo_solution_index,0,idx.data(),&ls);
  int ok=0;
  for(auto i:idx){ if(args(rocblas_gemm_algo_solution_index,i)==rocblas_status_success){ hipStreamSynchronize(st); ok++; } }
  printf("dispatched_ok=%d / %d\n",ok,ls);
  return 0;
}
