#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" void llama_flash_score_raw(const char *, const char *, const char *, float *, float2 *, int, int,
  uint3, int, int, int, int, int, int, int, int, int, long, int, int, long, int, int, int, int, int, int, long, cudaStream_t);

static void ck(cudaError_t e, const char *s) { if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", s, cudaGetErrorString(e)); std::exit(1); } }
static uint3 fastdiv(unsigned d) { unsigned l=0; while (l<32 && (1u<<l)<d) l++; unsigned m=(unsigned)(((1ull<<32)*((1ull<<l)-d))/d+1); return make_uint3(m,l,d); }

int main(int argc, char **argv) {
  const bool nonzero = argc > 1 && std::string(argv[1]) == "--nonzero";
  constexpr int D=128, T=512, HQ=32, HKV=8;
  std::vector<float> q((size_t)T*HQ*D), out((size_t)T*HQ*D);
  std::vector<__half> k((size_t)T*HKV*D), v((size_t)T*HKV*D);
  for (size_t i=0; i<q.size(); i++) q[i] = nonzero ? 0.001f * float(i%17) : 0.0f;
  for (size_t i=0; i<k.size(); i++) k[i] = __float2half(nonzero ? 0.001f * float(i%11) : 0.0f);
  for (size_t i=0; i<v.size(); i++) v[i] = __float2half(nonzero ? 0.001f * float(i%13) : 0.0f);
  float *dq,*dd; __half *dk,*dv; float2 *dm;
  ck(cudaMalloc(&dq,q.size()*sizeof(float)),"q alloc"); ck(cudaMalloc(&dk,k.size()*sizeof(__half)),"k alloc");
  ck(cudaMalloc(&dv,v.size()*sizeof(__half)),"v alloc"); ck(cudaMalloc(&dd,out.size()*sizeof(float)),"out alloc");
  ck(cudaMalloc(&dm,(size_t)T*HQ*sizeof(float2)),"meta alloc");
  ck(cudaMemcpy(dq,q.data(),q.size()*sizeof(float),cudaMemcpyHostToDevice),"q copy");
  ck(cudaMemcpy(dk,k.data(),k.size()*sizeof(__half),cudaMemcpyHostToDevice),"k copy");
  ck(cudaMemcpy(dv,v.data(),v.size()*sizeof(__half),cudaMemcpyHostToDevice),"v copy"); ck(cudaMemset(dd,0,out.size()*sizeof(float)),"out clear");
  const uint3 ne01 = fastdiv(T);
  llama_flash_score_raw((const char*)dq,(const char*)dk,(const char*)dv,dd,dm,T,D,ne01,HQ,
    1, D*sizeof(float),T*D*sizeof(float),T*HQ*D*sizeof(float), D,T,HKV,1,
    D*sizeof(__half),T*D*sizeof(__half),T*HKV*D*sizeof(__half), D*sizeof(__half),T*D*sizeof(__half),T*HKV*D*sizeof(__half),
    0,0,0,0,0,0);
  ck(cudaGetLastError(),"launch"); ck(cudaDeviceSynchronize(),"sync"); ck(cudaMemcpy(out.data(),dd,out.size()*sizeof(float),cudaMemcpyDeviceToHost),"out copy");
  bool finite=true, zero=true; for(float x:out) { finite &= std::isfinite(x); zero &= x==0.0f; }
  std::printf("fixture=%s finite=%d zero_output=%d\n", nonzero?"nonzero":"zero", finite, zero);
  return finite && (!nonzero ? zero : true) ? 0 : 1;
}
