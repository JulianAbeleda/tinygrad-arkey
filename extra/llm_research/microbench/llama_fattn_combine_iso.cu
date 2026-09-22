#include "ggml-cuda/fattn-common.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

// Isolated retained-llama six-part decode combine.  The production geometry is
// grid=(1,32,1), block=(128,1,1), D=128, parallel_blocks=6.
int main(int argc, char **argv) {
  int replays = 2000, warmup = 100;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--replays")) replays = atoi(argv[++i]);
    if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[++i]);
  }
  constexpr int D = 128, H = 32, S = 6;
  std::vector<float> parts(H*S*D), dst(H*D);
  std::vector<float2> meta(H*S);
  for (size_t i = 0; i < parts.size(); ++i) parts[i] = 0.001f*float(int(i%23)-11);
  for (size_t i = 0; i < meta.size(); ++i) meta[i] = make_float2(0.01f*float(i%7), 1.0f+0.01f*float(i%5));
  float *d_parts, *d_dst; float2 *d_meta;
  cudaMalloc(&d_parts, parts.size()*sizeof(float)); cudaMalloc(&d_meta, meta.size()*sizeof(float2));
  cudaMalloc(&d_dst, dst.size()*sizeof(float));
  cudaMemcpy(d_parts, parts.data(), parts.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_meta, meta.data(), meta.size()*sizeof(float2), cudaMemcpyHostToDevice);
  auto launch = [&]() { flash_attn_combine_results<D><<<dim3(1,H,1),dim3(D,1,1),S*sizeof(float2)>>>(d_parts,d_meta,d_dst,S); };
  for (int i=0;i<warmup;++i) launch(); cudaDeviceSynchronize();
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
  for (int i=0;i<replays;++i) launch();
  cudaEventRecord(b); cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b);
  cudaMemcpy(dst.data(),d_dst,dst.size()*sizeof(float),cudaMemcpyDeviceToHost);
  double checksum=0; for(float x:dst) checksum+=x;
  printf("err=%s us_per_launch=%.6f checksum=%.9f\n",cudaGetErrorString(cudaGetLastError()),1000.0*ms/replays,checksum);
  cudaFree(d_parts); cudaFree(d_meta); cudaFree(d_dst); return 0;
}
