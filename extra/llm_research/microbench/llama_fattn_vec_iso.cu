#include "ggml-cuda/fattn-vec.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

// Standalone isolated probe for llama flash_attn_ext_vec, mirroring
// tinygrad's nv_flash_body_device_timing.py methodology: back-to-back
// launches on the default stream, timed by nsys CUPTI (kernel duration).
// Launch config pinned from the llama d512 trace: grid (1,2,32), block (32,4,1),
// Hd=128, Q->ne=(128,1,1,1), K/V->ne=(128,512,8,1), fp16 K/V, no mask.

int main(int argc, char** argv) {
  int replays = 400, warmup = 20, grids_y = 2;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--replays")) replays = atoi(argv[++i]);
    if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[++i]);
    if (!strcmp(argv[i], "--gridy")) grids_y = atoi(argv[++i]);
  }
  constexpr int D = 128, Hq = 32, Hkv = 8, TC = 512;
  constexpr int ncols = 1;
  constexpr ggml_type type_K = GGML_TYPE_F16, type_V = GGML_TYPE_F16;
  constexpr bool softcap = false;

  // Allocate: Q [Hq*D] f16, K/V [Hkv*TC*D] f16, dst [Hq*D] f32, dst_meta [Hq*2] f32
  std::vector<__half> q(Hq*D), kv(2*Hkv*TC*D);
  std::vector<float> dst(Hq*D), meta(Hq*2);
  for (size_t i = 0; i < q.size(); i++) q[i] = __float2half(0.001f * (float)(i % 7));
  for (size_t i = 0; i < kv.size(); i++) kv[i] = __float2half(0.001f * (float)(i % 11));

  __half *d_q, *d_k, *d_v; float *d_dst; float2 *d_meta;
  cudaMalloc(&d_q, Hq*D*sizeof(__half));
  cudaMalloc(&d_k, Hkv*TC*D*sizeof(__half));
  cudaMalloc(&d_v, Hkv*TC*D*sizeof(__half));
  cudaMalloc(&d_dst, Hq*D*sizeof(float));
  cudaMalloc(&d_meta, Hq*2*sizeof(float));
  cudaMemcpy(d_q, q.data(), Hq*D*sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, kv.data(), Hkv*TC*D*sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, kv.data()+Hkv*TC*D, Hkv*TC*D*sizeof(__half), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  dim3 grid(1, grids_y, Hq);      // pinned: (1,2,32); sweep 1/2/4
  dim3 block(32, 4, 1);           // pinned: (32,4,1)
  const uint3 ne01 = init_fastdiv_values(1);  // Q->ne[1]=1 (decode)
  const float scale = 1.0f / sqrtf(D);

  auto launch = [&]() {
    flash_attn_ext_vec<D, ncols, type_K, type_V, softcap><<<grid, block>>>(
      (const char*)d_q, (const char*)d_k, (const char*)d_v,
      nullptr, nullptr, nullptr, d_dst, d_meta,
      scale, 0.0f, 0.0f, 0.0f, Hq, 0.0f,
      D, ne01, 1, 1,                     // Q ne00..03
      D*sizeof(__half), D*sizeof(__half), 0,  // Q nb01..03
      D, TC, Hkv, 1,                     // K ne10..13
      D*sizeof(__half), TC*D*sizeof(__half), 0,  // K nb11..13
      D*sizeof(__half), TC*D*sizeof(__half), 0,  // V nb21..23
      0, 0, 0,                           // mask ne31..33
      0, 0, 0);                          // mask nb31..33
  };

  for (int i = 0; i < warmup; i++) launch();
  cudaDeviceSynchronize();
  for (int i = 0; i < replays; i++) launch();
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  printf("err=%s replays=%d\n", cudaGetErrorString(err), replays);
  cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_dst); cudaFree(d_meta);
  return err == cudaSuccess ? 0 : 1;
}
