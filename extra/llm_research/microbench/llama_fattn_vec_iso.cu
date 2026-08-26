#include "ggml-cuda/fattn-vec.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

// Standalone isolated probe for llama flash_attn_ext_vec, mirroring
// tinygrad's nv_flash_body_device_timing.py methodology: back-to-back
// launches on the default stream, timed by nsys CUPTI (kernel duration).
// Launch config pinned from the current llama d512 trace: grid (1,6,32),
// block (32,4,1), Hd=128, Q->ne=(128,1,1,1), K/V fp16, no mask.
// The CLI keeps grid-y and logical KV length variable so the partition service
// curve can be measured without rebuilding.  Output allocations include the
// parallel-block dimension (the older probe underallocated these buffers for
// grid-y > 1, making that sweep invalid).

int main(int argc, char** argv) {
  int replays = 400, warmup = 20, grids_y = 6, tc = 768;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--replays")) replays = atoi(argv[++i]);
    if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[++i]);
    if (!strcmp(argv[i], "--gridy")) grids_y = atoi(argv[++i]);
    if (!strcmp(argv[i], "--tc")) tc = atoi(argv[++i]);
  }
  constexpr int D = 128, Hq = 32, Hkv = 8, MAX_TC = 768;
  if (grids_y < 1 || grids_y > 16 || tc < 1 || tc > MAX_TC) {
    fprintf(stderr, "invalid --gridy/--tc\n");
    return 2;
  }
  constexpr int ncols = 1;
  constexpr ggml_type type_K = GGML_TYPE_F16, type_V = GGML_TYPE_F16;
  constexpr bool softcap = false;

  // Allocate for the maximum KV length and every parallel output part.
  std::vector<__half> q(Hq*D), kv(2*Hkv*MAX_TC*D);
  for (size_t i = 0; i < q.size(); i++) q[i] = __float2half(0.001f * (float)(i % 7));
  for (size_t i = 0; i < kv.size(); i++) kv[i] = __float2half(0.001f * (float)(i % 11));

  __half *d_q, *d_k, *d_v; float *d_dst; float2 *d_meta;
  cudaMalloc(&d_q, Hq*D*sizeof(__half));
  cudaMalloc(&d_k, Hkv*MAX_TC*D*sizeof(__half));
  cudaMalloc(&d_v, Hkv*MAX_TC*D*sizeof(__half));
  cudaMalloc(&d_dst, Hq*grids_y*D*sizeof(float));
  cudaMalloc(&d_meta, Hq*grids_y*sizeof(float2));
  cudaMemcpy(d_q, q.data(), Hq*D*sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, kv.data(), Hkv*MAX_TC*D*sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, kv.data()+Hkv*MAX_TC*D, Hkv*MAX_TC*D*sizeof(__half), cudaMemcpyHostToDevice);
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
      D, tc, Hkv, 1,                     // K ne10..13
      D*sizeof(__half), MAX_TC*D*sizeof(__half), 0,  // K nb11..13
      D*sizeof(__half), MAX_TC*D*sizeof(__half), 0,  // V nb21..23
      0, 0, 0,                           // mask ne31..33
      0, 0, 0);                          // mask nb31..33
  };

  for (int i = 0; i < warmup; i++) launch();
  cudaDeviceSynchronize();
  cudaEvent_t begin, end;
  cudaEventCreate(&begin); cudaEventCreate(&end);
  cudaEventRecord(begin);
  for (int i = 0; i < replays; i++) launch();
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, begin, end);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  printf("err=%s replays=%d gridy=%d tc=%d us_per_launch=%.6f\n",
         cudaGetErrorString(err), replays, grids_y, tc, 1000.0f*elapsed_ms/replays);
  cudaEventDestroy(begin); cudaEventDestroy(end);
  cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_dst); cudaFree(d_meta);
  return err == cudaSuccess ? 0 : 1;
}
