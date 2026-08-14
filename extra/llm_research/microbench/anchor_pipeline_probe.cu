// anchor_pipeline_probe.cu - does llama's activation-quantize-behind-matmul
// pipeline work on this driver at decode sizes?
//
// llama's decode graph splits each GEMV into quantize_q8_1 (~2.2 us) +
// mul_mat_vec_q (~16 us) and pipelines the quantize behind the matmul. We fuse
// both into one kernel. This probe models the two structures with real event
// dependencies and measures span vs node-sum.
//
//   matmul_i   depends on prep_i (reads the quantized activation)
//   matmul_i+1 depends on matmul_i (activation chain)
//   prep_i+1   independent of matmul_i (its own activation is ready)
//
// Arm "fused": one kernel per layer = prep + matmul (our current shape).
// Arm "split": prep on stream A, matmul on stream B, event prep_i -> matmul_i.
//
//   nvcc -O3 -arch=sm_120 -o anchor_pipeline_probe anchor_pipeline_probe.cu
//   ./anchor_pipeline_probe --layers 217 --rows 2048 --k 4096 --json /tmp/anchor.json

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define CUDA_CHECK(x) do { \
  cudaError_t e_ = (x); \
  if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(e_), (int)e_); \
    exit(1); \
  } \
} while (0)

// prep: quantize the activation into xq (small, launch-bound, ~2 us).
__global__ void prep_kernel(const float* __restrict__ x, float* __restrict__ xq, int k) {
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += stride)
    xq[i] = roundf(x[i] * 127.0f) * (1.0f / 127.0f);
}

// matmul: dot each weight row with the quantized activation (compute-bound).
__global__ void matmul_kernel(const float* __restrict__ w, const float* __restrict__ xq,
                              float* __restrict__ y, int rows, int k) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const float* wr = w + (long long)row * k;
  float acc = 0.0f;
  #pragma unroll 8
  for (int j = 0; j < k; j++) acc += wr[j] * xq[j];
  y[row] = acc;
}

// fused: prep + matmul in one kernel (our current shape).
__global__ void fused_kernel(const float* __restrict__ w, const float* __restrict__ x,
                             float* __restrict__ y, int rows, int k) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const float* wr = w + (long long)row * k;
  float acc = 0.0f;
  #pragma unroll 8
  for (int j = 0; j < k; j++) acc += wr[j] * roundf(x[j] * 127.0f) * (1.0f / 127.0f);
  y[row] = acc;
}

static double elapsed(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return (double)ms * 1000.0;
}

int main(int argc, char** argv) {
  int layers = 217, rows = 2048, k = 4096;
  const char* out = "/tmp/anchor.json";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--layers") && i+1 < argc) layers = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--rows") && i+1 < argc) rows = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--k") && i+1 < argc) k = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--json") && i+1 < argc) out = argv[++i];
  }

  cudaStream_t sa, sb;
  CUDA_CHECK(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));

  long long wn = (long long)rows * k;
  float *w, *x, *xq, *yf, *ys;
  CUDA_CHECK(cudaMalloc(&w, wn * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&x, (long long)k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&xq, (long long)k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&yf, (long long)rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ys, (long long)rows * sizeof(float)));

  float* hw = (float*)malloc(wn * sizeof(float));
  float* hx = (float*)malloc((long long)k * sizeof(float));
  for (long long i = 0; i < wn; i++) hw[i] = (float)((i % 101) - 50) / 50.0f;
  for (int i = 0; i < k; i++) hx[i] = (float)((i % 37) - 18) / 18.0f;
  CUDA_CHECK(cudaMemcpy(w, hw, wn * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(x, hx, (long long)k * sizeof(float), cudaMemcpyHostToDevice));

  int pg = (k + 255) / 256, mg = (rows + 31) / 32;
  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));

  // fused arm: serial on one stream.
  cudaEventRecord(ev0, sa);
  for (int l = 0; l < layers; l++)
    fused_kernel<<<mg, 32, 0, sa>>>(w, x, yf, rows, k);
  cudaEventRecord(ev1, sa);
  CUDA_CHECK(cudaDeviceSynchronize());
  double fused_span = elapsed(ev0, ev1);

  // split arm: prep stream A, matmul stream B, per-layer event.
  cudaEvent_t* ready = (cudaEvent_t*)malloc(layers * sizeof(cudaEvent_t));
  for (int l = 0; l < layers; l++) CUDA_CHECK(cudaEventCreate(&ready[l]));
  cudaEventRecord(ev0, sb);
  for (int l = 0; l < layers; l++) {
    prep_kernel<<<pg, 256, 0, sa>>>(x, xq, k);
    CUDA_CHECK(cudaEventRecord(ready[l], sa));
    CUDA_CHECK(cudaStreamWaitEvent(sb, ready[l], 0));
    matmul_kernel<<<mg, 32, 0, sb>>>(w, xq, ys, rows, k);
  }
  cudaEventRecord(ev1, sb);
  CUDA_CHECK(cudaDeviceSynchronize());
  double split_span = elapsed(ev0, ev1);

  float* hyf = (float*)malloc((long long)rows * sizeof(float));
  float* hys = (float*)malloc((long long)rows * sizeof(float));
  CUDA_CHECK(cudaMemcpy(hyf, yf, (long long)rows * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hys, ys, (long long)rows * sizeof(float), cudaMemcpyDeviceToHost));
  double maxerr = 0;
  for (int i = 0; i < rows; i++) {
    double e = fabs((double)hyf[i] - (double)hys[i]);
    if (e > maxerr) maxerr = e;
  }

  double saving = fused_span - split_span;
  double pct = 100.0 * saving / fused_span;
  printf("layers=%d rows=%d k=%d\n", layers, rows, k);
  printf("fused_span=%.1f us  split_span=%.1f us  saving=%.1f us (%.2f%%)  maxerr=%.2e\n",
         fused_span, split_span, saving, pct, maxerr);
  printf("PASS=%s\n", (pct >= 5.0 && maxerr < 1e-2) ? "yes" : "no");

  if (out) {
    FILE* f = fopen(out, "w");
    if (f) {
      fprintf(f, "{\"layers\":%d,\"rows\":%d,\"k\":%d,\"fused_span_us\":%.3f,"
              "\"split_span_us\":%.3f,\"saving_us\":%.3f,\"saving_pct\":%.3f,\"max_err\":%.3e}\n",
              layers, rows, k, fused_span, split_span, saving, pct, maxerr);
      fclose(f);
    }
  }
  return 0;
}
