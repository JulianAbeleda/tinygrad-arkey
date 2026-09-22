
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fffffff))
#endif
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }

extern "C" __global__ void __launch_bounds__(32) reduce_output_rmsnorm_32_128(float* data0_4096, float* data1_4096, half* data2_128) {
  int gidx0 = blockIdx.x; /* 32 */
  int lidx1 = threadIdx.y; /* 8 */
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (gidx0<<7);
  for (int Ridx2 = 0; Ridx2 < 16; Ridx2++) {
    float val0 = (*(data1_4096+(lidx1+alu1+(Ridx2<<3))));
    (*(buf0+0)) = ((*(buf0+0))+(val0*val0));
  }
  int lidx0 = threadIdx.x; /* 4 */
  __shared__ __align__(16) float buf1[8];
  if ((lidx0<1)) {
    *(buf1+lidx1) = (*(buf0+0));
  }
  __syncthreads();
  float val1 = (*(buf1+0));
  float val2 = (*(buf1+1));
  float val3 = (*(buf1+2));
  float val4 = (*(buf1+3));
  float val5 = (*(buf1+4));
  float val6 = (*(buf1+5));
  float val7 = (*(buf1+6));
  float val8 = (*(buf1+7));
  int alu8 = (lidx1+(lidx0<<3));
  for (int gidx2 = 0; gidx2 < 4; gidx2++) {
    int alu9 = (gidx2<<5);
    half val9 = (*(data2_128+(alu8+alu9)));
    int alu10 = (alu8+alu1+alu9);
    float val10 = (*(data1_4096+alu10));
    *(data0_4096+alu10) = (val10*(1/sqrt((((val1+val2+val3+val4+val5+val6+val7+val8)*0.0078125f)+1e-06f)))*((float)(val9)));
  }
}
extern "C" __global__ void __launch_bounds__(32) reduce_output_rmsnorm_8_128(float* data0_1024, float* data1_1024, half* data2_128) {
  int gidx0 = blockIdx.x; /* 8 */
  int lidx1 = threadIdx.y; /* 16 */
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (gidx0<<7);
  for (int Ridx2 = 0; Ridx2 < 8; Ridx2++) {
    float val0 = (*(data1_1024+(alu1+(lidx1<<3)+Ridx2)));
    (*(buf0+0)) = ((*(buf0+0))+(val0*val0));
  }
  int lidx0 = threadIdx.x; /* 2 */
  __shared__ __align__(16) float buf1[16];
  if ((lidx0<1)) {
    *(buf1+lidx1) = (*(buf0+0));
  }
  __syncthreads();
  float val1 = (*(buf1+0));
  float val2 = (*(buf1+1));
  float val3 = (*(buf1+2));
  float val4 = (*(buf1+3));
  float val5 = (*(buf1+4));
  float val6 = (*(buf1+5));
  float val7 = (*(buf1+6));
  float val8 = (*(buf1+7));
  float val9 = (*(buf1+8));
  float val10 = (*(buf1+9));
  float val11 = (*(buf1+10));
  float val12 = (*(buf1+11));
  float val13 = (*(buf1+12));
  float val14 = (*(buf1+13));
  float val15 = (*(buf1+14));
  float val16 = (*(buf1+15));
  int alu8 = (lidx1+(lidx0<<4));
  for (int gidx2 = 0; gidx2 < 4; gidx2++) {
    int alu9 = (gidx2<<5);
    half val17 = (*(data2_128+(alu8+alu9)));
    int alu10 = (alu8+alu1+alu9);
    float val18 = (*(data1_1024+alu10));
    *(data0_1024+alu10) = (val18*(1/sqrt((((val1+val2+val3+val4+val5+val6+val7+val8+val9+val10+val11+val12+val13+val14+val15+val16)*0.0078125f)+1e-06f)))*((float)(val17)));
  }
}
extern "C" __global__ void __launch_bounds__(32) reduce_output_rmsnorm_rope_32_128(float* data0_4096, float* data1_4096, half* data2_128, float* data3_128) {
  int gidx0 = blockIdx.x; /* 32 */
  int lidx1 = threadIdx.y; /* 8 */
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (gidx0<<7);
  for (int Ridx2 = 0; Ridx2 < 16; Ridx2++) {
    float val0 = (*(data1_4096+(lidx1+alu1+(Ridx2<<3))));
    (*(buf0+0)) = ((*(buf0+0))+(val0*val0));
  }
  int lidx0 = threadIdx.x; /* 4 */
  __shared__ __align__(16) float buf1[8];
  if ((lidx0<1)) {
    *(buf1+lidx1) = (*(buf0+0));
  }
  __syncthreads();
  float val1 = (*(buf1+0));
  float val2 = (*(buf1+1));
  float val3 = (*(buf1+2));
  float val4 = (*(buf1+3));
  float val5 = (*(buf1+4));
  float val6 = (*(buf1+5));
  float val7 = (*(buf1+6));
  float val8 = (*(buf1+7));
  float alu8 = (1/sqrt((((val1+val2+val3+val4+val5+val6+val7+val8)*0.0078125f)+1e-06f)));
  int alu9 = (lidx1+(lidx0<<3));
  for (int gidx2 = 0; gidx2 < 2; gidx2++) {
    int alu10 = (gidx2<<5);
    int alu11 = (alu9+alu10);
    half val9 = (*(data2_128+alu11));
    int alu12 = (alu11+64);
    half val10 = (*(data2_128+alu12));
    int alu13 = (alu9+alu1+alu10);
    float val11 = (*(data1_4096+alu13));
    int alu14 = (alu13+64);
    float val12 = (*(data1_4096+alu14));
    float val13 = (*(data3_128+alu11));
    float val14 = (*(data3_128+alu12));
    float alu15 = (val11*alu8*((float)(val9)));
    float alu16 = (val12*alu8*((float)(val10)));
    *(data0_4096+alu13) = ((alu15*val13)-(alu16*val14));
    *(data0_4096+alu14) = ((alu16*val13)+(alu15*val14));
  }
}
extern "C" __global__ void __launch_bounds__(32) reduce_output_rmsnorm_rope_8_128(float* data0_1024, float* data1_1024, half* data2_128, float* data3_128) {
  int gidx0 = blockIdx.x; /* 8 */
  int lidx1 = threadIdx.y; /* 16 */
  float buf0[1];
  (*(buf0+0)) = 0.0f;
  int alu1 = (gidx0<<7);
  for (int Ridx2 = 0; Ridx2 < 8; Ridx2++) {
    float val0 = (*(data1_1024+(alu1+(lidx1<<3)+Ridx2)));
    (*(buf0+0)) = ((*(buf0+0))+(val0*val0));
  }
  int lidx0 = threadIdx.x; /* 2 */
  __shared__ __align__(16) float buf1[16];
  if ((lidx0<1)) {
    *(buf1+lidx1) = (*(buf0+0));
  }
  __syncthreads();
  float val1 = (*(buf1+0));
  float val2 = (*(buf1+1));
  float val3 = (*(buf1+2));
  float val4 = (*(buf1+3));
  float val5 = (*(buf1+4));
  float val6 = (*(buf1+5));
  float val7 = (*(buf1+6));
  float val8 = (*(buf1+7));
  float val9 = (*(buf1+8));
  float val10 = (*(buf1+9));
  float val11 = (*(buf1+10));
  float val12 = (*(buf1+11));
  float val13 = (*(buf1+12));
  float val14 = (*(buf1+13));
  float val15 = (*(buf1+14));
  float val16 = (*(buf1+15));
  float alu8 = (1/sqrt((((val1+val2+val3+val4+val5+val6+val7+val8+val9+val10+val11+val12+val13+val14+val15+val16)*0.0078125f)+1e-06f)));
  int alu9 = (lidx1+(lidx0<<4));
  for (int gidx2 = 0; gidx2 < 2; gidx2++) {
    int alu10 = (gidx2<<5);
    int alu11 = (alu9+alu10);
    half val17 = (*(data2_128+alu11));
    int alu12 = (alu11+64);
    half val18 = (*(data2_128+alu12));
    int alu13 = (alu9+alu1+alu10);
    float val19 = (*(data1_1024+alu13));
    int alu14 = (alu13+64);
    float val20 = (*(data1_1024+alu14));
    float val21 = (*(data3_128+alu11));
    float val22 = (*(data3_128+alu12));
    float alu15 = (val19*alu8*((float)(val17)));
    float alu16 = (val20*alu8*((float)(val18)));
    *(data0_1024+alu13) = ((alu15*val21)-(alu16*val22));
    *(data0_1024+alu14) = ((alu16*val21)+(alu15*val22));
  }
}

// Faithful copy of the half-rotate rope applied by tinygrad apply_rope:
// out[0:64] = x[0:64]*cos - x[64:128]*sin ; out[64:128] = x[64:128]*cos + x[0:64]*sin
// freqs is (128,) = [cos(64), sin(64)].
__global__ void rope_128(const float* x, float* out, const float* freqs) {
  int row = blockIdx.x;
  int e = threadIdx.x;
  int h = e & 63;
  float cosv = freqs[h];
  float sinv = freqs[h + 64];
  float xi = x[row * 128 + e];
  if (e < 64) out[row * 128 + e] = xi * cosv - x[row * 128 + e + 64] * sinv;
  else        out[row * 128 + e] = xi * cosv + x[row * 128 + e - 64] * sinv;
}

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(2); }
}

static void launch_norm(int which, float* out, float* x, half* w) {
  if (which == 0) reduce_output_rmsnorm_32_128<<<dim3(32,1,1), dim3(4,8)>>>(out, x, w);
  else            reduce_output_rmsnorm_8_128<<<dim3(8,1,1), dim3(2,16)>>>(out, x, w);
}
static void launch_fused(int which, float* out, float* x, half* w, float* freqs) {
  if (which == 0) reduce_output_rmsnorm_rope_32_128<<<dim3(32,1,1), dim3(4,8)>>>(out, x, w, freqs);
  else            reduce_output_rmsnorm_rope_8_128<<<dim3(8,1,1), dim3(2,16)>>>(out, x, w, freqs);
}

static double hot_loop(int which, int rows, int iters, float* out, float* x, half* w, float* freqs, int fused) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  if (fused) launch_fused(which, out, x, w, freqs); else { launch_norm(which, out, x, w); rope_128<<<rows, 128>>>(out, out, freqs); }
  cudaDeviceSynchronize();
  cudaEventRecord(s);
  for (int i = 0; i < iters; i++) {
    if (fused) launch_fused(which, out, x, w, freqs);
    else { launch_norm(which, out, x, w); rope_128<<<rows, 128>>>(out, out, freqs); }
  }
  cudaEventRecord(e); cudaDeviceSynchronize();
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms * 1000.0 / iters;
}

static double max_abs_diff(const float* a, const float* b, int n) {
  double m = 0.0;
  for (int i = 0; i < n; i++) m = fmax(m, fabs((double)a[i] - (double)b[i]));
  return m;
}

static unsigned int f2i(float f) { unsigned int u; memcpy(&u, &f, 4); return u; }

int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 2000;
  int rows = argc > 2 ? atoi(argv[2]) : 32;  // 32 = Q, 8 = K
  int which = rows == 32 ? 0 : 1;
  int n = rows * 128;

  float* x; float* ctrl; float* cand; float* freqs; half* w;
  check(cudaMalloc(&x, n * sizeof(float)), "x");
  check(cudaMalloc(&ctrl, n * sizeof(float)), "ctrl");
  check(cudaMalloc(&cand, n * sizeof(float)), "cand");
  check(cudaMalloc(&freqs, 128 * sizeof(float)), "freqs");
  check(cudaMalloc(&w, 128 * sizeof(half)), "w");

  float* hx = (float*)malloc(n * sizeof(float));
  float* hf = (float*)malloc(128 * sizeof(float));
  half* hw = (half*)malloc(128 * sizeof(half));
  for (int i = 0; i < n; i++) hx[i] = 0.1f + 0.001f * (float)(i % 17);
  for (int i = 0; i < 128; i++) { hf[i] = (i < 64) ? 0.9f + 0.001f * (float)i : 0.4f - 0.001f * (float)(i - 64); }
  for (int i = 0; i < 128; i++) hw[i] = __float2half(1.0f + 0.001f * (float)(i % 5));
  check(cudaMemcpy(x, hx, n * sizeof(float), cudaMemcpyHostToDevice), "x h2d");
  check(cudaMemcpy(freqs, hf, 128 * sizeof(float), cudaMemcpyHostToDevice), "f h2d");
  check(cudaMemcpy(w, hw, 128 * sizeof(half), cudaMemcpyHostToDevice), "w h2d");

  // Control: norm then rope.  Candidate: fused.
  launch_norm(which, ctrl, x, w); rope_128<<<rows, 128>>>(ctrl, ctrl, freqs);
  launch_fused(which, cand, x, w, freqs);
  check(cudaDeviceSynchronize(), "sync");

  float* hc = (float*)malloc(n * sizeof(float));
  float* hd = (float*)malloc(n * sizeof(float));
  check(cudaMemcpy(hc, ctrl, n * sizeof(float), cudaMemcpyDeviceToHost), "ctrl d2h");
  check(cudaMemcpy(hd, cand, n * sizeof(float), cudaMemcpyDeviceToHost), "cand d2h");
  printf("rows=%d max_abs_diff=%.9g\n", rows, max_abs_diff(hc, hd, n));
  for (int i = 0, printed = 0; i < n && printed < 8; i++) {
    if (f2i(hc[i]) != f2i(hd[i])) {
      printf("  idx=%d ctrl=%08x (%.9g) cand=%08x (%.9g)\n",
             i, f2i(hc[i]), hc[i], f2i(hd[i]), hd[i]);
      printed++;
    }
  }
  printf("hot control us=%.4f\n", hot_loop(which, rows, iters, ctrl, x, w, freqs, 0));
  printf("hot candidate us=%.4f\n", hot_loop(which, rows, iters, cand, x, w, freqs, 1));
  check(cudaGetLastError(), "last error");
  return 0;
}
