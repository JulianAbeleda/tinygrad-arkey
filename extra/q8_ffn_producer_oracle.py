#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, subprocess, tempfile

HIP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

typedef struct { half d; half s; signed char qs[32]; } block_q8_1;

static void check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(2);
  }
}

__global__ __launch_bounds__(256, 1) void rmsnorm_only(float *out, const float *x, const float *w, int n, float eps) {
  __shared__ float red[256];
  const int tid = threadIdx.x;
  float ss = 0.0f;
  for (int i = tid; i < n; i += 256) ss += x[i] * x[i];
  red[tid] = ss;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  const float rinv = rsqrtf(red[0] / (float)n + eps);
  for (int i = tid; i < n; i += 256) out[i] = x[i] * rinv * w[i];
}

__global__ __launch_bounds__(256, 1) void rmsnorm_q8_sidechannel(float *out, block_q8_1 *q8,
                                                                 const float *x, const float *w, int n, float eps) {
  __shared__ float red[256];
  const int tid = threadIdx.x;
  float ss = 0.0f;
  for (int i = tid; i < n; i += 256) ss += x[i] * x[i];
  red[tid] = ss;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  const float rinv = rsqrtf(red[0] / (float)n + eps);
  for (int i = tid; i < n; i += 256) out[i] = x[i] * rinv * w[i];
  __syncthreads();
  const int blocks = n / 32;
  for (int b = tid; b < blocks; b += 256) {
    float vals[32];
    float mx = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
      const int idx = b * 32 + j;
      vals[j] = x[idx] * rinv * w[idx];
      mx = fmaxf(mx, fabsf(vals[j]));
    }
    const float scale = (mx == 0.0f) ? 1.0f : mx / 127.0f;
    q8[b].d = __float2half(scale);
    q8[b].s = __float2half(0.0f);
    #pragma unroll
    for (int j = 0; j < 32; j++) {
      int qi = (int)nearbyintf(vals[j] / scale);
      qi = qi < -128 ? -128 : (qi > 127 ? 127 : qi);
      q8[b].qs[j] = (signed char)qi;
    }
  }
}

static float max_abs_fp(const std::vector<float> &a, const std::vector<float> &b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); i++) m = fmaxf(m, fabsf(a[i] - b[i]));
  return m;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s n iters\n", argv[0]);
    return 2;
  }
  const int n = std::atoi(argv[1]);
  const int iters = std::atoi(argv[2]);
  const float eps = 1e-6f;
  if (n % 32 != 0) { fprintf(stderr, "n must be q8 block aligned\n"); return 2; }
  std::vector<float> hx(n), hw(n), ref(n);
  for (int i = 0; i < n; i++) {
    hx[i] = sinf((float)i * 0.013f) * 1.7f + cosf((float)i * 0.007f) * 0.3f;
    hw[i] = 0.7f + 0.001f * (float)(i % 97);
  }
  float ss = 0.0f;
  for (float v: hx) ss += v * v;
  const float rinv = 1.0f / sqrtf(ss / (float)n + eps);
  for (int i = 0; i < n; i++) ref[i] = hx[i] * rinv * hw[i];

  float *dx = nullptr, *dw = nullptr, *dout0 = nullptr, *dout1 = nullptr;
  block_q8_1 *dq8 = nullptr;
  check(hipMalloc(&dx, n * sizeof(float)), "malloc x");
  check(hipMalloc(&dw, n * sizeof(float)), "malloc w");
  check(hipMalloc(&dout0, n * sizeof(float)), "malloc out0");
  check(hipMalloc(&dout1, n * sizeof(float)), "malloc out1");
  check(hipMalloc(&dq8, (n / 32) * sizeof(block_q8_1)), "malloc q8");
  check(hipMemcpy(dx, hx.data(), n * sizeof(float), hipMemcpyHostToDevice), "copy x");
  check(hipMemcpy(dw, hw.data(), n * sizeof(float), hipMemcpyHostToDevice), "copy w");
  for (int i = 0; i < 20; i++) {
    rmsnorm_only<<<1, 256>>>(dout0, dx, dw, n, eps);
    rmsnorm_q8_sidechannel<<<1, 256>>>(dout1, dq8, dx, dw, n, eps);
  }
  check(hipDeviceSynchronize(), "warm sync");

  hipEvent_t s0, e0, s1, e1;
  check(hipEventCreate(&s0), "event s0");
  check(hipEventCreate(&e0), "event e0");
  check(hipEventCreate(&s1), "event s1");
  check(hipEventCreate(&e1), "event e1");
  check(hipEventRecord(s0), "record s0");
  for (int i = 0; i < iters; i++) rmsnorm_only<<<1, 256>>>(dout0, dx, dw, n, eps);
  check(hipEventRecord(e0), "record e0");
  check(hipEventSynchronize(e0), "sync e0");
  check(hipEventRecord(s1), "record s1");
  for (int i = 0; i < iters; i++) rmsnorm_q8_sidechannel<<<1, 256>>>(dout1, dq8, dx, dw, n, eps);
  check(hipEventRecord(e1), "record e1");
  check(hipEventSynchronize(e1), "sync e1");
  float ms0 = 0.0f, ms1 = 0.0f;
  check(hipEventElapsedTime(&ms0, s0, e0), "elapsed0");
  check(hipEventElapsedTime(&ms1, s1, e1), "elapsed1");

  std::vector<float> got0(n), got1(n);
  std::vector<block_q8_1> hq8(n / 32);
  check(hipMemcpy(got0.data(), dout0, n * sizeof(float), hipMemcpyDeviceToHost), "copy out0");
  check(hipMemcpy(got1.data(), dout1, n * sizeof(float), hipMemcpyDeviceToHost), "copy out1");
  check(hipMemcpy(hq8.data(), dq8, (n / 32) * sizeof(block_q8_1), hipMemcpyDeviceToHost), "copy q8");
  float q8_max_abs = 0.0f;
  for (int b = 0; b < n / 32; b++) {
    const float d = __half2float(hq8[b].d);
    for (int j = 0; j < 32; j++) q8_max_abs = fmaxf(q8_max_abs, fabsf(ref[b*32+j] - d * (float)hq8[b].qs[j]));
  }
  const float only_us = ms0 * 1000.0f / (float)iters;
  const float side_us = ms1 * 1000.0f / (float)iters;
  printf("{\"n\":%d,\"iters\":%d,\"rmsnorm_us\":%.6f,\"sidechannel_us\":%.6f,"
         "\"incremental_us\":%.6f,\"fp_max_abs_rmsnorm\":%.9g,\"fp_max_abs_sidechannel\":%.9g,"
         "\"q8_dequant_max_abs\":%.9g,\"q8_blocks\":%d,\"q8_bytes\":%zu}\n",
         n, iters, only_us, side_us, side_us - only_us, max_abs_fp(got0, ref), max_abs_fp(got1, ref),
         q8_max_abs, n / 32, (n / 32) * sizeof(block_q8_1));
  hipFree(dx); hipFree(dw); hipFree(dout0); hipFree(dout1); hipFree(dq8);
  return 0;
}
"""

def main() -> None:
  parser = argparse.ArgumentParser(description="Q8H-3 fused RMSNorm/q8 producer oracle")
  parser.add_argument("--n", type=int, default=4096)
  parser.add_argument("--iters", type=int, default=20000)
  parser.add_argument("--arch", default="gfx1100")
  parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-handwritten-oracle/producer_cost.json"))
  args = parser.parse_args()

  with tempfile.TemporaryDirectory(prefix="q8h3_") as td:
    tmp = pathlib.Path(td)
    src, exe = tmp/"q8h3_producer.hip", tmp/"q8h3_producer"
    src.write_text(HIP_SOURCE)
    subprocess.run(["hipcc", f"--offload-arch={args.arch}", "-O3", str(src), "-o", str(exe)], check=True)
    proc = subprocess.run([str(exe), str(args.n), str(args.iters)], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  result = json.loads(proc.stdout.strip().splitlines()[-1])
  result.update({
    "date": "2026-06-19",
    "phase": "Q8H-3",
    "arch": args.arch,
    "stderr": proc.stderr.strip(),
    "strong_gate_incremental_us_lte": 4.8,
    "verdict": "PASS" if result["incremental_us"] <= 4.8 and result["fp_max_abs_sidechannel"] <= 1e-5 else "FAIL",
  })
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  if result["verdict"] != "PASS": raise SystemExit(1)

if __name__ == "__main__":
  main()
