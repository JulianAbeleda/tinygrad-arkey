#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, subprocess, tempfile, time

import numpy as np

from extra.qk_layout import GGML_Q4_K, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, read_metadata, tensor_shape

HIP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define QK_K 256
typedef struct { half d; half dmin; unsigned char scales[12]; unsigned char qs[128]; } block_q4_K;
typedef struct { half d; half s; signed char qs[32]; } block_q8_1;

static void check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(2);
  }
}

static std::vector<unsigned char> read_bytes(const char *path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "open failed: %s\n", path); std::exit(2); }
  f.seekg(0, std::ios::end);
  size_t n = (size_t)f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<unsigned char> out(n);
  f.read((char*)out.data(), n);
  return out;
}

static std::vector<float> read_f32(const char *path) {
  auto b = read_bytes(path);
  if (b.size() % sizeof(float) != 0) { fprintf(stderr, "bad f32 file size\n"); std::exit(2); }
  std::vector<float> out(b.size() / sizeof(float));
  std::memcpy(out.data(), b.data(), b.size());
  return out;
}

__device__ __host__ static inline void get_scale_min(int j, const unsigned char *q, unsigned char *d, unsigned char *m) {
  if (j < 4) { *d = q[j] & 63; *m = q[j+4] & 63; }
  else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j] >> 6) << 4); }
}

#define NW 4
#define WS 32
__global__ void __launch_bounds__(NW*WS, 1) mmvq_q4k(float *dst, const block_q4_K *x, const block_q8_1 *y,
                                                     int ncols, int nrows) {
  const int row = blockIdx.x;
  const int tid = threadIdx.y * WS + threadIdx.x;
  const int nbpr = ncols / QK_K;
  float tmp = 0.0f;
  for (int kb = tid / 8; kb < nbpr; kb += (NW * WS) / 8) {
    const block_q4_K *bx = x + row * nbpr + kb;
    const int sub = tid & 7;
    const block_q8_1 *by = y + kb * 8 + sub;
    unsigned char sc, mn;
    get_scale_min(sub, bx->scales, &sc, &mn);
    const unsigned char *q = bx->qs + (sub / 2) * 32;
    int sumi = 0, sumq = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
      int qb = ((const int*)q)[k];
      int nib = (sub & 1) ? ((qb >> 4) & 0x0F0F0F0F) : (qb & 0x0F0F0F0F);
      int u = ((const int*)by->qs)[k];
      sumi = __builtin_amdgcn_sudot4(false, nib, true, u, sumi, false);
      sumq = __builtin_amdgcn_sudot4(false, 0x01010101, true, u, sumq, false);
    }
    const float d8 = __half2float(by->d);
    const float dd = __half2float(bx->d), dm = __half2float(bx->dmin);
    tmp += d8 * (dd * (float)sc * (float)sumi - dm * (float)mn * (float)sumq);
  }
  __shared__ float sm[NW];
  for (int o = WS / 2; o > 0; o >>= 1) tmp += __shfl_xor(tmp, o);
  if (threadIdx.x == 0) sm[threadIdx.y] = tmp;
  __syncthreads();
  if (tid == 0) {
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < NW; i++) s += sm[i];
    dst[row] = s;
  }
}

int main(int argc, char **argv) {
  if (argc != 7) {
    fprintf(stderr, "usage: %s q4.bin q8.bin ref.bin rows ncols iters\n", argv[0]);
    return 2;
  }
  const int rows = std::atoi(argv[4]);
  const int ncols = std::atoi(argv[5]);
  const int iters = std::atoi(argv[6]);
  const int nbpr = ncols / QK_K;
  auto q4 = read_bytes(argv[1]);
  auto q8 = read_bytes(argv[2]);
  auto ref = read_f32(argv[3]);
  if (q4.size() != (size_t)rows * nbpr * sizeof(block_q4_K)) { fprintf(stderr, "bad q4 size\n"); return 2; }
  if (q8.size() != (size_t)(ncols / 32) * sizeof(block_q8_1)) { fprintf(stderr, "bad q8 size\n"); return 2; }
  if (ref.size() != (size_t)rows) { fprintf(stderr, "bad ref size\n"); return 2; }

  block_q4_K *dq4 = nullptr;
  block_q8_1 *dq8 = nullptr;
  float *dout = nullptr;
  check(hipMalloc(&dq4, q4.size()), "hipMalloc q4");
  check(hipMalloc(&dq8, q8.size()), "hipMalloc q8");
  check(hipMalloc(&dout, rows * sizeof(float)), "hipMalloc out");
  check(hipMemcpy(dq4, q4.data(), q4.size(), hipMemcpyHostToDevice), "copy q4");
  check(hipMemcpy(dq8, q8.data(), q8.size(), hipMemcpyHostToDevice), "copy q8");
  dim3 grid(rows), block(WS, NW);
  for (int i = 0; i < 10; i++) mmvq_q4k<<<grid, block>>>(dout, dq4, dq8, ncols, rows);
  check(hipGetLastError(), "warm launch");
  check(hipDeviceSynchronize(), "warm sync");

  hipEvent_t s, e;
  check(hipEventCreate(&s), "event s");
  check(hipEventCreate(&e), "event e");
  check(hipEventRecord(s), "event record s");
  for (int i = 0; i < iters; i++) mmvq_q4k<<<grid, block>>>(dout, dq4, dq8, ncols, rows);
  check(hipEventRecord(e), "event record e");
  check(hipEventSynchronize(e), "event sync e");
  float ms = 0.0f;
  check(hipEventElapsedTime(&ms, s, e), "elapsed");

  std::vector<float> got(rows);
  check(hipMemcpy(got.data(), dout, rows * sizeof(float), hipMemcpyDeviceToHost), "copy out");
  float max_abs = 0.0f, max_rel = 0.0f, mean_abs = 0.0f;
  int worst = 0;
  for (int i = 0; i < rows; i++) {
    const float err = std::fabs(got[i] - ref[i]);
    const float rel = err / std::max(1e-6f, std::fabs(ref[i]));
    mean_abs += err;
    if (err > max_abs) { max_abs = err; worst = i; }
    if (rel > max_rel) max_rel = rel;
  }
  mean_abs /= (float)rows;
  const float us = ms * 1000.0f / (float)iters;
  const double q4_gbs = ((double)q4.size()) / (us * 1e-6) / 1e9;
  printf("{\"rows\":%d,\"ncols\":%d,\"iters\":%d,\"us\":%.6f,\"q4_gbs\":%.6f,"
         "\"max_abs\":%.9g,\"max_rel\":%.9g,\"mean_abs\":%.9g,\"worst_row\":%d,"
         "\"worst_got\":%.9g,\"worst_ref\":%.9g}\n",
         rows, ncols, iters, us, q4_gbs, max_abs, max_rel, mean_abs, worst, got[worst], ref[worst]);
  hipFree(dq4); hipFree(dq8); hipFree(dout);
  return 0;
}
"""

def q8_blocks(x:np.ndarray) -> bytes:
  blocks = x.astype(np.float32).reshape(-1, 32)
  scales = np.max(np.abs(blocks), axis=1) / 127.0
  scales = np.where(scales == 0, 1.0, scales).astype(np.float32)
  qs = np.rint(blocks / scales[:, None]).clip(-128, 127).astype(np.int8)
  out = bytearray()
  for d, q in zip(scales.astype(np.float16), qs):
    out += np.float16(d).tobytes()
    out += np.float16(0.0).tobytes()
    out += q.tobytes()
  return bytes(out)

def q4_ref_rows(q4:bytes, rows:int, k:int, x:np.ndarray) -> np.ndarray:
  nbpr = k // Q4_K_BLOCK_ELEMS
  x_blocks = x.astype(np.float32).reshape(nbpr, 8, 32)
  out = np.empty(rows, dtype=np.float32)
  for row in range(rows):
    acc = np.float32(0.0)
    for kb in range(nbpr):
      off = (row * nbpr + kb) * Q4_K_BLOCK_BYTES
      block = q4[off:off + Q4_K_BLOCK_BYTES]
      d = np.frombuffer(block[0:2], dtype=np.float16).astype(np.float32)[0]
      dmin = np.frombuffer(block[2:4], dtype=np.float16).astype(np.float32)[0]
      s = np.frombuffer(block[4:16], dtype=np.uint8)
      sc = np.concatenate((s[0:4] & 63, (s[8:12] & 0xF) | ((s[0:4] >> 6) << 4))).astype(np.float32)
      mn = np.concatenate((s[4:8] & 63, (s[8:12] >> 4) | ((s[4:8] >> 6) << 4))).astype(np.float32)
      qs = np.frombuffer(block[16:144], dtype=np.uint8).reshape(4, 32)
      q = np.stack((qs & 0xF, qs >> 4), axis=1).reshape(8, 32).astype(np.float32)
      vals = d * sc[:, None] * q - dmin * mn[:, None]
      acc += np.sum(vals * x_blocks[kb], dtype=np.float32)
    out[row] = acc
  return out

def tensor_info(meta, name:str):
  matches = [x for x in meta.infos if x.name == name]
  if not matches: raise ValueError(f"tensor {name!r} not found")
  info = matches[0]
  if info.typ != GGML_Q4_K: raise ValueError(f"{name} is ggml_type={info.typ}, expected Q4_K")
  shape = tensor_shape(info)
  if len(shape) != 2: raise ValueError(f"{name} is not a matrix: shape={shape}")
  return info, shape

def run_one(args, tmp:pathlib.Path, exe:pathlib.Path, tensor:str) -> dict:
  meta = read_metadata(args.gguf)
  info, shape = tensor_info(meta, tensor)
  rows, k = min(args.rows or shape[0], shape[0]), shape[1]
  if k % Q4_K_BLOCK_ELEMS != 0: raise ValueError(f"K={k} is not Q4_K aligned")
  row_bytes = k // Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES
  byte_start = meta.data_start + info.off
  with args.gguf.open("rb") as f:
    f.seek(byte_start)
    q4 = f.read(rows * row_bytes)
  rng = np.random.default_rng(args.seed)
  x = rng.standard_normal(k).astype(np.float32)
  q8 = q8_blocks(x)
  # Dequantize q8 exactly as the kernel sees it for the q8-lossy reference.
  xq = np.empty(k, dtype=np.float32)
  for bi in range(k // 32):
    d = np.frombuffer(q8[bi*36:bi*36+2], dtype=np.float16).astype(np.float32)[0]
    q = np.frombuffer(q8[bi*36+4:bi*36+36], dtype=np.int8).astype(np.float32)
    xq[bi*32:(bi+1)*32] = d * q
  ref = q4_ref_rows(q4, rows, k, xq)

  q4_path, q8_path, ref_path = tmp/f"{tensor}.q4.bin", tmp/"x.q8.bin", tmp/f"{tensor}.ref.bin"
  q4_path.write_bytes(q4)
  q8_path.write_bytes(q8)
  ref_path.write_bytes(ref.tobytes())
  st = time.perf_counter()
  proc = subprocess.run([str(exe), str(q4_path), str(q8_path), str(ref_path), str(rows), str(k), str(args.iters)],
                        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  elapsed = time.perf_counter() - st
  line = proc.stdout.strip().splitlines()[-1]
  result = json.loads(line)
  result.update({
    "tensor": tensor, "shape": list(shape), "rows_tested": rows,
    "q4_bytes": len(q4), "q8_bytes": len(q8), "host_elapsed_s": elapsed,
    "stderr": proc.stderr.strip(),
  })
  return result

def main() -> None:
  parser = argparse.ArgumentParser(description="Q8H-1 handwritten Q4_K x q8_1 MMVQ correctness oracle")
  parser.add_argument("gguf", type=pathlib.Path)
  parser.add_argument("--tensor", action="append", default=["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"])
  parser.add_argument("--rows", type=int, default=1024, help="rows per tensor to verify; use 12288 for full gate/up")
  parser.add_argument("--iters", type=int, default=50)
  parser.add_argument("--seed", type=int, default=1337)
  parser.add_argument("--arch", default="gfx1100")
  parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-handwritten-oracle/mmvq_correctness.json"))
  args = parser.parse_args()

  with tempfile.TemporaryDirectory(prefix="q8h1_") as td:
    tmp = pathlib.Path(td)
    src, exe = tmp/"q8h1_mmvq.hip", tmp/"q8h1_mmvq"
    src.write_text(HIP_SOURCE)
    subprocess.run(["hipcc", f"--offload-arch={args.arch}", "-O3", str(src), "-o", str(exe)], check=True)
    results = [run_one(args, tmp, exe, t) for t in args.tensor]

  passed = all(r["max_abs"] <= 2e-2 for r in results)
  artifact = {
    "date": "2026-06-19",
    "phase": "Q8H-1",
    "model": str(args.gguf),
    "arch": args.arch,
    "rows_requested": args.rows,
    "iters": args.iters,
    "tolerance_max_abs": 2e-2,
    "results": results,
    "verdict": "PASS" if passed else "FAIL",
    "next": "Q8H-2 fused RMSNorm/q8 producer oracle" if passed else "STOP: fix real-GGUF handwritten MMVQ correctness first",
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(artifact, indent=2) + "\n")
  print(json.dumps(artifact, indent=2))
  if not passed: raise SystemExit(1)

if __name__ == "__main__":
  main()
