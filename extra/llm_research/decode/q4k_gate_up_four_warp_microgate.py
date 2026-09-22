#!/usr/bin/env python3
"""Standalone correctness + cudaEvent gate for the four-warp Q4_K gate/up candidate.

Control is the installed ``q4k_g3_lanemap_gemv_w1w3fused16_12288_4096``
(grid [12288,1,1], threads [32,1,1]).  Candidate is the research-only
``q4k_gate_up_four_warp_fp16_12288_4096`` (grid [12288,1,1], threads
[32,4,1]).  Both are rendered to CUDA source by the same renderer and compiled
with nvcc sm_120a; timing uses cudaEvent, and correctness compares the fp16
outputs against the installed control (the decode wall gate stays token SHA,
which this standalone gate cannot and does not substitute for).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_w1w3_kernel
from tinygrad.llm.q4k_gate_up_four_warp_mmvq import emit_q4k_gate_up_four_warp_fp16
from tinygrad.llm.qk_layout import Q4_K_BLOCK_ELEMS
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp

CUDA_BIN = "/usr/local/cuda-13.2/bin"
NCU = "/usr/local/bin/ncu"
ROWS, K = 12288, 4096
W_PER_TENSOR = (K // Q4_K_BLOCK_ELEMS) * 36 * ROWS


def _render(ren: CUDARenderer) -> dict[str, str]:
  def p(name: str, shape: tuple[int, ...], dtype, slot: int) -> UOp:
    return UOp.placeholder(shape, dtype, slot)

  out = p("out", (ROWS,), dtypes.float16, 0)
  gate = p("gate", (W_PER_TENSOR,), dtypes.uint32, 1)
  up = p("up", (W_PER_TENSOR,), dtypes.uint32, 2)
  x = p("x", (K,), dtypes.float16, 3)

  control = q4k_g3_lanemap_gemv_w1w3_kernel(ROWS, K, load_style="vector", store_fp16=True)(
    out, gate, up, x)
  candidate = emit_q4k_gate_up_four_warp_fp16(vector_loads=True)(out, gate, up, x)

  def src(u: UOp) -> str:
    return next(x.arg for x in to_program(u, ren).src if x.op is Ops.SOURCE)

  def strip(s: str) -> str:
    marker = 'extern "C" __global__'
    return s[s.index(marker):]

  return {"control": strip(src(control)), "candidate": strip(src(candidate))}


HARNESS = r"""
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
struct __align__(8) half4 { half x, y, z, w; };
__device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x,y,z,w}; return r; }

__SRC_CONTROL__
__SRC_CANDIDATE__

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(2); }
}

static double time_control(half* out, unsigned int* g, unsigned int* u, half* x, int passes) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++)
  q4k_g3_lanemap_gemv_w1w3vec16_12288_4096<<<12288, 32>>>(out, g, u, x);
  cudaEventRecord(e); check(cudaDeviceSynchronize(), "sync control");
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms * 1000.0 / passes;
}

static double time_candidate(half* out, unsigned int* g, unsigned int* u, half* x, int passes) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++)
  q4k_gate_up_four_warp_vec_fp16_12288_4096<<<12288, 128>>>(out, g, u, x);
  cudaEventRecord(e); check(cudaDeviceSynchronize(), "sync candidate");
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms * 1000.0 / passes;
}

int main(int argc, char** argv) {
  int passes = argc > 1 ? atoi(argv[1]) : 200;
  int reps = argc > 2 ? atoi(argv[2]) : 7;
  half* ctrl = nullptr; half* cand = nullptr;
  unsigned int* g = nullptr; unsigned int* u = nullptr; half* x = nullptr;
  check(cudaMalloc(&ctrl, ROWS_ARG * sizeof(half)), "ctrl");
  check(cudaMalloc(&cand, ROWS_ARG * sizeof(half)), "cand");
  check(cudaMalloc(&g, WORDS_ARG * sizeof(unsigned int)), "g");
  check(cudaMalloc(&u, WORDS_ARG * sizeof(unsigned int)), "u");
  check(cudaMalloc(&x, K_ARG * sizeof(half)), "x");
  check(cudaMemset(ctrl, 0, ROWS_ARG * sizeof(half)), "memset ctrl");
  check(cudaMemset(cand, 0, ROWS_ARG * sizeof(half)), "memset cand");
  check(cudaMemset(g, 0, WORDS_ARG * sizeof(unsigned int)), "memset g");
  check(cudaMemset(u, 0, WORDS_ARG * sizeof(unsigned int)), "memset u");
  check(cudaMemset(x, 0, K_ARG * sizeof(half)), "memset x");

    q4k_g3_lanemap_gemv_w1w3vec16_12288_4096<<<12288, 32>>>(ctrl, g, u, x);
    q4k_gate_up_four_warp_vec_fp16_12288_4096<<<12288, 128>>>(cand, g, u, x);
  check(cudaGetLastError(), "warmup launch"); check(cudaDeviceSynchronize(), "warmup sync");

  half* h1 = (half*)malloc(ROWS_ARG * sizeof(half));
  half* h2 = (half*)malloc(ROWS_ARG * sizeof(half));
  check(cudaMemcpy(h1, ctrl, ROWS_ARG * sizeof(half), cudaMemcpyDeviceToHost), "copy ctrl");
  check(cudaMemcpy(h2, cand, ROWS_ARG * sizeof(half), cudaMemcpyDeviceToHost), "copy cand");
  int bitwise = memcmp(h1, h2, ROWS_ARG * sizeof(half)) == 0;
  float max_abs = 0.0f, max_rel = 0.0f, sum_abs = 0.0f;
  for (int i = 0; i < ROWS_ARG; i++) {
    float a = __half2float(h1[i]), b = __half2float(h2[i]);
    float d = fabsf(a - b);
    if (d > max_abs) max_abs = d;
    float denom = fmaxf(fabsf(a), 1e-6f);
    float r = d / denom;
    if (r > max_rel) max_rel = r;
    sum_abs += d;
  }
  float mean_abs = sum_abs / ROWS_ARG;
  printf("bitwise_identical=%d\n", bitwise);
  printf("max_abs_diff=%.9g mean_abs_diff=%.9g max_rel_diff=%.9g\n", max_abs, mean_abs, max_rel);
  free(h1); free(h2);

  printf("shape=12288x4096 passes=%d reps=%d\n", passes, reps);
  for (int r = 0; r < reps; r++) {
    double c = time_control(ctrl, g, u, x, passes);
    double v = time_candidate(cand, g, u, x, passes);
    printf("rep=%d control=%.4f candidate=%.4f\n", r, c, v);
  }
  check(cudaFree(ctrl), "free ctrl"); check(cudaFree(cand), "free cand");
  check(cudaFree(g), "free g"); check(cudaFree(u), "free u"); check(cudaFree(x), "free x");
  return 0;
}
"""


def _ncu(binary: str, symbol: str) -> list[dict[str, str]]:
  metrics = ",".join([
    "dram__bytes.sum", "dram__bytes_op_read.sum", "dram__bytes_op_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed", "gpu__time_duration.sum",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed", "lts__t_bytes.sum",
    "lts__t_sector_op_read_hit_rate.pct", "sm__inst_executed.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed", "launch__registers_per_thread",
  ])
  cp = subprocess.run(["sudo", "-n", NCU, "-k", symbol, "--launch-skip", "1", "--launch-count", "1",
    "--cache-control", "all", "--metrics", metrics, "--csv", binary, "1", "1"],
    capture_output=True, text=True)
  if cp.returncode:
    raise RuntimeError(f"ncu {symbol} failed: {cp.stderr[-4000:]}")
  rows, header = [], None
  for cols in csv.reader(io.StringIO(cp.stdout)):
    if cols and cols[0] == "ID": header = cols; continue
    if header is not None and len(cols) == len(header):
      row = dict(zip(header, cols))
      rows.append({"metric": row["Metric Name"], "unit": row["Metric Unit"], "value": row["Metric Value"]})
  return rows


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--passes", type=int, default=200)
  ap.add_argument("--reps", type=int, default=7)
  ap.add_argument("--ncu", action="store_true")
  ap.add_argument("--out", default="")
  args = ap.parse_args()

  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)
  srcs = _render(ren)
  cu = (HARNESS.replace("__SRC_CONTROL__", srcs["control"])
               .replace("__SRC_CANDIDATE__", srcs["candidate"])
               .replace("ROWS_ARG", str(ROWS))
               .replace("WORDS_ARG", str(W_PER_TENSOR))
               .replace("K_ARG", str(K)))

  with tempfile.TemporaryDirectory(prefix="q4k_gate_up_four_warp_") as td:
    cu_path = os.path.join(td, "gate_up_four_warp.cu")
    binp = os.path.join(td, "gate_up_four_warp")
    with open(cu_path, "w") as f:
      f.write(cu)
    env = dict(os.environ)
    env["PATH"] = f"{CUDA_BIN}:" + env.get("PATH", "")
    cp = subprocess.run(
      ["nvcc", "-arch=sm_120a", "-O3", "-std=c++17", "--ptxas-options=-v",
       cu_path, "-o", binp], capture_output=True, text=True, env=env)
    if cp.returncode != 0:
      print(cp.stderr[-8000:], file=sys.stderr)
      return 3

    r = subprocess.run([binp, str(args.passes), str(args.reps)],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
      print(r.stderr[-4000:], file=sys.stderr)
      return 4

    bitwise = re.search(r"bitwise_identical=(\d)", r.stdout)
    diff = re.search(
      r"max_abs_diff=([0-9.eE+-]+) mean_abs_diff=([0-9.eE+-]+) max_rel_diff=([0-9.eE+-]+)",
      r.stdout)
    c_vals, v_vals = [], []
    for line in r.stdout.splitlines():
      m = re.search(r"rep=(\d+) control=([0-9.]+) candidate=([0-9.]+)", line)
      if m:
        c_vals.append(float(m.group(2)))
        v_vals.append(float(m.group(3)))

    out = {
      "schema": "tinygrad.q4k_gate_up_four_warp_microgate.v1",
      "commit": subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=str(ROOT)).stdout.strip(),
      "method": "render production CUDA -> nvcc sm_120a -> cudaEvent",
      "shape": {"rows": ROWS, "k": K},
      "bitwise_identical": bool(int(bitwise.group(1))) if bitwise else None,
      "max_abs_diff": float(diff.group(1)) if diff else None,
      "mean_abs_diff": float(diff.group(2)) if diff else None,
      "max_rel_diff": float(diff.group(3)) if diff else None,
      "passes": args.passes,
      "reps": args.reps,
      "ptxas": cp.stderr.strip().splitlines(),
      "raw_stdout": r.stdout.strip().splitlines(),
    }
    if c_vals and v_vals:
      med = statistics.median
      out["timing"] = {
        "unit": "us_per_launch_cuda_event",
        "control": c_vals,
        "candidate": v_vals,
        "control_median": med(c_vals),
        "candidate_median": med(v_vals),
        "candidate_over_control": med(v_vals) / med(c_vals),
      }
    if args.ncu:
      out["ncu"] = {
        "method": "Nsight Compute --cache-control all; one post-warmup launch",
        "current": {"symbol": "q4k_gate_up_four_warp_vec_fp16_12288_4096",
          "rows": _ncu(binp, "q4k_gate_up_four_warp_vec_fp16_12288_4096")},
      }

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
      Path(args.out).parent.mkdir(parents=True, exist_ok=True)
      with open(args.out, "w") as f:
        f.write(text + "\n")
    print(text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
