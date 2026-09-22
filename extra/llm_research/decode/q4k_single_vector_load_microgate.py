#!/usr/bin/env python3
"""Standalone correctness + cudaEvent gate for the vectorized-load Q4_K single-projection GEMV.

Control is the installed ``q4k_g3_lanemap_gemv_{rows}_{k}`` (scalar loads). Candidate is the
research vectorized spelling ``q4k_g3_lanemap_gemv_vec_{rows}_{k}`` (uint4 header loads,
deduplicated qpack loads, half4 activation loads, same 32-lane/1-row geometry). The dot product
is bit-identical to the scalar spelling by construction; this gate checks that claim against
nvcc-compiled production CUDA and reports cudaEvent per-launch timing. The full decode wall gate
remains token SHA, which this standalone gate cannot substitute for.
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
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel
from tinygrad.llm.qk_layout import Q4_K_BLOCK_ELEMS
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp

CUDA_BIN = "/usr/local/cuda-13.2/bin"
NCU = "/usr/local/bin/ncu"


def _ncu_csv(binary: str, symbol: str) -> tuple[list[dict[str, str]], str]:
  """Profile one post-warmup launch with NCU's explicit cold-cache replay."""
  metrics = ",".join([
    "dram__bytes.sum",
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__time_duration.sum",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_bytes.sum",
    "lts__t_sector_op_read_hit_rate.pct",
    "sm__inst_executed.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__registers_per_thread",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
  ])
  cp = subprocess.run([
    "sudo", "-n", NCU, "-k", symbol, "--launch-skip", "1", "--launch-count", "1",
    "--cache-control", "all", "--metrics", metrics, "--csv", binary, "1", "1",
  ], capture_output=True, text=True)
  rows: list[dict[str, str]] = []
  header: list[str] | None = None
  for cols in csv.reader(io.StringIO(cp.stdout)):
    if cols and cols[0] == "ID":
      header = cols
      continue
    if header is not None and len(cols) == len(header):
      row = dict(zip(header, cols))
      rows.append({"metric": row["Metric Name"], "unit": row["Metric Unit"], "value": row["Metric Value"]})
  if cp.returncode != 0:
    raise RuntimeError(f"ncu failed for {symbol} (rc={cp.returncode}): {cp.stderr[-4000:]}")
  return rows, cp.stderr.strip()


def _render(ren: CUDARenderer, rows: int, k: int) -> dict[str, str]:
  w_per_tensor = (k // Q4_K_BLOCK_ELEMS) * 36 * rows

  def p(name: str, shape: tuple[int, ...], dtype, slot: int) -> UOp:
    return UOp.placeholder(shape, dtype, slot)

  out = p("out", (rows,), dtypes.float32, 0)
  words = p("words", (w_per_tensor,), dtypes.uint32, 1)
  x = p("x", (k,), dtypes.float16, 2)

  control = q4k_g3_lanemap_gemv_kernel(rows, k, load_style="scalar")(out, words, x)
  candidate = q4k_g3_lanemap_gemv_kernel(rows, k, load_style="vector")(out, words, x)

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

static double time_control(float* out, unsigned int* w, half* x, int passes) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++)
    q4k_g3_lanemap_gemv_CTRL<<<ROWS_ARG, 32>>>(out, w, x);
  cudaEventRecord(e); check(cudaDeviceSynchronize(), "sync control");
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms * 1000.0 / passes;
}

static double time_candidate(float* out, unsigned int* w, half* x, int passes) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++)
    q4k_g3_lanemap_gemv_CAND<<<ROWS_ARG, 32>>>(out, w, x);
  cudaEventRecord(e); check(cudaDeviceSynchronize(), "sync candidate");
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms * 1000.0 / passes;
}

int main(int argc, char** argv) {
  int passes = argc > 1 ? atoi(argv[1]) : 200;
  int reps = argc > 2 ? atoi(argv[2]) : 7;
  float* ctrl = nullptr; float* cand = nullptr;
  unsigned int* w = nullptr; half* x = nullptr;
  check(cudaMalloc(&ctrl, ROWS_ARG * sizeof(float)), "ctrl");
  check(cudaMalloc(&cand, ROWS_ARG * sizeof(float)), "cand");
  check(cudaMalloc(&w, WORDS_ARG * sizeof(unsigned int)), "w");
  check(cudaMalloc(&x, K_ARG * sizeof(half)), "x");
  check(cudaMemset(ctrl, 0, ROWS_ARG * sizeof(float)), "memset ctrl");
  check(cudaMemset(cand, 0, ROWS_ARG * sizeof(float)), "memset cand");
  // Non-zero deterministic fill so the bitwise compare exercises the real dot product rather
  // than trivially matching two all-zero outputs.
  unsigned int* hw = (unsigned int*)malloc(WORDS_ARG * sizeof(unsigned int));
  half* hx = (half*)malloc(K_ARG * sizeof(half));
  for (int i = 0; i < WORDS_ARG; i++) hw[i] = (unsigned int)((i * 2654435761u) ^ 0x9e3779b9u);
  for (int i = 0; i < K_ARG; i++) hx[i] = __float2half(((i % 257) - 128) * 0.03125f);
  check(cudaMemcpy(w, hw, WORDS_ARG * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy w");
  check(cudaMemcpy(x, hx, K_ARG * sizeof(half), cudaMemcpyHostToDevice), "copy x");
  free(hw); free(hx);

  q4k_g3_lanemap_gemv_CTRL<<<ROWS_ARG, 32>>>(ctrl, w, x);
  q4k_g3_lanemap_gemv_CAND<<<ROWS_ARG, 32>>>(cand, w, x);
  check(cudaGetLastError(), "warmup launch"); check(cudaDeviceSynchronize(), "warmup sync");

  float* h1 = (float*)malloc(ROWS_ARG * sizeof(float));
  float* h2 = (float*)malloc(ROWS_ARG * sizeof(float));
  check(cudaMemcpy(h1, ctrl, ROWS_ARG * sizeof(float), cudaMemcpyDeviceToHost), "copy ctrl");
  check(cudaMemcpy(h2, cand, ROWS_ARG * sizeof(float), cudaMemcpyDeviceToHost), "copy cand");
  int bitwise = memcmp(h1, h2, ROWS_ARG * sizeof(float)) == 0;
  float max_abs = 0.0f, sum_abs = 0.0f;
  for (int i = 0; i < ROWS_ARG; i++) {
    float d = fabsf(h1[i] - h2[i]);
    if (d > max_abs) max_abs = d;
    sum_abs += d;
  }
  float mean_abs = sum_abs / ROWS_ARG;
  printf("bitwise_identical=%d\n", bitwise);
  printf("max_abs_diff=%.9g mean_abs_diff=%.9g\n", max_abs, mean_abs);
  free(h1); free(h2);

  printf("shape=ROWS_ARGxK_ARG passes=%d reps=%d\n", passes, reps);
  for (int r = 0; r < reps; r++) {
    double c = time_control(ctrl, w, x, passes);
    double v = time_candidate(cand, w, x, passes);
    printf("rep=%d control=%.4f candidate=%.4f\n", r, c, v);
  }
  check(cudaFree(ctrl), "free ctrl"); check(cudaFree(cand), "free cand");
  check(cudaFree(w), "free w"); check(cudaFree(x), "free x");
  return 0;
}
"""


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--rows", type=int, default=4096)
  ap.add_argument("--k", type=int, default=4096)
  ap.add_argument("--passes", type=int, default=200)
  ap.add_argument("--reps", type=int, default=7)
  ap.add_argument("--ncu", action="store_true",
                  help="also collect an explicit cold-cache NCU counter bracket")
  ap.add_argument("--out", default="")
  args = ap.parse_args()

  rows, k = args.rows, args.k
  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)
  srcs = _render(ren, rows, k)
  ctrl_name = f"q4k_g3_lanemap_gemv_{rows}_{k}"
  cand_name = f"q4k_g3_lanemap_gemv_vec_{rows}_{k}"

  cu = (HARNESS
        .replace("__SRC_CONTROL__", srcs["control"])
        .replace("__SRC_CANDIDATE__", srcs["candidate"])
        .replace("ROWS_ARG", str(rows))
        .replace("K_ARG", str(k))
        .replace("WORDS_ARG", str((k // Q4_K_BLOCK_ELEMS) * 36 * rows))
        .replace("q4k_g3_lanemap_gemv_CTRL", ctrl_name)
        .replace("q4k_g3_lanemap_gemv_CAND", cand_name))

  with tempfile.TemporaryDirectory(prefix="q4k_single_vector_") as td:
    cu_path = os.path.join(td, "single_vector.cu")
    binp = os.path.join(td, "single_vector")
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
      r"max_abs_diff=([0-9.eE+-]+) mean_abs_diff=([0-9.eE+-]+)", r.stdout)
    c_vals, v_vals = [], []
    for line in r.stdout.splitlines():
      m = re.search(r"rep=(\d+) control=([0-9.]+) candidate=([0-9.]+)", line)
      if m:
        c_vals.append(float(m.group(2)))
        v_vals.append(float(m.group(3)))

    out = {
      "schema": "tinygrad.q4k_single_vector_load_microgate.v1",
      "commit": subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=str(ROOT)).stdout.strip(),
      "method": "render production CUDA -> nvcc sm_120a -> cudaEvent",
      "shape": {"rows": rows, "k": k},
      "epilogue": "",
      "bitwise_identical": bool(int(bitwise.group(1))) if bitwise else None,
      "max_abs_diff": float(diff.group(1)) if diff else None,
      "mean_abs_diff": float(diff.group(2)) if diff else None,
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
      out["ncu"] = {"method": "Nsight Compute --cache-control all; one post-warmup launch"}
      for arm, symbol in (("control", ctrl_name), ("candidate", cand_name)):
        rows, stderr = _ncu_csv(binp, symbol)
        out["ncu"][arm] = {"symbol": symbol, "rows": rows}
        if stderr:
          out["ncu"][arm]["stderr_tail"] = stderr[-2000:]

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
      Path(args.out).parent.mkdir(parents=True, exist_ok=True)
      with open(args.out, "w") as f:
        f.write(text + "\n")
    print(text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
