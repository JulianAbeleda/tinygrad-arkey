#!/usr/bin/env python3
"""Standalone cudaEvent bracket for K four-warp Q4_K vs the installed K GEMV.

The Python-side microgate is correct but its host wall-clock loop cannot
resolve a ~5 us kernel because tinygrad graph replay adds ~120 us of host
overhead.  This tool renders both production-CUDA sources with CUDARenderer,
compiles one standalone binary with nvcc, and times control/candidate/control
with cudaEvent so only device elapsed time is compared.

It is measurement tooling only: no route, selector, or production import uses
the candidate, and no model numerics change.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
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
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.decode.q4k_exact_group_factorized import emit_q4k_exact_four_warp

CUDA_BIN = "/usr/local/cuda-13.2/bin"
NCU = "/usr/local/bin/ncu"


def _render(ren: CUDARenderer) -> dict[str, str]:
  def p(name: str, shape: tuple[int, ...], dtype, slot: int) -> UOp:
    return UOp.placeholder(shape, dtype, slot)

  control = q4k_g3_lanemap_gemv_kernel(1024, 4096)(
    p("out", (1024,), dtypes.float32, 0),
    p("words", (1024 * 16 * 36,), dtypes.uint32, 1),
    p("x", (4096,), dtypes.float16, 2))
  candidate = emit_q4k_exact_four_warp(1024, 4096)(
    p("out", (1024,), dtypes.float32, 0),
    p("words", (1024 * 16 * 36,), dtypes.uint32, 1),
    p("x", (4096,), dtypes.float16, 2))

  def src(u: UOp) -> str:
    return next(x.arg for x in to_program(u, ren).src if x.op is Ops.SOURCE)

  def strip(s: str) -> str:
    marker = 'extern "C" __global__'
    return s[s.index(marker):]

  return {"control": strip(src(control)), "candidate": strip(src(candidate))}


def _ncu_csv(which: str, name: str, binp: str) -> tuple[list[dict], str]:
  metrics = ",".join([
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
  ])
  cmd = [
    "sudo", "-n", NCU,
    "-k", name,
    "--launch-count", "1",
    "--launch-skip", "1",
    "--metrics", metrics,
    "--csv",
    binp, which, "1",
  ]
  cp = subprocess.run(cmd, capture_output=True, text=True)
  rows: list[dict] = []
  header_started = False
  reader = csv.reader(io.StringIO(cp.stdout))
  for cols in reader:
    if not cols:
      continue
    if not header_started:
      if cols and cols[0] == "ID":
        header_started = True
      continue
    if cols and cols[0] == "ID":
      continue
    if len(cols) >= 15:
      rows.append({"metric": cols[12], "unit": cols[13], "value": cols[14]})
  return rows, cp.stderr.strip()


HARNESS = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fffffff))
#endif
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }

__SRC_CONTROL__
__SRC_CANDIDATE__

#define X 4096

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(2); }
}

static double time_control(float* out, unsigned int* words, half* x, int passes) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++)
    q4k_g3_lanemap_gemv_1024_4096<<<1024, 32>>>(out, words, x);
  cudaEventRecord(e);
  check(cudaDeviceSynchronize(), "sync control");
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms * 1000.0 / passes;
}

static double time_candidate(float* out, unsigned int* words, half* x, int passes) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++)
    q4k_exact_four_warp_1024_4096<<<1024, 128>>>(out, words, x);
  cudaEventRecord(e);
  check(cudaDeviceSynchronize(), "sync candidate");
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms * 1000.0 / passes;
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "bracket";
  int passes = argc > 2 ? atoi(argv[2]) : 1000;
  int reps = argc > 3 ? atoi(argv[3]) : 7;

  float* out = nullptr; unsigned int* words = nullptr; half* x = nullptr;
  check(cudaMalloc(&out, 1024 * sizeof(float)), "out");
  check(cudaMalloc(&words, 589824 * sizeof(unsigned int)), "words");
  check(cudaMalloc(&x, X * sizeof(half)), "x");
  check(cudaMemset(out, 0, 1024 * sizeof(float)), "memset out");
  check(cudaMemset(words, 0, 589824 * sizeof(unsigned int)), "memset words");
  check(cudaMemset(x, 0, X * sizeof(half)), "memset x");

  q4k_g3_lanemap_gemv_1024_4096<<<1024, 32>>>(out, words, x);
  q4k_exact_four_warp_1024_4096<<<1024, 128>>>(out, words, x);
  check(cudaGetLastError(), "warmup launch");
  check(cudaDeviceSynchronize(), "warmup sync");

  if (!strcmp(mode, "bracket")) {
    printf("shape=1024x4096 control=1warp/row candidate=4warps/row passes=%d reps=%d\n", passes, reps);
    for (int r = 0; r < reps; r++) {
      double a = time_control(out, words, x, passes);
      double b = time_candidate(out, words, x, passes);
      double c = time_control(out, words, x, passes);
      printf("rep=%d control_a=%.4f candidate_b=%.4f control_c=%.4f\n", r, a, b, c);
    }
  } else if (!strcmp(mode, "control")) {
    double a = time_control(out, words, x, passes);
    printf("which=control passes=%d per_launch=%.4f\n", passes, a);
  } else if (!strcmp(mode, "candidate")) {
    double b = time_candidate(out, words, x, passes);
    printf("which=candidate passes=%d per_launch=%.4f\n", passes, b);
  } else {
    fprintf(stderr, "unknown mode %s\n", mode);
    return 3;
  }
  check(cudaFree(out), "free out"); check(cudaFree(words), "free words"); check(cudaFree(x), "free x");
  return 0;
}
"""


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--passes", type=int, default=1000)
  ap.add_argument("--reps", type=int, default=7)
  ap.add_argument("--ncu", action="store_true")
  ap.add_argument("--out", default="")
  args = ap.parse_args()

  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)
  srcs = _render(ren)
  cu = HARNESS.replace("__SRC_CONTROL__", srcs["control"]).replace(
    "__SRC_CANDIDATE__", srcs["candidate"])

  with tempfile.TemporaryDirectory(prefix="q4k_k_four_warp_bracket_") as td:
    cu_path = os.path.join(td, "bracket.cu")
    binp = os.path.join(td, "bracket")
    with open(cu_path, "w") as f:
      f.write(cu)
    env = dict(os.environ)
    env["PATH"] = f"{CUDA_BIN}:" + env.get("PATH", "")
    cp = subprocess.run(
      ["nvcc", "-arch=sm_120a", "-O3", "-std=c++17", "--ptxas-options=-v",
       cu_path, "-o", binp],
      capture_output=True, text=True, env=env)
    if cp.returncode != 0:
      print(cp.stderr[-6000:], file=sys.stderr)
      return 3

    r = subprocess.run([binp, "bracket", str(args.passes), str(args.reps)],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
      print(r.stderr[-4000:], file=sys.stderr)
      return 4

    out = {
      "schema": "tinygrad.q4k_k_four_warp_cuda_bracket.v1",
      "commit": subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=str(ROOT)).stdout.strip(),
      "method": "render production CUDA -> nvcc sm_120a -> cudaEvent control/candidate/control",
      "shape": {"rows": 1024, "k": 4096},
      "control": {"id": "q4k_g3_lanemap_gemv_1024_4096", "grid": 1024,
                  "block": 32, "warps_per_row": 1},
      "candidate": {"id": "q4k_exact_four_warp_1024_4096", "grid": 1024,
                    "block": 128, "warps_per_row": 4, "blocks_per_warp": 4},
      "passes": args.passes, "reps": args.reps,
      "timing": {"unit": "us_per_launch_cuda_event"},
      "raw_stdout": r.stdout.strip().splitlines(),
      "ptxas": cp.stderr.strip().splitlines(),
    }
    # Re-parse the printed rep lines into structured timing.
    import statistics
    import re
    a, b, c = [], [], []
    for line in r.stdout.splitlines():
      m = re.search(r"control_a=([0-9.]+) candidate_b=([0-9.]+) control_c=([0-9.]+)", line)
      if m:
        a.append(float(m.group(1)))
        b.append(float(m.group(2)))
        c.append(float(m.group(3)))
    if a and b and c:
      midpoint = (statistics.median(a) + statistics.median(c)) / 2
      cand = statistics.median(b)
      out["timing"] = {
        "unit": "us_per_launch_cuda_event",
        "control_a": a,
        "candidate_b": b,
        "control_c": c,
        "control_midpoint_median": midpoint,
        "candidate_median": cand,
        "delta": cand - midpoint,
        "ratio": cand / midpoint,
        "gate": "PASS" if cand <= 0.95 * midpoint else "FAIL",
      }

    if args.ncu:
      out["ncu"] = {}
      names = {
        "control": "q4k_g3_lanemap_gemv_1024_4096",
        "candidate": "q4k_exact_four_warp_1024_4096",
      }
      for which, name in names.items():
        rows, err = _ncu_csv(which, name, binp)
        out["ncu"][which] = {"name": name, "rows": rows}
        if err:
          out["ncu"][which]["stderr"] = err[-2000:]
          print(f"ncu {which} stderr tail:\n{err[-2000:]}", file=sys.stderr)

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
      with open(args.out, "w") as f:
        f.write(text + "\n")
    print(text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
