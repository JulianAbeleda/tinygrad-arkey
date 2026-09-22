#!/usr/bin/env python3
"""DRAM/occupancy bracket for the production K/V/gate_up anchor GEMVs.

Renders the exact production kernels through CUDARenderer (the same source the
NV route lowers), compiles them standalone with nvcc, and brackets each one
with ncu for DRAM bytes/throughput and occupancy.  This mirrors the 2026-08-14
FFN-down occupancy proof (`ffn_down_wall_harness.cu` + ncu), extended to the
three attention-side anchors the user is questioning.

It is measurement tooling only.  It never changes model numerics or production
routing, and it does not claim the nvcc-compiled SASS is byte-identical to the
NAK path: the geometry (grid/block) and the weight stream are identical, which
is the quantity the occupancy->DRAM causal chain depends on.

Usage (GPU serialized; ncu needs passwordless sudo):
  flock /tmp/gpu-bench.lock -c "PATH=/usr/local/cuda-13.2/bin:$PATH \
    PYTHONPATH=. .venv/bin/python extra/llm_research/decode/nv_anchor_geometry_dram.py"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel, q4k_g3_lanemap_gemv_w1w3_kernel
from tinygrad.llm.q6k_v_mmvq import emit_q6k_v_four_warp_fp16_direct
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp

CUDA_BIN = "/usr/local/cuda-13.2/bin"
NCU = "/usr/local/bin/ncu"

# (name, grid, block, weight_bytes) for each production anchor.
SPECS = {
  "K": {
    "name": "q4k_g3_lanemap_gemv_1024_4096", "grid": 1024, "block": 32,
    "weight_bytes": 1024 * 16 * 36 * 4,
  },
  "V": {
    "name": "q6k_v_four_warp_fp16_direct_1024_4096", "grid": 1024, "block": 128,
    "weight_bytes": 1024 * 16 * 105 * 2,
  },
  "gate_up": {
    "name": "q4k_g3_lanemap_gemv_w1w3fused16_12288_4096", "grid": 12288, "block": 32,
    "weight_bytes": 12288 * 16 * 36 * 4 * 2,
  },
}


def _render(ren: CUDARenderer) -> dict[str, str]:
  def p(name: str, shape: tuple[int, ...], dtype, slot: int) -> UOp:
    return UOp.placeholder(shape, dtype, slot)

  k = q4k_g3_lanemap_gemv_kernel(1024, 4096)(
    p("out", (1024,), dtypes.float32, 0),
    p("words", (1024 * 16 * 36,), dtypes.uint32, 1),
    p("x", (4096,), dtypes.float16, 2))
  v = emit_q6k_v_four_warp_fp16_direct()(
    p("out", (1024,), dtypes.float32, 0),
    p("halfs", (1024 * 16 * 105,), dtypes.uint16, 1),
    p("x", (4096,), dtypes.float16, 2))
  g = q4k_g3_lanemap_gemv_w1w3_kernel(12288, 4096, load_style="scalar", store_fp16=True)(
    p("out", (12288,), dtypes.float16, 0),
    p("gate_words", (12288 * 16 * 36,), dtypes.uint32, 1),
    p("up_words", (12288 * 16 * 36,), dtypes.uint32, 2),
    p("x", (4096,), dtypes.float16, 3))

  def src(u: UOp) -> str:
    return next(x.arg for x in to_program(u, ren).src if x.op is Ops.SOURCE)

  # The renderer preamble (INFINITY/NAN/tg_bitcast) is supplied once by HARNESS;
  # strip each kernel source down to its __global__ entry so nvcc sees one copy.
  def strip(s: str) -> str:
    marker = 'extern "C" __global__'
    return s[s.index(marker):]

  return {"K": strip(src(k)), "V": strip(src(v)), "gate_up": strip(src(g))}


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

__SRC_K__
__SRC_V__
__SRC_GATE_UP__

#define X 4096

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(2); }
}

int main(int argc, char** argv) {
  const char* which = argc > 1 ? argv[1] : "K";
  int passes = argc > 2 ? atoi(argv[2]) : 200;

  float* fout = nullptr; half* hout = nullptr; half* x = nullptr;
  unsigned int* kw = nullptr; unsigned int* gw = nullptr; unsigned int* uw = nullptr;
  unsigned short* vh = nullptr;

  check(cudaMalloc(&x, X * sizeof(half)), "x");
  check(cudaMemset(x, 0, X * sizeof(half)), "memset x");

  if (!strcmp(which, "K")) {
    check(cudaMalloc(&fout, 1024 * sizeof(float)), "fout");
    check(cudaMalloc(&kw, 589824 * sizeof(unsigned int)), "kw");
    check(cudaMemset(fout, 0, 1024 * sizeof(float)), "memset fout");
    check(cudaMemset(kw, 0, 589824 * sizeof(unsigned int)), "memset kw");
    q4k_g3_lanemap_gemv_1024_4096<<<1024, 32>>>(fout, kw, x);
  } else if (!strcmp(which, "V")) {
    check(cudaMalloc(&fout, 1024 * sizeof(float)), "fout");
    check(cudaMalloc(&vh, 1720320 * sizeof(unsigned short)), "vh");
    check(cudaMemset(fout, 0, 1024 * sizeof(float)), "memset fout");
    check(cudaMemset(vh, 0, 1720320 * sizeof(unsigned short)), "memset vh");
    q6k_v_four_warp_fp16_direct_1024_4096<<<1024, 128>>>(fout, vh, x);
  } else if (!strcmp(which, "gate_up")) {
    check(cudaMalloc(&hout, 12288 * sizeof(half)), "hout");
    check(cudaMalloc(&gw, 7077888 * sizeof(unsigned int)), "gw");
    check(cudaMalloc(&uw, 7077888 * sizeof(unsigned int)), "uw");
    check(cudaMemset(hout, 0, 12288 * sizeof(half)), "memset hout");
    check(cudaMemset(gw, 0, 7077888 * sizeof(unsigned int)), "memset gw");
    check(cudaMemset(uw, 0, 7077888 * sizeof(unsigned int)), "memset uw");
    q4k_g3_lanemap_gemv_w1w3fused16_12288_4096<<<12288, 32>>>(hout, gw, uw, x);
  } else {
    fprintf(stderr, "unknown kernel %s\n", which);
    return 3;
  }
  check(cudaGetLastError(), "launch warmup");
  check(cudaDeviceSynchronize(), "sync warmup");

  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++) {
    if (!strcmp(which, "K")) q4k_g3_lanemap_gemv_1024_4096<<<1024, 32>>>(fout, kw, x);
    else if (!strcmp(which, "V")) q6k_v_four_warp_fp16_direct_1024_4096<<<1024, 128>>>(fout, vh, x);
    else q4k_g3_lanemap_gemv_w1w3fused16_12288_4096<<<12288, 32>>>(hout, gw, uw, x);
  }
  cudaEventRecord(e);
  check(cudaDeviceSynchronize(), "sync loop");
  float ms = 0;
  cudaEventElapsedTime(&ms, s, e);
  printf("which=%s passes=%d total=%.3f ms per_launch=%.3f us\n", which, passes, ms, ms * 1000.0 / passes);
  return 0;
}
"""


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
      # The CSV section begins with the quoted column header; app stdout lines
      # (which=..., mode=...) appear before it and must be ignored.
      if cols and cols[0] == "ID":
        header_started = True
      continue
    if cols and cols[0] == "ID":
      continue
    if len(cols) >= 15:
      rows.append({"metric": cols[12], "unit": cols[13], "value": cols[14]})
  return rows, cp.stderr.strip()


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--kernels", default="K,V,gate_up")
  ap.add_argument("--passes", type=int, default=200)
  ap.add_argument("--skip-ncu", action="store_true")
  ap.add_argument("--out", default="")
  args = ap.parse_args()

  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)
  srcs = _render(ren)
  cu = (HARNESS.replace("__SRC_K__", srcs["K"])
               .replace("__SRC_V__", srcs["V"])
               .replace("__SRC_GATE_UP__", srcs["gate_up"]))

  with tempfile.TemporaryDirectory(prefix="nv_anchor_geom_") as td:
    cu_path = os.path.join(td, "anchors.cu")
    binp = os.path.join(td, "anchors")
    with open(cu_path, "w") as f:
      f.write(cu)
    env = dict(os.environ)
    env["PATH"] = f"{CUDA_BIN}:" + env.get("PATH", "")
    cp = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-std=c++17", "--ptxas-options=-v",
                         cu_path, "-o", binp], capture_output=True, text=True, env=env)
    if cp.returncode != 0:
      print(cp.stderr[-6000:], file=sys.stderr)
      return 3

    out: dict = {
      "schema": "tinygrad.nv_anchor_geometry_dram.v1",
      "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT)).stdout.strip(),
      "method": "render production CUDA source -> nvcc sm_120a standalone -> ncu isolated cold replay",
      "kernel_specs": SPECS,
      "wall": {},
      "ncu": {},
    }
    for which in [w for w in args.kernels.split(",") if w]:
      r = subprocess.run([binp, which, str(args.passes)], capture_output=True, text=True)
      if r.returncode != 0:
        print(f"{which}: {r.stderr[-2000:]}", file=sys.stderr)
        return 4
      print(r.stdout.strip())
      m = re.search(r"per_launch=([0-9.]+) us", r.stdout)
      out["wall"][which] = {
        "per_launch_us": float(m.group(1)) if m else None,
        "weight_bytes": SPECS[which]["weight_bytes"],
      }
      if not args.skip_ncu:
        rows, err = _ncu_csv(which, SPECS[which]["name"], binp)
        out["ncu"][which] = rows
        if err:
          print(f"ncu {which} stderr tail:\n{err[-2000:]}", file=sys.stderr)

    if args.out:
      with open(args.out, "w") as f:
        f.write(json.dumps(out, indent=2))
      print(f"wrote {args.out}")
    else:
      print(json.dumps(out, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
