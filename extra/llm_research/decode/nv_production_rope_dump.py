#!/usr/bin/env python3
"""Render and disassemble the production Q/K apply_rope arithmetic (tooling only).

Builds the exact flattened apply_rope expression used by the installed decode
graph (low = x1*cos - x2*sin, high = x2*cos + x1*sin with freqs laid out as
cos[:half], sin[half:]) and dumps the CUDARenderer source plus the NVRTC PTX for
the production compile path (use_nvcc=False, --gpu-architecture=sm_120). This
pins the FP contraction the fused norm+rope candidate must reproduce.
"""
from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp


def rope_kernel(rows: int, dim: int):
  half = dim // 2

  def kernel(out: UOp, x: UOp, freqs: UOp) -> UOp:
    idx = UOp.range(rows * dim, 0, AxisType.GLOBAL)
    e = idx % dim
    low = e < half
    rot = low.where(e, e - half)
    k1 = x[idx - e + rot].cast(dtypes.float32)
    k2 = x[idx - e + rot + half].cast(dtypes.float32)
    cos = freqs[rot].cast(dtypes.float32)
    sin = freqs[half + rot].cast(dtypes.float32)
    val = low.where(k1 * cos - k2 * sin, k2 * cos + k1 * sin)
    return out[idx].store(val).end(idx).sink(
      arg=KernelInfo(name=f"apply_rope_{rows}_{dim}", opts_to_apply=()))
  return kernel


def main() -> int:
  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)
  for rows, dim in ((32, 128), (8, 128)):
    kernel = rope_kernel(rows, dim)
    u = kernel(
      UOp.placeholder((rows * dim,), dtypes.float32, 0),
      UOp.placeholder((rows * dim,), dtypes.float32, 1),
      UOp.placeholder((dim,), dtypes.float32, 2))
    program = to_program(u, ren)
    src = next(x.arg for x in program.src if x.op is Ops.SOURCE)
    print(f"===== rows={rows} launch_dims={program.arg.launch_dims({})} =====")
    print(src)
    cubin = ren.compiler.compile(src)
    pathlib.Path(f"/tmp/prod_rope_{rows}.cubin").write_bytes(cubin)
    print(f"----- cubin rows={rows} bytes={len(cubin)} -----")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
