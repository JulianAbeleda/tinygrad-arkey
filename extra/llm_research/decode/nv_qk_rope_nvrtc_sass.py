#!/usr/bin/env python3
"""Dump NVRTC SASS for the production rope and the fused norm+rope candidate (tooling only)."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.late.reduce_output import ReduceOutputSpec
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp

from extra.llm_research.decode.nv_qk_norm_rope_fuse_microgate import emit_reduce_output_rope
from extra.llm_research.decode.nv_production_rope_dump import rope_kernel


def source_of(program) -> str:
  return next(x.arg for x in program.src if x.op is Ops.SOURCE)


def main() -> int:
  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)
  for rows in (32, 8):
    # production rope
    rk = rope_kernel(rows, 128)
    ru = rk(UOp.placeholder((rows * 128,), dtypes.float32, 0),
             UOp.placeholder((rows * 128,), dtypes.float32, 1),
             UOp.placeholder((128,), dtypes.float32, 2))
    rsrc = source_of(to_program(ru, ren))
    # fused candidate
    spec = ReduceOutputSpec(rows=rows, dim=128, eps=1e-6, out_dtype=dtypes.float32,
                            affine=True, recipe="sumsq_rsqrt_affine", reduce_op=Ops.ADD,
                            warps=rows, lanes=32, per_lane=4)
    ck = emit_reduce_output_rope(spec, dtypes.float32, dtypes.float16)
    cu = ck(UOp.placeholder((rows * 128,), dtypes.float32, 0),
            UOp.placeholder((rows * 128,), dtypes.float32, 1),
            UOp.placeholder((128,), dtypes.float16, 2),
            UOp.placeholder((128,), dtypes.float32, 3))
    csrc = source_of(to_program(cu, ren))
    for tag, src in (("rope", rsrc), ("fused", csrc)):
      cubin = ren.compiler.compile(src)
      path = f"/tmp/nvrtc_{tag}_{rows}.cubin"
      pathlib.Path(path).write_bytes(cubin)
      print(f"rows={rows} {tag}: {path} ({len(cubin)} bytes)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
