#!/usr/bin/env python3
"""Decisive NVRTC bit-exactness gate for the fused Q/K norm+rope candidate.

The earlier microgate compiled its harness with nvcc, whose PTXAS schedules FP
contraction differently from the NVRTC path production actually uses.  This
probe removes that confound entirely: it renders the installed norm kernel,
the installed apply_rope kernel, and the fused norm+rope candidate, compiles
every one of them with NVRTC (``CUDARenderer(..., use_nvcc=False)``, the same
compiler production uses), loads the cubins through the real NV runtime, and
compares the production chain (norm -> apply_rope) against the single fused
kernel bit-for-bit on device.

It changes no production model/renderer/scheduler/runtime/route file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import Device, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.late.reduce_output import ReduceOutputSpec, emit_reduce_output
from tinygrad.device import BufferSpec
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.uop.ops import Ops, UOp

from extra.llm_research.decode.nv_production_rope_dump import rope_kernel
from extra.llm_research.decode.nv_qk_norm_rope_fuse_microgate import emit_reduce_output_rope

EPS = 1e-6


def _src(u: UOp, ren: CUDARenderer) -> str:
  return next(x.arg for x in to_program(u, ren).src if x.op is Ops.SOURCE)


def _alloc(dev, size: int):
  return dev.allocator._alloc(size, BufferSpec())


def _copyin(dev, buf, data: bytes):
  dev.allocator._copyin(buf, memoryview(data))


def _copyout(dev, buf) -> bytes:
  host = memoryview(bytearray(buf.size))
  dev.allocator._copyout(host, buf)
  return bytes(host)


def _float32_bytes(vals) -> bytes:
  return np.ascontiguousarray(np.asarray(vals, dtype=np.float32)).tobytes()


def _half_bytes(vals) -> bytes:
  return np.ascontiguousarray(np.asarray(vals, dtype=np.float16)).tobytes()


def _launch_dims(u: UOp, ren: CUDARenderer):
  g, l = to_program(u, ren).arg.launch_dims({})
  return tuple(int(x) for x in g), tuple(int(x) for x in l)


def build(ren: CUDARenderer) -> dict:
  def p(name, shape, dtype, slot):
    return UOp.placeholder(shape, dtype, slot)

  out = {}
  for rows in (32, 8):
    spec = ReduceOutputSpec(rows=rows, dim=128, eps=EPS, out_dtype=dtypes.float32,
                            affine=True, recipe="sumsq_rsqrt_affine", reduce_op=Ops.ADD,
                            warps=rows, lanes=32, per_lane=4)
    nk = emit_reduce_output(spec, dtypes.float32, dtypes.float16)
    nu = nk(p("out", (rows * 128,), dtypes.float32, 0),
            p("x", (rows * 128,), dtypes.float32, 1),
            p("w", (128,), dtypes.float16, 2))
    rk = rope_kernel(rows, 128)
    ru = rk(p("out", (rows * 128,), dtypes.float32, 0),
            p("x", (rows * 128,), dtypes.float32, 1),
            p("f", (128,), dtypes.float32, 2))
    ck = emit_reduce_output_rope(spec, dtypes.float32, dtypes.float16)
    cu = ck(p("out", (rows * 128,), dtypes.float32, 0),
            p("x", (rows * 128,), dtypes.float32, 1),
            p("w", (128,), dtypes.float16, 2),
            p("f", (128,), dtypes.float32, 3))
    out[rows] = {
      "norm": {"src": _src(nu, ren), "name": f"reduce_output_rmsnorm_{rows}_128",
               "grid": _launch_dims(nu, ren)[0], "block": _launch_dims(nu, ren)[1]},
      "rope": {"src": _src(ru, ren), "name": f"apply_rope_{rows}_128",
               "grid": _launch_dims(ru, ren)[0], "block": _launch_dims(ru, ren)[1]},
      "fused": {"src": _src(cu, ren), "name": f"reduce_output_rmsnorm_rope_{rows}_128",
                "grid": _launch_dims(cu, ren)[0], "block": _launch_dims(cu, ren)[1]},
    }
  return out


def run_rows(dev: NVProgram, ren: CUDARenderer, rows: int, manifest: dict, identity: bool) -> dict:
  n = rows * 128
  # Deterministic inputs matching the microgate (plus an identity-rotation arm
  # where cos=1, sin=0 isolates the norm half of the fusion).
  xs = [0.1 + 0.001 * (i % 17) for i in range(n)]
  ws = [1.0 + 0.001 * (i % 5) for i in range(128)]
  if identity:
    fr = [1.0] * 64 + [0.0] * 64
  else:
    fr = [0.9 + 0.001 * i for i in range(64)] + [0.4 - 0.001 * (i - 64) for i in range(64, 128)]

  x = _alloc(dev, n * 4)
  w = _alloc(dev, 128 * 2)
  f = _alloc(dev, 128 * 4)
  ctrl = _alloc(dev, n * 4)
  roped = _alloc(dev, n * 4)
  cand = _alloc(dev, n * 4)
  _copyin(dev, x, _float32_bytes(xs))
  _copyin(dev, w, _half_bytes(ws))
  _copyin(dev, f, _float32_bytes(fr))

  norm = NVProgram(dev, manifest["norm"]["name"], ren.compiler.compile(manifest["norm"]["src"]))
  rope = NVProgram(dev, manifest["rope"]["name"], ren.compiler.compile(manifest["rope"]["src"]))
  fused = NVProgram(dev, manifest["fused"]["name"], ren.compiler.compile(manifest["fused"]["src"]))

  norm(ctrl, x, w, global_size=manifest["norm"]["grid"], local_size=manifest["norm"]["block"], wait=True)
  # Production apply_rope writes a fresh buffer, never in-place: an in-place
  # launch reads the partner element that a sibling thread may already have
  # overwritten (data race).  Reproduce the production chain faithfully.
  rope(roped, ctrl, f, global_size=manifest["rope"]["grid"], local_size=manifest["rope"]["block"], wait=True)
  fused(cand, x, w, f, global_size=manifest["fused"]["grid"], local_size=manifest["fused"]["block"], wait=True)

  ctrl_b = _copyout(dev, roped)
  cand_b = _copyout(dev, cand)
  ctrl_a = np.frombuffer(ctrl_b, dtype=np.float32)
  cand_a = np.frombuffer(cand_b, dtype=np.float32)
  diff = ctrl_a.view(np.uint32) != cand_a.view(np.uint32)
  max_abs = float(np.max(np.abs(ctrl_a.astype(np.float64) - cand_a.astype(np.float64)))) if n else 0.0
  return {
    "rows": rows,
    "identity": identity,
    "max_abs_diff": max_abs,
    "mismatch_count": int(diff.sum()),
    "bit_exact": bool(diff.sum() == 0),
    "ctrl_sha256": hashlib.sha256(ctrl_b).hexdigest(),
    "cand_sha256": hashlib.sha256(cand_b).hexdigest(),
  }

def timed_rows(dev, ren, rows, manifest, reps=20):
  n = rows * 128
  x = _alloc(dev, n*4); w = _alloc(dev, 128*2); f = _alloc(dev, 128*4)
  out = _alloc(dev, n*4); tmp = _alloc(dev, n*4)
  _copyin(dev, x, _float32_bytes([0.1 + 0.001*(i%17) for i in range(n)]))
  _copyin(dev, w, _half_bytes([1.0 + 0.001*(i%5) for i in range(128)]))
  _copyin(dev, f, _float32_bytes([0.9 + 0.001*i for i in range(64)] + [0.4 - 0.001*i for i in range(64)]))
  norm = NVProgram(dev, manifest["norm"]["name"], ren.compiler.compile(manifest["norm"]["src"]))
  rope = NVProgram(dev, manifest["rope"]["name"], ren.compiler.compile(manifest["rope"]["src"]))
  fused = NVProgram(dev, manifest["fused"]["name"], ren.compiler.compile(manifest["fused"]["src"]))
  def split():
    norm(tmp,x,w,global_size=manifest["norm"]["grid"],local_size=manifest["norm"]["block"])
    rope(out,tmp,f,global_size=manifest["rope"]["grid"],local_size=manifest["rope"]["block"])
  def one(): fused(out,x,w,f,global_size=manifest["fused"]["grid"],local_size=manifest["fused"]["block"])
  for _ in range(10): split(); one()
  dev.synchronize(); t0=time.perf_counter_ns()
  for _ in range(reps): split()
  dev.synchronize(); split_us=(time.perf_counter_ns()-t0)/reps/1000
  t0=time.perf_counter_ns()
  for _ in range(reps): one()
  dev.synchronize(); fused_us=(time.perf_counter_ns()-t0)/reps/1000
  return {"rows":rows,"reps":reps,"split_us":split_us,"fused_us":fused_us,"delta_us":split_us-fused_us}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out-json", type=pathlib.Path, required=True)
  args = ap.parse_args()

  dev = Device["NV"]
  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)
  manifest = build(ren)

  rows_out = []
  for rows in (32, 8):
    for identity in (False, True):
      rows_out.append(run_rows(dev, ren, rows, manifest[rows], identity))

  result = {
    "schema": "tinygrad.nv_qk_norm_rope_nvrtc_bit_exact.v1",
    "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "method": "render + NVRTC-compile installed norm/apply_rope/fused; run on real NV device; bitwise compare",
    "eps": EPS,
    "results": rows_out,
    "verdict": {f"{r['rows']}_{'identity' if r['identity'] else 'real'}_bit_exact": r["bit_exact"] for r in rows_out},
  }
  args.out_json.parent.mkdir(parents=True, exist_ok=True)
  args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result["verdict"], indent=2))
  for r in rows_out:
    print(f"rows={r['rows']} identity={r['identity']} max_abs_diff={r['max_abs_diff']:.9g} "
          f"mismatch={r['mismatch_count']} bit_exact={r['bit_exact']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
