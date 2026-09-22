#!/usr/bin/env python3
"""FINAL FORM: pin the exact ordinary fp32 q/k reduce association (Wave 1, A).

Captures the ordinary reduce PROGRAM for a (rows,128) fp32 RMSNorm, prints the
accumulator x-load index chains (the ADD-chain srcs inside the reduce), and
verifies candidate numpy models BITWISE against the captured per-row scale.
This is the association the row-mode emitter body must reproduce.

Derived and pinned 2026-08-10 (verified bitwise, 0 bad rows):
  * NV rows=32 (q), r_2_8_4_4_16 (global=(2,1,1), local=(8,4,1)): 8 threads/row
    x 16 STRIDE-8 elements each (x[r*128 + t + 8*R]), per-thread serial chain
    FMA-contracted, then a serial 8-chain over threads in lidx0 order.
  * NV rows=8 (k), r_8_16_8 (global=(8,1,1), local=(16,1,1)): 16 threads x 8
    CONTIGUOUS serial, FMA, serial 16-thread chain.
  * CPU (the bitwise tripwire device), r_32_32_4 / r_8_32_4: one thread per
    row, plain dim-contiguous serial FMA chain -- which is exactly the row-mode
    emitter body (plain contiguous per warp), so the CPU tripwire is bitwise.

Run under the shared-GPU lock with DEV=NV, or CPU-only with an argument:
  flock -w 90 /tmp/gpu-bench.lock env DEV=NV PYTHONPATH=/home/ubuntu/tinygrad-arkey python3 scratchpad/nv_fp32_qk_association_probe.py
  DEV=CPU PYTHONPATH=/home/ubuntu/tinygrad-arkey python3 scratchpad/nv_fp32_qk_association_probe.py CPU
"""
from __future__ import annotations

import numpy as np, sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad import Tensor, nn
from tinygrad.dtype import dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import Ops
import tinygrad.engine.realize as realize

def _fma(a, b, c):
  """Correctly-rounded fp32 a*b+c (clang contracts ``acc + v*v`` into fma)."""
  return np.float32(np.longdouble(a) * np.longdouble(b) + np.longdouble(c))

def _serial_sum(seg, use_fma):
  acc = np.float32(0.0)
  for v in seg:
    vv = np.float32(v)
    acc = _fma(vv, vv, acc) if use_fma else np.float32(np.float32(vv * vv) + acc)
  return acc

def _scale_of(partials, dim, eps):
  acc = partials[0]
  for p in partials[1:]: acc = np.float32(acc + p)
  return np.float32(1.0 / np.sqrt(np.float32(np.float32(acc / dim) + eps)))

def _capture_scale(dev, rows, dim, eps, x_np):
  """Run the ordinary RMSNorm reduce on ``dev`` and capture its per-row scale
  buffer by re-running the captured PROGRAM on the same x bytes."""
  x = Tensor(x_np).realize()
  n = nn.RMSNorm(dim, eps=eps)
  n.weight = Tensor(np.ones(dim, dtype=np.float32)).realize()
  out = n(x)
  captured = {}
  orig = realize.get_runtime
  def patched(device, ast, cache=True):
    if ast.op is Ops.PROGRAM and ast.arg.function_name.startswith("r_"):
      captured["reduce"] = ast
    return orig(device, ast, cache)
  realize.get_runtime = patched
  try:
    out.realize()
  finally:
    realize.get_runtime = orig
  ast = captured["reduce"]
  in_buf = Buffer(dev, rows * dim, dtypes.float32).allocate()
  in_buf.copyin(memoryview(x_np.reshape(-1).copy(order="C")))
  out_buf = Buffer(dev, rows, dtypes.float32).allocate()
  rt = realize.get_runtime(dev, ast)
  if dev == "NV":
    rt(out_buf._buf, in_buf._buf, global_size=ast.arg.global_size, local_size=ast.arg.local_size, vals={}, wait=True)
  else:
    rt(out_buf._buf, in_buf._buf, vals={}, wait=True)
  scale_np = np.empty(rows, dtype=np.float32)
  out_buf.copyout(memoryview(scale_np.data))
  return ast, np.array(scale_np)

def _print_reduce_chains(ast):
  """Print the accumulator x-load index chains (ADD-chain srcs) inside the
  ordinary reduce, with symbolic RANGE names, so the mapping is inspectable."""
  range_names: dict[int, str] = {}
  counter = [0]
  def rname(u) -> str:
    if id(u) not in range_names:
      range_names[id(u)] = f"r{counter[0]}"; counter[0] += 1
    return range_names[id(u)]
  def expr(u, depth=0) -> str:
    if depth > 8: return "..."
    if u.op is Ops.CONST: return str(u.arg)
    if u.op is Ops.RANGE: return f"{rname(u)}:[{u.src[0].arg},{u.arg[1].name}]"
    if u.op is Ops.SPECIAL: return f"id{u.arg[0]}"
    if u.op is Ops.INDEX and len(u.src) > 1: return f"{expr(u.src[0], depth+1)}[{expr(u.src[1], depth+1)}]"
    if u.op in (Ops.ADD, Ops.MUL, Ops.SUB): return f"({expr(u.src[0], depth+1)} {u.op.name} {expr(u.src[1], depth+1)})"
    if u.op is Ops.SHL: return f"({expr(u.src[0], depth+1)} SHL {expr(u.src[1], depth+1)})"
    if u.op is Ops.LOAD and len(u.src) > 1: return f"x[{expr(u.src[1], depth+1)}]"
    if u.op is Ops.SHRINK: return f"{expr(u.src[0], depth+1)}<{u.arg}>"
    if u.op is Ops.CAST: return f"CAST_{u.dtype}({expr(u.src[0], depth+1)})"
    if u.op is Ops.PARAM: return f"p{u.arg.slot}{u._shape}"
    return f"{u.op.name}({','.join(expr(s, depth+1) for s in u.src)})" if len(u.src) else u.op.name
  sink = ast.src[0]
  topo_list = list(sink.toposort())
  print(f"    reduce kernel: {ast.arg.function_name} global={ast.arg.global_size} local={ast.arg.local_size}")
  for u in topo_list:
    if u.op is Ops.STORE and u.src[1].op in (Ops.ADD, Ops.MAX):
      # Only the accumulator chains that read the input, not the smem/STACK
      # cross-thread combine stores.
      reads = [s for s in u.src[1].backward_slice if s.op is Ops.LOAD and s.src and s.src[0].op in (Ops.PARAM, Ops.SHRINK)]
      if reads:
        r = max(reads, key=topo_list.index)
        if len(r.src) > 1:
          idx = r.src[1]
        elif r.src[0].op is Ops.SHRINK and len(r.src[0].src) > 1:
          idx = r.src[0].src[1]
        else:
          idx = r
        print(f"    accumulator x-load idx: {expr(idx)}")

def _verify(dev, rows, dim, eps, x_np, scale_np):
  """Reconstruct the scale with the pinned models and report bitwise rows."""
  models = {}
  if dev == "NV" and rows == 32:
    for use_fma in (False, True):
      tag = "fma" if use_fma else "nofma"
      sc = np.zeros(rows, dtype=np.float32)
      for r in range(rows):
        partials = [_serial_sum(x_np[r, t + 8 * np.arange(16)], use_fma) for t in range(8)]
        sc[r] = _scale_of(partials, dim, eps)
      models[f"NV stride8 8x16 {tag}"] = sc
  if dev == "NV" and rows == 8:
    for use_fma in (False, True):
      tag = "fma" if use_fma else "nofma"
      sc = np.zeros(rows, dtype=np.float32)
      for r in range(rows):
        partials = [_serial_sum(x_np[r, t * 8:(t + 1) * 8], use_fma) for t in range(16)]
        sc[r] = _scale_of(partials, dim, eps)
      models[f"NV contig16 16x8 {tag}"] = sc
  for use_fma in (False, True):
    tag = "fma" if use_fma else "nofma"
    sc = np.zeros(rows, dtype=np.float32)
    for r in range(rows):
      sc[r] = _scale_of([_serial_sum(x_np[r], use_fma)], dim, eps)
    models[f"plain{rows}x{dim} {tag}"] = sc
  bad_rows = {k: np.argwhere(v != scale_np).reshape(-1).tolist() for k, v in models.items()}
  return models, bad_rows

def _run(dev, rows, dim, eps, seed):
  rng = np.random.default_rng(20260810)
  x_np = rng.normal(0, 0.2, (rows, dim)).astype(np.float32)
  print(f"=== rows={rows} dim={dim} dev={dev}")
  ast, scale_np = _capture_scale(dev, rows, dim, eps, x_np)
  _print_reduce_chains(ast)
  models, bad_rows = _verify(dev, rows, dim, eps, x_np, scale_np)
  for name, bad in bad_rows.items():
    print(f"    model {name}: bad rows {bad} ({len(bad)}/{rows})")
  exact = [k for k, v in bad_rows.items() if not v]
  print(f"    bitwise-exact models: {exact}")
  if not exact:
    raise SystemExit(f"FAIL: no candidate model reproduces the ordinary {dev} {rows}x{dim} scale bitwise")

def main() -> None:
  dev = sys.argv[1] if len(sys.argv) > 1 else "NV"
  rows, dim, eps = 32, 128, 1e-6
  if len(sys.argv) > 2:
    # One shape per process: the realize pipeline caches schedules/UOp rewrites
    # process-wide, so a second (rows,dim) capture in one process would see the
    # first shape's cached programs and never hit get_runtime again.
    _run(dev, int(sys.argv[2]), 128, eps, 20260810 + int(sys.argv[2]))
    return
  import subprocess
  for probe_rows in (32, 8):
    subprocess.run([sys.executable, __file__, dev, str(probe_rows)], check=True)
  print("association probe: PASS (pinned bitwise for q rows=32 and k rows=8)")


if __name__ == "__main__":
  main()
