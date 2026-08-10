#!/usr/bin/env python3
"""Dump the ordinary NV reduce+epilogue CUDA source for a (rows,128) fp32
RMSNorm (the q/k norm shape) so the fp32-route scope can pin the exact
association the fused body must reproduce bitwise."""
from __future__ import annotations

import numpy as np, sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad import Tensor, dtypes, nn, Device


def dump(rows: int) -> None:
  dim = 128
  rng = np.random.default_rng(20260810)
  x = Tensor(rng.normal(0, 0.2, (rows, dim)).astype(np.float32)).realize()
  w = Tensor(rng.normal(1, 0.05, (dim,)).astype(np.float32)).realize()
  n = nn.RMSNorm(dim, eps=1e-6)
  n.weight = w
  out = n(x)
  bodies: dict[str, object] = {}
  import tinygrad.engine.realize as realize
  orig = realize.get_runtime
  def patched(device, ast, cache=True):
    if ast.op is not __import__("tinygrad.uop.ops", fromlist=["Ops"]).Ops.PROGRAM: return orig(device, ast, cache)
    name = ast.arg.function_name
    if name.startswith(("r_", "E_")): bodies.setdefault(name, ast)
    return orig(device, ast, cache)
  realize.get_runtime = patched
  try:
    out.realize()
  finally:
    realize.get_runtime = orig
  print(f"=== rows={rows}: programs {sorted(bodies)}")
  for name, ast in bodies.items():
    print(f"--- {name}:")
    sink = ast.src[0]
    print(f"    sink arg: {sink.arg}")
    from tinygrad.uop.ops import AxisType, Ops

    def expr(u) -> str:
      if u.op is Ops.CONST: return str(u.arg)
      if u.op is Ops.SPECIAL: return f"{u.arg[0]}({u.arg[1]})"
      if u.op is Ops.MUL: return f"({expr(u.src[0])}*{expr(u.src[1])})"
      if u.op is Ops.ADD: return f"({expr(u.src[0])}+{expr(u.src[1])})"
      if u.op is Ops.SUB: return f"({expr(u.src[0])}-{expr(u.src[1])})"
      if u.op is Ops.INDEX: return f"{u.src[0].arg}[{expr(u.src[1])}]" if u.src[0].op in (Ops.PARAM, Ops.BUFFER) else f"idx({expr(u.src[0])},{expr(u.src[1])})"
      if u.op is Ops.RANGE: return "RANGE" if len(u.src) == 0 else f"RANGE({','.join(expr(s) for s in u.src)})"
      if u.op is Ops.BUFFER: return f"BUF({u.arg})"
      if u.op is Ops.PARAM: return f"PARAM{u.arg}"
      if u.op is Ops.STORE: return f"STORE({expr(u.src[0])} <- {u.src[1].op.name})"
      return u.op.name

    for u in sink.toposort():
      if u.op.name in ("RANGE", "RANGE2"):
        print(f"      RANGE {u.arg} src={[expr(s) for s in u.src]}")
      elif u.op.name == "STORE":
        print(f"      STORE {expr(u)}")
      elif u.op.name in ("BARRIER", "SPECIAL", "PARAM", "BUFFER"):
        print(f"      {u.op.name} {u.arg} shape={u._shape}")


if __name__ == "__main__":
  dump(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
