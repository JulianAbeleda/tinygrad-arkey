#!/usr/bin/env python3
"""Dump the ordinary NV reduce+epilogue CUDA source and UOp trees for a
(rows,128) fp32 RMSNorm (the q/k norm shape) so the fp32-route scope can pin
the exact association the fused body must reproduce bitwise.

Prints each captured PROGRAM's rendered CUDA source plus the full sink UOp
tree with symbolic RANGE variable names and complete ALU/INDEX chains, so the
accumulator x-load index chains are visible."""
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
  from tinygrad.uop.ops import AxisType, Ops
  range_names: dict[int, str] = {}
  counter = [0]
  def rname(u) -> str:
    if id(u) not in range_names:
      range_names[id(u)] = f"r{counter[0]}"
      counter[0] += 1
    return range_names[id(u)]
  def expr(u) -> str:
    if u.op is Ops.CONST: return str(u.arg)
    if u.op is Ops.SPECIAL: return f"{u.arg[0]}({u.arg[1]})"
    if u.op is Ops.RANGE:
      end = u.src[0].arg if u.src and u.src[0].op is Ops.CONST else "?"
      return f"{rname(u)}:RANGE[{end},{u.arg[1].name}]"
    if u.op is Ops.END: return f"END[{','.join(expr(s) for s in u.src)}]"
    if u.op is Ops.AFTER: return f"AFTER[{','.join(expr(s) for s in u.src)}]"
    if u.op is Ops.BARRIER: return "BARRIER"
    if u.op is Ops.INDEX: return f"{expr(u.src[0])}[{expr(u.src[1])}]" if len(u.src) > 1 else f"{expr(u.src[0])}[.]"
    if u.op is Ops.LOAD:
      idx = expr(u.src[1]) if len(u.src) > 1 else ""
      return f"LOAD({expr(u.src[0])}[{idx}])"
    if u.op in (Ops.ADD, Ops.MUL, Ops.SUB, Ops.FDIV, Ops.MAX, Ops.CMPLT, Ops.CMPEQ, Ops.CDIV):
      return f"({expr(u.src[0])} {u.op.name} {expr(u.src[1])})"
    if u.op is Ops.SHL: return f"({expr(u.src[0])} SHL {expr(u.src[1])})"
    if u.op is Ops.CAST: return f"CAST_{u.dtype}({expr(u.src[0])})"
    if u.op in (Ops.RECIPROCAL, Ops.SQRT, Ops.EXP2, Ops.LOG2): return f"{u.op.name}({expr(u.src[0])})"
    if u.op is Ops.PARAM: return f"p{u.arg.slot}{u._shape}"
    if u.op is Ops.BUFFER: return f"b{u.arg.slot}{u._shape}"
    if u.op is Ops.SHRINK: return f"SHRINK({expr(u.src[0])})"
    if u.op is Ops.STACK: return f"STACK({','.join(expr(s) for s in u.src)})"
    if u.op is Ops.GEP: return f"GEP({expr(u.src[0])},{u.arg})"
    return f"{u.op.name}({','.join(expr(s) for s in u.src)})" if len(u.src) else u.op.name
  for name, ast in bodies.items():
    print(f"--- {name}:")
    if len(ast.src) > 3 and ast.src[3].op is Ops.SOURCE:
      print("==== CUDA SOURCE ====")
      print(ast.src[3].arg)
      print("==== UOP TREE ====")
    sink = ast.src[0]
    print(f"    sink arg: {sink.arg}")
    for u in sink.toposort():
      if u.op.name in ("RANGE", "RANGE2"):
        print(f"      RANGE {u.arg} -> {expr(u)}")
      elif u.op.name == "STORE":
        print(f"      STORE {expr(u)}")
      elif u.op.name == "BARRIER":
        print(f"      BARRIER")
      elif u.op.name in ("SPECIAL", "PARAM", "BUFFER"):
        print(f"      {u.op.name} {u.arg} shape={u._shape}")
      elif u.op.name in ("END", "AFTER", "GROUP"):
        print(f"      {u.op.name} {expr(u)}")


if __name__ == "__main__":
  dump(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
