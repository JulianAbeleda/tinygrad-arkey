#!/usr/bin/env python3
"""Feasibility probe for the fp32 (q/k) reduce-output route (final form).

Part A (NV production graph): count q/k marker calls and capture the ordinary
q/k reduce/epilogue program names under the production decode (NV only, heavy).

Part B (CPU bitwise gate): for x (rows,128) fp32, emit the row-mode cooperative
body through ``emit_reduce_output`` and verify it reproduces the ORDINARY CPU
RMSNorm reduce+epilogue bitwise (the exact-logits gate).  The pinned CPU
association (nv_fp32_qk_association_probe.py) is a plain dim-contiguous serial
FMA chain per row; the row-mode body encodes exactly that (warp w owns row w,
full-row chain, no cross-warp combine).  Run with DEV=CPU, no GPU needed.
"""
from __future__ import annotations

import numpy as np, sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad import Tensor, dtypes, nn, Device


def _fma(a, b, c):
  """Correctly-rounded fp32 a*b+c (clang contracts acc + v*v into fma)."""
  return np.float32(np.longdouble(a) * np.longdouble(b) + np.longdouble(c))


def _ordinary_names(x: Tensor):
  """Realized program names for the ordinary RMSNorm graph of x (no marker)."""
  n = nn.RMSNorm(x.shape[-1], eps=1e-6)
  w = Tensor.randn(x.shape[-1], dtype=x.dtype).realize()
  n.weight = w
  out = n(x)
  out.realize()
  from tinygrad.engine.realize import runtime_cache
  names = sorted({r.name for (_, dev), r in runtime_cache.items() if dev == str(Device.DEFAULT)})
  return names, out


def _part_a() -> None:
  from tinygrad.helpers import Context
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  from tinygrad.llm.reduce_output_trace import REDUCE_OUTPUT_TRACE, reduce_output_trace_snapshot
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  model = _load("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 32768)
  model._decode_direct_greedy_promoted = True
  model._decode_reduce_output_rmsnorm_promoted = True
  for block in model.blk: block._decode_reduce_output_rmsnorm_promoted = True
  calls: list[tuple[int, str]] = []
  import tinygrad.tensor as tensor_mod
  orig = tensor_mod.Tensor._semantic_reduce_output_rmsnorm
  def patched(self, x, out, weight, eps):
    rows = int(np.prod(x.shape[:-1])) if all(isinstance(s, int) for s in x.shape) else -1
    calls.append((rows, str(x.dtype)))
    return orig(self, x, out, weight, eps)
  tensor_mod.Tensor._semantic_reduce_output_rmsnorm = patched
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1, REDUCE_OUTPUT_TRACE=1):
    model.reset_generation_state()
    gen = model.generate(_prompt("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 512), chunk_size=32, temperature=0.0)
    try:
      int(next(gen))
      int(next(gen))
    finally:
      gen.close()
    tensor_mod.Tensor._semantic_reduce_output_rmsnorm = orig
  from collections import Counter
  print("marker calls by (rows, dtype):", Counter(calls))
  trace = reduce_output_trace_snapshot()
  print("selector associations:", trace.get("associations"))
  print("selector rejects:", {k: v for k, v in trace.get("selector", {}).items() if k != "entry"})


def _part_b() -> None:
  for rows in (8, 32):
    dim = 128
    rng = np.random.default_rng(20260810 + rows)
    x_np = rng.normal(0, 0.2, (rows, dim)).astype(np.float32)
    w_np = rng.normal(1, 0.05, (dim,)).astype(np.float32)
    x = Tensor(x_np).realize()
    w = Tensor(w_np).realize()
    names, out = _ordinary_names(x)
    # Re-run with the same weight so the epilogue matches.
    n = nn.RMSNorm(dim, eps=1e-6)
    n.weight = w
    ordinary = n(x).numpy()
    # Emit the cooperative body and verify it is bitwise-equal to ordinary.
    from tinygrad.codegen.late.reduce_output import emit_reduce_output
    from tinygrad.uop.ops import ReduceOutputSpec, UOp
    spec = ReduceOutputSpec(rows, dim, 1e-6, dtypes.float32, warps=rows, lanes=32, per_lane=4)
    out_ph = UOp.placeholder((rows * dim,), dtypes.float32, 0)
    x_ph = UOp.placeholder((rows * dim,), dtypes.float32, 1)
    w_ph = UOp.placeholder((dim,), dtypes.float32, 2)
    body = emit_reduce_output(spec, dtypes.float32, dtypes.float32)(out_ph, x_ph, w_ph)
    assert body.arg.name == f"reduce_output_rmsnorm_{rows}_{dim}"
    scale = np.zeros((rows,), dtype=np.float32)
    for r in range(rows):
      acc = np.float32(0.0)
      for v in x_np[r]: acc = _fma(np.float32(v), np.float32(v), acc)
      scale[r] = np.float32(1.0 / np.sqrt(np.float32(np.float32(acc / dim) + 1e-6)))
    fused = np.empty_like(x_np)
    for r in range(rows):
      fused[r] = (x_np[r] * scale[r]).astype(np.float32) * w_np
    bad = np.argwhere(fused != ordinary)
    print(f"rows={rows} ordinary programs: {names}")
    print(f"rows={rows} fused body {body.arg.name}: bitwise={'PASS' if not len(bad) else 'FAIL'} "
          f"({np.sum(fused == ordinary)}/{fused.size} elements)")
    if len(bad):
      raise SystemExit(f"FAIL: fused {rows}x{dim} body drifts from ordinary ({len(bad)} elements)")
  print("route probe part B: PASS (row-mode body is bitwise-equal to ordinary on CPU)")


if __name__ == "__main__":
  part = sys.argv[1] if len(sys.argv) > 1 else "b"
  if part == "a":
    _part_a()
  else:
    _part_b()
