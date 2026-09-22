#!/usr/bin/env python3
"""Capture candidate/control program sources and q/k marker input chains.

The fp32 q/k wall-bracket record shows +38 ``E_32_4_0fd8e427...`` kernels per
decode token in the candidate census (one per fused q/k body, 19 q + 19 k).
This probe runs one decode token under the exact harness arm conditions
(``--arm candidate|control``) and dumps the rendered source for every captured
program whose name starts with ``E_32_4`` or ``E_2_8_16_4`` / ``E_8_2_16_4``
(the ordinary q/k epilogues), plus the fused body names.  With
``--trace-markers`` it also records the marker input uop chain for every q/k
marker call so the admitted vs rejected carriers can be compared directly.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from extra.llm_research.decode.nv_reduce_output_fp32_qk_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt
from tinygrad.helpers import Context


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", default="candidate", choices=("candidate", "control"))
  ap.add_argument("--trace-markers", action="store_true")
  args = ap.parse_args()
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  import tinygrad.engine.realize as realize
  import tinygrad.tensor as tensor_mod

  captured: dict[str, object] = {}
  orig = realize.get_runtime
  marker_chains: list[tuple[str, str, list[str]]] = []

  def patched(device, ast, cache=True):
    if ast.op is not __import__("tinygrad.uop.ops", fromlist=["Ops"]).Ops.PROGRAM: return orig(device, ast, cache)
    name = ast.arg.function_name
    if name.startswith(("E_32_4", "E_2_8_16_4", "E_8_2_16_4", "E_4_2_8_16", "reduce_output_rmsnorm", "r_2_8_4_4_16", "r_8_16_8")):
      captured.setdefault(name, ast)
    return orig(device, ast, cache)

  def chain(u, limit=10):
    out = []
    while u is not None and len(out) < limit:
      out.append(f"{u.op.name}{'/' + str(u.dtype) if u.dtype else ''}{u._shape}")
      u = u.src[0] if u.src else None
    return out

  orig_marker = tensor_mod.Tensor._semantic_reduce_output_rmsnorm
  def observed_marker(self, x, out, weight, eps):
    rows = int(__import__("math").prod(x.shape[:-1])) if all(isinstance(s, int) for s in x.shape) else -1
    if rows in (8, 32):
      marker_chains.append((f"{rows}x{x.shape[-1]}", str(x.dtype),
                            chain(x.uop) + ["| weight:"] + chain(weight.uop)))
    return orig_marker(self, x, out, weight, eps)

  realize.get_runtime = patched
  if args.trace_markers:
    tensor_mod.Tensor._semantic_reduce_output_rmsnorm = observed_marker
  try:
    if args.arm == "candidate":
      ctx = Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1)
    else:
      ctx = Context()
    with ctx:
      model, gates = _model(args.arm, "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 32768)
      gen = model.generate(_prompt("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 512), chunk_size=32, temperature=0.0)
      try:
        int(next(gen))
        int(next(gen))
      finally:
        gen.close()
  finally:
    realize.get_runtime = orig
    tensor_mod.Tensor._semantic_reduce_output_rmsnorm = orig_marker

  print(f"gates: {gates}")
  if args.trace_markers:
    from collections import Counter
    by_shape = Counter((shape, dtype) for shape, dtype, _ in marker_chains)
    print(f"\nmarker chain counts by (shape, dtype): {dict(by_shape)}")
    for shape, dtype, ch in marker_chains[:12]:
      print(f"\n--- marker {shape} {dtype} ---")
      print("  " + " -> ".join(ch))
  print(f"captured programs: {sorted(captured)}")
  for name, ast in sorted(captured.items()):
    print(f"\n{'=' * 80}\n=== {name} ===\n{'=' * 80}")
    for u in ast.src:
      if getattr(u, "op", None) is not None and u.op.name == "SOURCE":
        print("==== CUDA SOURCE ====")
        print(u.arg)
        break
    else:
      print("(no SOURCE uop found; showing sink tree)")
      sink = ast.src[0]
      print(f"sink arg: {sink.arg}")
      for u in sink.toposort():
        if u.op.name in ("STORE", "BARRIER", "RANGE", "RANGE2", "SPECIAL", "PARAM", "BUFFER", "END", "AFTER", "GROUP"):
          print(f"  {u.op.name} {u.arg} shape={getattr(u, '_shape', None)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
