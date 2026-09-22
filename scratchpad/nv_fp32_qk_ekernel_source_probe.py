#!/usr/bin/env python3
"""Capture rendered CUDA source for the q/k materialized-reduce kernels.

Runs the campaign's exact two-token census window and captures, at compile
time, the CUDA source for every program whose name matches the q/k families:
``reduce_output_rmsnorm_32_128`` / ``_8_128`` (fused bodies),
``r_2_8_4_4_16`` / ``r_8_16_8`` (ordinary q/k reduces) and any
``E_32_32_4`` / ``E_8_32_4`` elementwise (the candidate's materialized
reduces render here).  Prints the executed kernel histogram and the rendered
source for one representative of each family so the numeric mismatch can be
read directly from the generated code.
"""
from __future__ import annotations

import argparse, contextlib, io, re, sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from extra.llm_research.decode.nv_reduce_output_fp32_qk_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt
from tinygrad.helpers import Context

TM_RE = re.compile(r"^\*\*\* NV\s+\d+\s+(\S+)\s+arg\s+\d+.*?tm\s+([\d.]+)(us|ms)/")


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", default="candidate", choices=("candidate", "control"))
  args = ap.parse_args()
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  import tinygrad.engine.realize as realize

  captured: dict[str, object] = {}
  orig = realize.get_runtime

  def patched(device, ast, cache=True):
    from tinygrad.uop.ops import Ops
    if ast.op is not Ops.PROGRAM: return orig(device, ast, cache)
    name = ast.arg.name
    if name.startswith(("reduce_output_rmsnorm_32_128", "reduce_output_rmsnorm_8_128",
                        "r_2_8_4_4_16", "r_8_16_8", "E_32_32_4", "E_8_32_4", "r_8_32_4_4")):
      captured.setdefault(name, ast)
    return orig(device, ast, cache)

  realize.get_runtime = patched
  try:
    if args.arm == "candidate":
      ctx = Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1)
    else:
      ctx = Context()
    with ctx:
      model, _ = _model(args.arm, "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 32768)
      gen = model.generate(_prompt("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 512), chunk_size=32, temperature=0.0)
      with Context(DEBUG=0): next(gen)
      capture = io.StringIO()
      try:
        with contextlib.redirect_stdout(capture):
          with Context(DEBUG=2): token = int(next(gen))
      finally:
        gen.close()
  finally:
    realize.get_runtime = orig

  rows = []
  for line in capture.getvalue().splitlines():
    if (match := TM_RE.match(line)):
      rows.append((match.group(1), float(match.group(2)) * (1000.0 if match.group(3) == "ms" else 1.0)))
  from collections import Counter
  print(f"arm={args.arm} executed={len(rows)} token={token}")
  counts = Counter(name for name, _ in rows)
  for name in sorted(counts):
    if name.startswith(("r_", "E_", "reduce_output_rmsnorm", "q4k_warp_coop", "q4k_g3")):
      print(f"  {name[:78]:78s} x{counts[name]}")
  for name, ast in sorted(captured.items()):
    print(f"\n{'=' * 90}\n=== {name} ===\n{'=' * 90}")
    src = ""
    for u in ast.toposort():
      if u.op.name == "SOURCE" and isinstance(u.arg, str):
        src = u.arg
        break
    lines = src.splitlines() if src else []
    # print the kernel body only (skip host-side struct/headers noise)
    print("\n".join(lines[:120]) if lines else "(no SOURCE uop)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
