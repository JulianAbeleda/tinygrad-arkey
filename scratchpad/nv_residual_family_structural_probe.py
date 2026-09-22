#!/usr/bin/env python3
"""Map every residual-family kernel (E_32_32_4_*, E_128_32_3) to its graph
position: which GEMV output feeds it and which consumer reads its output.

Runs the candidate arm on NV, intercepts ``create_schedule`` for the DEBUG=2
token, and for each elementwise program records the producer call of its
input buffer and the consumer calls of its output buffer.
"""
from __future__ import annotations

import contextlib, io, sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad.helpers import Context
from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
from extra.llm_research.decode.nv_epilogue_absorption_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt
from extra.llm_research.decode.nv_reduce_output_primitive_ab import TM_RE

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


def _call_name(call):
  fn = call.src[0]
  arg = getattr(fn, "arg", None)
  if arg is not None and getattr(arg, "name", None) is not None: return arg.name
  return f"{fn.op.name}"


def main() -> int:
  import tinygrad.schedule as sched_mod
  from tinygrad.uop.ops import Ops

  linear_schedules = []
  orig = sched_mod.create_schedule

  def observed(sched_sink):
    linear = orig(sched_sink)
    linear_schedules.append(linear)
    return linear

  sched_mod.create_schedule = observed
  capture = io.StringIO()
  try:
    with Context(DEBUG=0, CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1,
                 CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
      model, _ = _model("candidate", MODEL, 32768)
      gen = model.generate(_prompt(MODEL, 64), chunk_size=32, temperature=0.0)
      try:
        int(next(gen))
        with contextlib.redirect_stdout(capture):
          with Context(DEBUG=2): int(next(gen))
      finally:
        gen.close()
  finally:
    sched_mod.create_schedule = orig

  rows = []
  for line in capture.getvalue().splitlines():
    if (m := TM_RE.match(line)):
      us = float(m.group(2)) * (1000.0 if m.group(3) == "ms" else 1.0)
      rows.append((m.group(1), us))
  executed = [n for n, _ in rows]
  from collections import Counter
  print(f"executed programs: {len(executed)}, residual family: "
        f"{sum(1 for n in executed if n.startswith(('E_32_32_4', 'E_128_32_3')))}")

  calls = []
  for linear in linear_schedules:
    if linear is not None and linear.op is Ops.LINEAR:
      for c in linear.src:
        n = _call_name(c)
        if n.startswith(("E_32_32_4", "E_128_32_3", "q4k_g3_lanemap_gemv", "q6k_gen_coop")):
          calls.append(c)
  print(f"family calls across schedule builds: {len(calls)}")

  def buf_id(a):
    try: return a.buf_uop
    except RuntimeError: return None

  write_map = {}
  for c in calls:
    name = _call_name(c)
    if len(c.src) >= 2:
      b = buf_id(c.src[1])
      if b is not None: write_map.setdefault(b, []).append(name)

  for c in calls:
    name = _call_name(c)
    args = c.src[1:]
    inputs = []
    for a in args[1:]:
      b = buf_id(a)
      producers = write_map.get(b, [])
      inputs.append(f"{producers or ['PARAM?']}")
    out = buf_id(args[0]) if args else None
    consumers = write_map.get(out, []) if out is not None else []
    if name.startswith(("E_32_32_4", "E_128_32_3")) or consumers:
      print(f"{name[:70]:70s} in={inputs[:4]} out_consumers={consumers[:6]}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
