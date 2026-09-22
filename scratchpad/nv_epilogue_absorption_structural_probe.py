#!/usr/bin/env python3
"""Candidate-arm structural probe for the M2 fp16-store absorption: map every
E_128_32_3 / E_32_32_4 program to its producer/consumer calls and print the
execution-order neighborhood, with the M2 lease installed."""
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
  lines = capture.getvalue().splitlines()
  print(f"DEBUG=2 lines: {len(lines)}, parsed rows: {len(rows)}")
  for line in lines[:3]: print("SAMPLE:", repr(line[:160]))

  calls = []
  all_names = []
  for linear in linear_schedules:
    if linear is not None and linear.op is Ops.LINEAR:
      for c in linear.src:
        n = _call_name(c)
        all_names.append(n)
        if n.startswith(("E_32_32_4", "E_128_32_3")):
          calls.append(c)
  print(f"residual-family calls found across schedule builds: {len(calls)}")

  def buf_id(a):
    try: return a.buf_uop
    except RuntimeError: return None

  write_map = {}
  for c in calls:
    name = _call_name(c)
    if len(c.src) >= 2:
      b = buf_id(c.src[1])
      if b is not None: write_map.setdefault(b, []).append(name)

  names = [n for n, _ in rows]
  seen_neigh: dict[str, tuple] = {}
  for i, n in enumerate(names):
    if not n.startswith(("E_32_32_4", "E_128_32_3")): continue
    parts = n.split("_")
    key = parts[0] + "_" + (parts[1] if len(parts) > 1 else "") + "_" + (parts[2][:8] if len(parts) > 2 else "")
    if key not in seen_neigh:
      lo, hi = max(0, i - 3), min(len(names), i + 4)
      seen_neigh[key] = (n, names[lo:i], names[i + 1:hi])
  for key, (name, before, after) in sorted(seen_neigh.items()):
    print(f"ORDER {name[:60]:60s}")
    print(f"   before: {[x[:44] for x in before]}")
    print(f"   after:  {[x[:44] for x in after]}")

  for c in calls:
    name = _call_name(c)
    if not name.startswith(("E_32_32_4", "E_128_32_3")): continue
    args = c.src[1:]
    inputs = []
    for a in args[1:]:
      b = buf_id(a)
      producers = write_map.get(b, [])
      inputs.append(f"{producers or ['PARAM?']}")
    out = buf_id(args[0]) if args else None
    consumers = write_map.get(out, []) if out is not None else []
    print(f"{name[:70]:70s} in={inputs[:4]} out_consumers={consumers[:6]}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
