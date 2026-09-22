#!/usr/bin/env python3
"""Walk the executed schedule for the candidate arm and explain the q/k path.

Mirrors the campaign census window exactly (first token DEBUG=0, second token
DEBUG=2 stdout kernel list) and additionally intercepts ``create_schedule`` so
the LINEAR call list is retained for that same token.  Every fused q/k body
(``reduce_output_rmsnorm_32_128`` / ``_8_128``) is printed with its call
arguments, and the producer call that writes the x input buffer is classified
(materialized reduce, warp-coop partials, or missing).  The 17+17
``E_32_32_4_da50`` / ``E_8_32_4_1bd9`` kernels are dumped with rendered source
so they can be identified against the ledger.
"""
from __future__ import annotations

import contextlib, io, re, sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops
from extra.llm_research.decode.nv_reduce_output_fp32_qk_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt

TM_RE = re.compile(r"^ \*\*\* (?:GPU|CPU|NV|CUDA) +\d+ +(.*?) +arg +\d+ +mem [\d.]+ GB +tm (\d+\.\d+)(ms|us)/")


def _call_name(call) -> str:
  fn = call.src[0]
  arg = getattr(fn, "arg", None)
  if arg is not None and getattr(arg, "name", None) is not None: return arg.name
  return f"{fn.op.name}({getattr(arg, 'kind', '')})"


def _buf_label(u) -> str:
  while u is not None and len(u.src) and u.op not in {Ops.AFTER, Ops.BUFFER, Ops.PARAM, Ops.BIND}:
    u = u.src[0]
  return f"{u.op.name}{u._shape}" if u is not None else "?"


def main() -> int:
  import tinygrad.schedule.rangeify as rng
  import tinygrad.schedule as sched_mod
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  from tinygrad.llm.reduce_output_trace import REDUCE_OUTPUT_TRACE, reset_reduce_output_trace, reduce_output_trace_snapshot

  x_bindings: list[tuple[str, str]] = []
  linear_schedules: list = []
  orig = rng._reduce_derived_materialized_view
  orig_create_schedule = sched_mod.create_schedule

  def observed(x):
    ret = orig(x)
    label = "NONE"
    if ret is not None and ret.op is Ops.AFTER:
      label = f"AFTER({_buf_label(ret.src[0])}, src={[s.op.name for s in ret.src[1:]]})"
    x_bindings.append((str(x._shape), label))
    return ret

  def observed_create_schedule(sched_sink):
    linear = orig_create_schedule(sched_sink)
    linear_schedules.append(linear)
    return linear

  rng._reduce_derived_materialized_view = observed
  sched_mod.create_schedule = observed_create_schedule
  try:
    with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1,
                 REDUCE_OUTPUT_TRACE=1):
      reset_reduce_output_trace()
      model, _ = _model("candidate", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 32768)
      gen = model.generate(_prompt("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 512), chunk_size=32, temperature=0.0)
      with Context(DEBUG=0): next(gen)
      capture = io.StringIO()
      try:
        with contextlib.redirect_stdout(capture):
          with Context(DEBUG=2): token = int(next(gen))
      finally:
        gen.close()
      trace = reduce_output_trace_snapshot()
  finally:
    rng._reduce_derived_materialized_view = orig
    sched_mod.create_schedule = orig_create_schedule

  from collections import Counter
  print("materialized-view bindings:", dict(Counter(x_bindings)))
  rows = []
  for line in capture.getvalue().splitlines():
    if (match := TM_RE.match(line)):
      rows.append((match.group(1), float(match.group(2)) * (1000.0 if match.group(3) == "ms" else 1.0)))
  print(f"executed programs (DEBUG=2 token): {len(rows)}")

  by_name: dict[str, list] = {}
  for name, _ in rows: by_name.setdefault(name, []).append(None)

  interesting = [n for n in by_name if n.startswith(("r_", "E_", "q4k_warp_coop", "reduce_output_rmsnorm"))]
  for name in sorted(interesting):
    print(f"  {name[:76]:76s} x{len(by_name[name])}")

  print(f"\nschedule builds observed: {len(linear_schedules)}")
  fused_walked = 0
  for linear in linear_schedules:
    calls = list(linear.src) if linear is not None and linear.op is Ops.LINEAR else []
    fused = [c for c in calls if _call_name(c).startswith(("reduce_output_rmsnorm_32_128", "reduce_output_rmsnorm_8_128"))]
    fused_walked += len(fused)
    for call in fused[:8]:
      name = _call_name(call)
      args = call.src[1:]
      print(f"  {name}: args={[ _buf_label(a) for a in args ]}")
      x = args[1] if len(args) > 1 else None
      while x is not None and len(x.src) and x.op not in {Ops.AFTER, Ops.BUFFER, Ops.PARAM}:
        x = x.src[0]
      if x is not None and x.op is Ops.AFTER:
        print(f"    x AFTER srcs: {[s.op.name for s in x.src[1:]]}")
      xb = None
      if x is not None:
        try: xb = x.buf_uop
        except RuntimeError: pass
      if xb is not None:
        producers = []
        for c in calls:
          for a in c.src[1:]:
            try:
              if a.buf_uop is xb: producers.append(c)
            except RuntimeError: pass
        print(f"    x buffer producers: {[ _call_name(c) for c in producers ] or 'NONE (never written!)'}")
      print()
  print(f"fused q/k bodies walked: {fused_walked}")

  # Show the rendered source of one E_32_32_4_da50 / E_8_32_4_1bd9 kernel if present.
  for prefix in ("E_32_32_4_da50", "E_8_32_4_1bd9"):
    matches = [n for n in by_name if n.startswith(prefix)]
    print(f"\n{prefix}: {len(matches)} in executed set")
  print("\ntrace snapshot:", trace)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
