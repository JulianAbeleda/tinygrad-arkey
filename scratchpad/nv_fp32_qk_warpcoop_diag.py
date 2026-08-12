#!/usr/bin/env python3
"""Diagnose why the warp-coop REDUCE-derived q/k markers do not admit on GPU.

Monkeypatches ``lower_reduce_output_store`` to record, for every q/k marker
that reaches rangeify (rows 8/32), the marker spec flags and the exact
``marker.src[1]`` chain, plus the REDUCE_OUTPUT_TRACE selector snapshot
(admission/reject reasons per association).  The production census runs
without this detail, so this probe reproduces the candidate decode window
and prints the evidence directly.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")


def chain(u, limit=16):
  out = []
  while u is not None and len(out) < limit:
    out.append(f"{u.op.name}/{u.dtype}{u._shape}")
    u = u.src[0] if u.src else None
  return out


def main() -> int:
  import tinygrad.schedule.rangeify as rng
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  from tinygrad.helpers import Context
  from tinygrad.llm.reduce_output_trace import REDUCE_OUTPUT_TRACE, reduce_output_trace_snapshot, reset_reduce_output_trace
  from extra.llm_research.decode.nv_reduce_output_fp32_qk_ab import _model
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt

  rows_seen: list[tuple[str, dict, list[str]]] = []
  orig = rng.lower_reduce_output_store

  def observed(store, carrier=None, marker=None, target=None):
    m = marker if marker is not None else (carrier.src[0] if carrier is not None else store.src[1])
    arg = getattr(m, "arg", None)
    rows = getattr(arg, "rows", None)
    if rows in (8, 32):
      flags = {name: bool(getattr(arg, name, False)) for name in
               ("input_identity_at_marker", "owned_contiguous_candidate", "reduce_input_at_marker")}
      rows_seen.append((f"{rows}x{getattr(arg, 'dim', '?')}", flags, chain(m.src[1])))
    return orig(store, carrier, marker, target)

  rng.lower_reduce_output_store = observed
  try:
    with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1,
                 REDUCE_OUTPUT_TRACE=1):
      reset_reduce_output_trace()
      model, _ = _model("candidate", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 32768)
      gen = model.generate(_prompt("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 512), chunk_size=32, temperature=0.0)
      try:
        int(next(gen))
        int(next(gen))
      finally:
        gen.close()
      trace = reduce_output_trace_snapshot()
  finally:
    rng.lower_reduce_output_store = orig

  from collections import Counter
  print("selector q/k markers at rangeify:", dict(Counter((shape, tuple(flags.values())) for shape, flags, _ in rows_seen)))
  for shape, flags, ch in rows_seen:
    print(f"--- {shape} flags={flags}")
    print("    " + " -> ".join(ch))
  print("trace snapshot:", trace)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
