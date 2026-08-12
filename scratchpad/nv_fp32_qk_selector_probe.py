#!/usr/bin/env python3
"""Dump marker input uop chains inside the rangeify selector.

Records the exact ``marker.src[1]`` chain seen by ``lower_reduce_output_store``
for q/k markers (rows 8/32), split by reject reason, so the REDUCE-derived
carrier spelling can be compared against the AFTER-carrier spelling at
rangeify time (post-callify), not just at marker creation.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")


def chain(u, limit=12):
  out = []
  while u is not None and len(out) < limit:
    out.append(f"{u.op.name}/{u.dtype}{u._shape}")
    u = u.src[0] if u.src else None
  return out


def main() -> int:
  import tinygrad.schedule.rangeify as rng
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  from tinygrad.helpers import Context
  from extra.llm_research.decode.nv_reduce_output_fp32_qk_ab import _model
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt

  rows_seen: list[tuple[str, str, list[str]]] = []
  orig = rng.lower_reduce_output_store

  def observed(store, carrier=None, marker=None, target=None):
    m = marker if marker is not None else (carrier.src[0] if carrier is not None else store.src[1])
    spec = getattr(m.arg, "rows", None)
    if spec in (8, 32):
      eligible = bool(getattr(m.arg, "input_identity_at_marker", False) or getattr(m.arg, "owned_contiguous_candidate", False))
      rows_seen.append((f"{spec}x{m.arg.dim}", "eligible" if eligible else "not_eligible", chain(m.src[1])))
    return orig(store, carrier, marker, target)

  rng.lower_reduce_output_store = observed
  try:
    with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
      model, _ = _model("candidate", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 32768)
      gen = model.generate(_prompt("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 512), chunk_size=32, temperature=0.0)
      try:
        int(next(gen))
        int(next(gen))
      finally:
        gen.close()
  finally:
    rng.lower_reduce_output_store = orig

  from collections import Counter
  print("selector q/k marker chains by (shape, eligibility):",
        dict(Counter((shape, elig) for shape, elig, _ in rows_seen)))
  seen: set = set()
  for shape, elig, ch in rows_seen:
    key = (shape, elig, tuple(ch))
    if key in seen: continue
    seen.add(key)
    print(f"\n--- {shape} {elig} ---")
    print("  " + " -> ".join(ch))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
