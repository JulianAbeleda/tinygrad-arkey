#!/usr/bin/env python3
"""Record WHY rows=1 block-norm markers are rejected by the reduce-output
selector (input proof), with the exact marker input chain at rangeify.

Runs the candidate arm on NV (callify flags + block promotion), wraps
``lower_reduce_output_store``, and for every rows=1 marker records the
eligibility bits, each input-proof verdict, and the marker.src[1] chain.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad.helpers import Context
from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
from extra.llm_research.decode.nv_reduce_output_fp32_qk_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


def _chain(u, limit=14):
  out = []
  while u is not None and len(out) < limit:
    out.append(f"{u.op.name}/{u.dtype}{u._shape}")
    u = u.src[0] if u.src else None
  return out


def main() -> int:
  import tinygrad.schedule.rangeify as rng
  import tinygrad.tensor as tt

  rows: list[dict] = []

  def chain_len(u, max_=4):
    n = 0
    while u is not None and len(u.src) and n < max_:
      u = u.src[0]; n += 1
    return n

  def describe_view(v):
    if v is None: return None
    return f"{v.op.name}{v._shape}"

  orig_lower = rng.lower_reduce_output_store
  orig_id = rng._identity_buffer_view
  orig_m4 = rng._reduce_output_m4_input_view
  orig_reduce = rng._reduce_derived_materialized_view
  orig_inv = rng._proven_invocation_input_view
  orig_owned = rng._owned_precompiled_output_after_view

  def observed_lower(store, carrier=None, marker=None, target=None):
    m = marker if marker is not None else (carrier.src[0] if carrier is not None else store.src[1])
    spec = getattr(m, "arg", None)
    if getattr(spec, "rows", None) == 1:
      x = m.src[1]
      rows.append({
        "shape": str(x._shape), "dtype": str(x.dtype),
        "eligible": bool(spec.input_identity_at_marker or spec.owned_contiguous_candidate or spec.reduce_input_at_marker),
        "identity_bit": bool(getattr(spec, "input_identity_at_marker", None)),
        "owned_bit": bool(getattr(spec, "owned_contiguous_candidate", None)),
        "reduce_bit": bool(getattr(spec, "reduce_input_at_marker", None)),
        "inv_slot": getattr(spec, "invocation_input_slot", None),
        "id_view": describe_view(orig_id(x)),
        "inv_view": describe_view(orig_inv(x, spec.invocation_input_slot) if spec.invocation_input_slot is not None else None),
        "owned_view": describe_view(orig_owned(x)),
        "m4_view": describe_view(orig_m4(x)),
        "reduce_view": describe_view(orig_reduce(x)),
        "chain": _chain(x),
      })
    return orig_lower(store, carrier, marker, target)

  rng.lower_reduce_output_store = observed_lower
  try:
    with Context(DEBUG=0, CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1,
                 CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
      model, _ = _model("candidate", MODEL, 32768)
      gen = model.generate(_prompt(MODEL, 64), chunk_size=32, temperature=0.0)
      try:
        int(next(gen))
        int(next(gen))
      finally:
        gen.close()
  finally:
    rng.lower_reduce_output_store = orig_lower

  from collections import Counter
  verdicts = Counter(
    (r["id_view"] is not None, r["m4_view"] is not None, r["reduce_view"] is not None) for r in rows)
  print("input-proof verdict triple (identity, m4, reduce):", dict(verdicts))
  chains = Counter(tuple(r["chain"][:6]) for r in rows)
  print(f"\n{len(rows)} rows=1 selector entries; {len(chains)} distinct 6-deep chains:")
  for chain, count in sorted(chains.items(), key=lambda kv: -kv[1]):
    print(f"  x{count}:", " -> ".join(chain))
  print("\nper-row detail (full):")
  for r in rows:
    print(r)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
