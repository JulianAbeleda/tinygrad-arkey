#!/usr/bin/env python3
"""Dump marker-creation identity proofs for rows=1 block norms (attn/ffn/final).

Runs the candidate arm on NV and wraps ``Tensor._semantic_reduce_output_rmsnorm``
to record, for every rows=1 marker, the input chain at creation plus each
identity proof verdict (buffer/precompiled/contiguous/after/reduce).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad.helpers import Context
from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
from extra.llm_research.decode.nv_reduce_output_fp32_qk_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


def _chain(u, limit=12):
  out = []
  while u is not None and len(out) < limit:
    out.append(f"{u.op.name}/{u.dtype}{u._shape}")
    u = u.src[0] if u.src else None
  return out


def main() -> int:
  import tinygrad.tensor as tt
  from tinygrad.uop.ops import Ops

  seen: set = set()
  rows: list[dict] = []

  orig = tt.Tensor._semantic_reduce_output_rmsnorm

  def observed(self, x, out, weight, eps):
    ret = orig(self, x, out, weight, eps)
    spec = getattr(ret.uop, "arg", None)
    if getattr(spec, "rows", None) == 1:
      x_uop = x.uop
      identity_uop = x_uop
      while identity_uop.op in {Ops.RESHAPE, Ops.MEMORY_SEMANTIC, Ops.PERMUTE}:
        if identity_uop.op is Ops.PERMUTE and len(identity_uop.src) != 1: break
        identity_uop = identity_uop.src[0]
      precompiled_contiguous = identity_uop.op is Ops.CONTIGUOUS and identity_uop.src[0].has_precompiled_output_identity()
      key = (str(x_uop._shape), tuple(_chain(x_uop, 8)))
      if key not in seen:
        seen.add(key)
        rows.append({
          "shape": str(x_uop._shape),
          "identity": bool(spec.input_identity_at_marker),
          "owned": bool(spec.owned_contiguous_candidate),
          "reduce": bool(spec.reduce_input_at_marker),
          "landed": f"{identity_uop.op.name}{identity_uop._shape}",
          "precompiled_contiguous": bool(precompiled_contiguous),
          "buffer_identity": bool(identity_uop.has_buffer_identity()),
          "precompiled_identity": bool(identity_uop.has_precompiled_output_identity()),
          "after_identity": bool(tt._bounded_after_output_identity(identity_uop)),
          "reduce_identity": bool(tt._bounded_reduce_output_identity(identity_uop)),
          "chain": _chain(x_uop, 8),
        })
    return ret

  tt.Tensor._semantic_reduce_output_rmsnorm = observed
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
    tt.Tensor._semantic_reduce_output_rmsnorm = orig

  import json
  print(json.dumps(rows, indent=1))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
