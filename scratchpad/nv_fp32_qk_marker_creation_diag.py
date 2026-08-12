#!/usr/bin/env python3
"""Dump the marker-creation (pre-callify) identity proof for q/k markers.

The production warp-coop carriers still arrive at rangeify with
``reduce_input_at_marker=False`` even though the hermetic unit fixture
passes, so the pre-callify chain must differ from the fixture.  This probe
wraps ``Tensor._semantic_reduce_output_rmsnorm`` and prints, for every q/k
marker, the full ``x.uop`` chain plus the verdict of each identity proof
(``_bounded_reduce_output_identity`` and its AFTER/precompiled sub-proof).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")


def chain(u, limit=14):
  out = []
  while u is not None and len(out) < limit:
    out.append(f"{u.op.name}/{u.dtype}{u._shape}")
    u = u.src[0] if u.src else None
  return out


def main() -> int:
  import tinygrad.tensor as tt
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  from tinygrad.helpers import Context
  from tinygrad.uop.ops import Ops
  from extra.llm_research.decode.nv_reduce_output_fp32_qk_ab import _model
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt

  seen: set = set()
  orig = tt.Tensor._semantic_reduce_output_rmsnorm

  def observed(self, x, out, weight, eps):
    ret = orig(self, x, out, weight, eps)
    arg = getattr(ret.uop, "arg", None)
    rows = getattr(arg, "rows", None)
    if rows in (8, 32):
      x_uop = x.uop
      key = (rows, tuple(chain(x_uop)))
      if key not in seen:
        seen.add(key)
        landed = x_uop
        while landed.op in {Ops.RESHAPE, Ops.MEMORY_SEMANTIC, Ops.PERMUTE}:
          if landed.op is Ops.PERMUTE and len(landed.src) != 1: break
          landed = landed.src[0]
        print(f"=== rows={rows} landed={landed.op.name}{landed._shape}")
        if landed.op in {Ops.CONTIGUOUS, Ops.REDUCE, Ops.AFTER}:
          print(f"  landed len_src={len(landed.src)}")
        print("  x chain:", " -> ".join(chain(x_uop)))
        if landed.op is Ops.CONTIGUOUS: landed = landed.src[0]
        expected = landed.numel()
        while landed.op is Ops.RESHAPE and len(landed.src) and landed.src[0].numel() == expected:
          landed = landed.src[0]
        if landed.op is Ops.REDUCE:
          expected = landed.src[0].numel()
          u = landed.src[0]
          while u.op is Ops.RESHAPE and len(u.src) and u.src[0].numel() == expected: u = u.src[0]
          print("  reduce-input chain:", " -> ".join(chain(u, 6)))
          if u.op is Ops.AFTER:
            base, call = u.src
            print("  after base:", base.op.name, base._shape, "->", " -> ".join(chain(base, 3)))
            print("  call:", call.op.name, getattr(call.arg, "name", None), "precompile=", getattr(call.arg, "precompile", None),
                  "args:", [(a.op.name, a._shape) for a in call.src[1:]][:8])
            body = call.src[0]
            print("  body op:", body.op.name)
            stores = []
            for su in body.toposort():
              if su.op is Ops.STORE and len(su.src):
                tgt = su.src[0]
                while tgt.op in (Ops.RESHAPE, Ops.INDEX) and len(tgt.src): tgt = tgt.src[0]
                stores.append((tgt.op.name, getattr(getattr(tgt, "arg", None), "slot", None), tgt is base))
            print("  body stores:", stores[:6])
          print("  bounded_after:", tt._bounded_after_output_identity(u),
                " precompiled:", u.has_precompiled_output_identity(),
                " reduce_identity:", tt._bounded_reduce_output_identity(x_uop))
          # step-by-step reimplementation of the opaque proof for diagnosis
          after = u
          print("  opaque steps: after_op", after.op.name, "len_src", len(after.src),
                "call_op", after.src[1].op.name, "call_args", len(after.src[1].src) - 1)
          base_buf = after.src[0]
          while base_buf.op is Ops.RESHAPE and len(base_buf.src) and base_buf.src[0].numel() == base_buf.numel():
            base_buf = base_buf.src[0]
          print("  base_buf:", base_buf.op.name, base_buf._shape, "numel", base_buf.numel(),
                "after_numel", after.numel(), "dtype_ok", base_buf.dtype == after.dtype)
          matches = 0
          for arg in after.src[1].src[1:]:
            a = arg
            while a.op is Ops.RESHAPE and len(a.src) and a.src[0].numel() == a.numel(): a = a.src[0]
            if a is base_buf: matches += 1
          print("  opaque matches:", matches)
          # step-by-step walk from the marker input (x_uop) to the REDUCE
          walk = x_uop
          if walk.op is Ops.CONTIGUOUS:
            print("  walk: contiguous len_src", len(walk.src))
            walk = walk.src[0]
          print("  walk: after contiguous", walk.op.name, walk._shape, "numel", walk.numel())
          while walk.op is Ops.RESHAPE and len(walk.src) and walk.src[0].numel() == walk.numel():
            walk = walk.src[0]
          print("  walk: after reshapes", walk.op.name, walk._shape, "len_src", len(walk.src))
          if walk.op is Ops.REDUCE:
            print("  walk: reduce src", [(w.op.name, w._shape, w.numel()) for w in walk.src])
        else:
          print("  landed op not REDUCE -> reduce_identity False")
    return ret

  tt.Tensor._semantic_reduce_output_rmsnorm = observed
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
    tt.Tensor._semantic_reduce_output_rmsnorm = orig
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
