#!/usr/bin/env python3
"""Dump the rendered source of every E_128_32_3 program under the M2 candidate
arm, plus the w1+w3 route's store_fp16 flag and output dtype, to explain why
the ffn-activation cast survives the fp16 store."""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad.helpers import Context
from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
from extra.llm_research.decode.nv_epilogue_absorption_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


def main() -> int:
  arm = sys.argv[1] if len(sys.argv) > 1 else "candidate"
  import tinygrad.renderer.cstyle as cstyle
  orig = cstyle.CStyleLanguage.render_kernel

  def hooked(self, function_name, kernel, bufs, uops, prefix=None):
    src = orig(self, function_name, kernel, bufs, uops, prefix)
    if function_name.startswith("E_128_32_3") or function_name.startswith("q4k_g3_lanemap_gemv_w1w3fused"):
      print(f"=== E_128_32_3 source ===\n{src}\n", flush=True)
    return src

  cstyle.CStyleLanguage.render_kernel = hooked
  import tinygrad.llm.model as model_mod
  from tinygrad.llm.decode_routes import q4k_gate_up_primitive_linear_call as route_call

  def patched_route(gate, up, x, fallback, **kw):
    z = route_call(gate, up, x, fallback, **kw)
    print(f"W1W3 ROUTE store_fp16={kw.get('store_fp16', False)} x.dtype={x.dtype} z.dtype={z.dtype}", flush=True)
    return z

  model_mod.q4k_gate_up_primitive_linear_call = patched_route
  print(f"ARM={arm}", flush=True)
  with Context(DEBUG=0, CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1,
               CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    model, _ = _model(arm, MODEL, 32768)
    gen = model.generate(_prompt(MODEL, 64), chunk_size=32, temperature=0.0)
    try:
      int(next(gen))
      for _ in range(3): int(next(gen))
    finally:
      gen.close()
  print("done", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
