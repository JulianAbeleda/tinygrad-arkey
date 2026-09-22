#!/usr/bin/env python3
"""Check whether the M2b absorbed block function takes the declared-AFTER output redirect."""
import sys
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from tinygrad.helpers import Context
from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER, _DECLARED_TYPED_OUTPUTS
import tinygrad.callify as callify
from tinygrad.uop.ops import Ops
from extra.llm_research.decode.nv_epilogue_absorption_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

def main() -> int:
  real = callify.transform_precompiled_call
  def spy(c):
    if not c.arg.precompile: return real(c)
    srcs = c.src[0].src
    out_route = callify.CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT and (
      callify._body_output_carries_reduce_output_marker(srcs) or callify._body_output_is_declared_after(srcs))
    if out_route or any(s.op is Ops.AFTER for s in srcs):
      print(f"TRANSFORM name={c.arg.name} n_results={len(srcs)} declared_after={callify._body_output_is_declared_after(srcs)} out_route={out_route} result_ops={[s.op.name for s in srcs]}", flush=True)
    return real(c)
  callify.transform_precompiled_call = spy
  with Context(DEBUG=0, CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    model, _ = _model("candidate", MODEL, 32768)
    gen = model.generate(_prompt(MODEL, 64), chunk_size=32, temperature=0.0)
    try:
      for _ in range(3): int(next(gen))
    finally:
      gen.close()
  print("DONE", flush=True)
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
