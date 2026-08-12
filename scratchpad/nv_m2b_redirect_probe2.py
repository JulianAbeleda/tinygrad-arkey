#!/usr/bin/env python3
"""Check whether the M2b absorbed block function takes the declared-AFTER output redirect.

The precompile-boundary PatternMatcher captures transform_precompiled_call at import time, so
patching the module attribute (probe 1) never fired.  Here the pattern matcher itself is rebuilt
with a spy around the transform."""
import sys
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from tinygrad.helpers import Context
from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
import tinygrad.callify as callify
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, Ops
from extra.llm_research.decode.nv_epilogue_absorption_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

_real_transform = None

def _spy(c: UOp):
  real = _real_transform
  assert real is not None
  srcs = c.src[0].src
  declared = callify._body_output_is_declared_after(srcs)
  marker = callify._body_output_carries_reduce_output_marker(srcs)
  if c.arg.precompile or declared or marker:
    print(f"SPY name={c.arg.name} precompile={c.arg.precompile} n_results={len(srcs)} "
          f"marker={marker} declared={declared} "
          f"result_ops={[s.op.name for s in srcs]} "
          f"registry_size={len(callify._DECLARED_TYPED_OUTPUTS)}", flush=True)
  return real(c)

def main() -> int:
  global _real_transform
  _real_transform = callify.transform_precompiled_call
  callify.pm_precompile_function_boundary = PatternMatcher([
    (UPat(Ops.FUNCTION, name="c"), _spy),
    (UPat(Ops.GETTUPLE, src=(UPat(Ops.TUPLE, name="t"),), name="g"), lambda g, t: t.src[g.arg]),
  ])
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
