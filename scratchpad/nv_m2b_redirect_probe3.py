#!/usr/bin/env python3
"""Trace declaration recording vs the FUNCTION boundary transform (probe 3).

Probe 2 showed registry_size=0 and result_ops=['MEMORY_SEMANTIC'] at transform time.  Find out
whether _execute_outputs ever records a declaration (and when), and what op chain sits under the
MEMORY_SEMANTIC body result."""
import sys
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from tinygrad.helpers import Context
from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
import tinygrad.callify as callify
from tinygrad.llm import kernel_program as kp
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, Ops
from extra.llm_research.decode.nv_epilogue_absorption_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

_real_transform = None

def _chain(x: UOp, depth: int = 0) -> str:
  if depth > 6: return "..."
  decl = "D" if callify._DECLARED_TYPED_OUTPUTS.get(x) is not None else ""
  return f"{x.op.name}{decl}({','.join(_chain(s, depth+1) for s in x.src[:2])})"

def _spy(c: UOp):
  real = _real_transform
  assert real is not None
  if not c.arg.precompile: return real(c)
  srcs = c.src[0].src
  print(f"SPY name={c.arg.name} n_results={len(srcs)} registry={len(callify._DECLARED_TYPED_OUTPUTS)} "
        f"chain={_chain(srcs[0]) if srcs else '-'}", flush=True)
  return real(c)

def main() -> int:
  global _real_transform
  _real_transform = callify.transform_precompiled_call
  callify.pm_precompile_function_boundary = PatternMatcher([
    (UPat(Ops.FUNCTION, name="c"), _spy),
    (UPat(Ops.GETTUPLE, src=(UPat(Ops.TUPLE, name="t"),), name="g"), lambda g, t: t.src[g.arg]),
  ])

  real_exec = kp._execute_outputs
  def spy_exec(output, inputs, program, allowed, boundary):
    out = real_exec(output, inputs, program, allowed, boundary)
    spec = program.output_spec
    if spec is not None and spec.typed_output is not None:
      print(f"RECORD program={program.program_id} typed={spec.typed_output is not None} "
            f"results={len(out)} ops={[r.uop.op.name for r in out]} "
            f"registry={len(callify._DECLARED_TYPED_OUTPUTS)}", flush=True)
    return out
  kp._execute_outputs = spy_exec

  with Context(DEBUG=0, CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    model, _ = _model("candidate", MODEL, 32768)
    gen = model.generate(_prompt(MODEL, 64), chunk_size=32, temperature=0.0)
    try:
      for _ in range(2): int(next(gen))
    finally:
      gen.close()
  print("DONE", flush=True)
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
