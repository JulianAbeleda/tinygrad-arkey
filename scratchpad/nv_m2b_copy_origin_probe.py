#!/usr/bin/env python3
"""Find what the remaining E_32_32_4_86a23e1a copy kernels actually copy."""
import contextlib, io, sys
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from tinygrad.helpers import Context
from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
from extra.llm_research.decode.nv_epilogue_absorption_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt
from extra.llm_research.decode.nv_reduce_output_primitive_ab import TM_RE

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

def _call_name(call):
  fn = call.src[0]
  arg = getattr(fn, "arg", None)
  if arg is not None and getattr(arg, "name", None) is not None: return arg.name
  return f"{fn.op.name}"

def main() -> int:
  import tinygrad.schedule as sched_mod
  from tinygrad.uop.ops import Ops
  orig = sched_mod.create_schedule
  captures = []
  def observed(sched_sink):
    linear = orig(sched_sink)
    captures.append(linear)
    return linear
  sched_mod.create_schedule = observed
  out = io.StringIO()
  try:
    with Context(DEBUG=0, CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
      model, _ = _model("candidate", MODEL, 32768)
      gen = model.generate(_prompt(MODEL, 64), chunk_size=32, temperature=0.0)
      try:
        int(next(gen))
        with contextlib.redirect_stdout(out):
          with Context(DEBUG=2): int(next(gen))
      finally:
        gen.close()
  finally:
    sched_mod.create_schedule = orig

  n_copies = 0
  for linear in captures:
    if linear is None or linear.op is not Ops.LINEAR: continue
    for c in linear.src:
      name = _call_name(c)
      if not name.startswith("E_32_32_4_86a23e1a"): continue
      n_copies += 1
      body = c.src[0]
      srcs = []
      for u in body.toposort():
        if u.op is Ops.STORE:
          srcs.append(u.src[1])
      for s in srcs:
        chain = []
        cur = s
        for _ in range(6):
          if cur is None or len(cur.src) == 0: break
          chain.append(f"{cur.op.name}@{tuple(cur.shape)}")
          cur = cur.src[0]
        print(f"COPY_SRC {name} store_src={s.op.name} chain={chain}", flush=True)
        # who produced this source buffer?
        try: b = s.buf_uop
        except RuntimeError: b = None
        prods = []
        for c2 in linear.src:
          if len(c2.src) >= 2:
            try:
              if c2.src[1].buf_uop is b: prods.append(_call_name(c2))
            except RuntimeError: pass
        print(f"  producer_calls={prods[:4]}", flush=True)
  print(f"TOTAL_86A23E1A {n_copies}", flush=True)
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
