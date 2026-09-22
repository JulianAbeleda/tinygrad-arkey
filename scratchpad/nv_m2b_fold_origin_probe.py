#!/usr/bin/env python3
"""Pinpoint where the opaque-base CONTIGUOUS inputs are created: fold result + custom_kernel contiguify."""
import sys
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from tinygrad.helpers import Context
from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
from tinygrad.llm import kernel_program as kp
from extra.llm_research.decode.nv_epilogue_absorption_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

def main() -> int:
  import tinygrad.uop.ops as ops
  real_fold = kp._fold_residual_input_views
  def spy_fold(inputs, program):
    out = real_fold(inputs, program)
    if program.residual_input_views and len(out) > 2:
      before = inputs[2].uop.base.op if len(inputs) > 2 else None
      after = out[2].uop.op
      print(f"FOLD {program.route_id.split('/')[0]} resviews={len(program.residual_input_views)} base_before={before} slot2_after={after}", flush=True)
    return out
  kp._fold_residual_input_views = spy_fold

  real_ck = ops.UOp.custom_kernel
  n = [0]
  def spy_ck(*srcs, fxn, grad_fxn=None):
    for i, s in enumerate(srcs):
      b = s.base
      if b.op in (ops.Ops.AFTER, ops.Ops.GETTUPLE) and s.op not in (ops.Ops.AFTER,) and not s.has_precompiled_output_identity():
        n[0] += 1
        print(f"CK_CONTIG {i} src={s.op.name}@{tuple(s.shape)} base={b.op.name}", flush=True)
    return real_ck(*srcs, fxn=fxn, grad_fxn=grad_fxn)
  ops.UOp.custom_kernel = spy_ck

  with Context(DEBUG=0, CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    model, _ = _model("candidate", MODEL, 32768)
    gen = model.generate(_prompt(MODEL, 64), chunk_size=32, temperature=0.0)
    try:
      for _ in range(3): int(next(gen))
    finally:
      gen.close()
  print(f"TOTAL_CK_CONTIG {n[0]}", flush=True)
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
