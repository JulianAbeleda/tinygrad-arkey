#!/usr/bin/env python3
"""Probe the exact residual-slot uop chain under the M2b candidate arm."""
import sys
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from tinygrad.helpers import Context
from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
from tinygrad.llm import kernel_program as kp
from extra.llm_research.decode.nv_epilogue_absorption_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
orig = kp._fold_residual_input_views
seen = 0
def hooked(inputs, program):
  global seen
  if program.program_id.endswith(".gemv") and len(inputs) > 2:
    u = inputs[2].uop
    print(f"RESIDUAL {program.route_id}/{program.program_id} resviews={program.residual_input_views} uop={u.op} dtype={u.dtype} shape={u.shape} base={u.base.op} base_id={u.base.has_buffer_identity()}/{u.base.has_precompiled_output_identity()}", flush=True)
    cur = u
    for _ in range(8):
      if len(cur.src) == 0: break
      cur = cur.src[0]
      print(f"  leg {cur.op} dtype={cur.dtype} shape={cur.shape} precomp={cur.has_precompiled_output_identity()} buf={cur.has_buffer_identity()}", flush=True)
    seen += 1
    if seen > 4: raise SystemExit(0)
  return orig(inputs, program)
kp._fold_residual_input_views = hooked
with Context(DEBUG=0, CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
  model, _ = _model("candidate", MODEL, 32768)
  gen = model.generate(_prompt(MODEL, 64), chunk_size=32, temperature=0.0)
  try:
    int(next(gen))
    for _ in range(3): int(next(gen))
  finally:
    gen.close()
