#!/usr/bin/env python3
"""Log every materializing CONTIGUOUS created over a GETTUPLE/AFTER base during the candidate trace."""
import sys
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from tinygrad.helpers import Context
from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
from extra.llm_research.decode.nv_epilogue_absorption_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

def main() -> int:
  import tinygrad.uop.ops as ops
  real = ops.UOp.contiguous
  n = [0]
  def spy(self, *args, **kwargs):
    r = real(self, *args, **kwargs)
    if r is not self and r.op is ops.Ops.CONTIGUOUS:
      base = self.base
      if base.op in (ops.Ops.GETTUPLE, ops.Ops.AFTER):
        n[0] += 1
        print(f"CONTIG_OPAQUE {self.op.name}@{tuple(self.shape)} base={base.op.name} precomp={self.has_precompiled_output_identity()} -> {r.op.name}", flush=True)
    return r
  ops.UOp.contiguous = spy
  with Context(DEBUG=0, CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    model, _ = _model("candidate", MODEL, 32768)
    gen = model.generate(_prompt(MODEL, 64), chunk_size=32, temperature=0.0)
    try:
      for _ in range(4): int(next(gen))
    finally:
      gen.close()
  print(f"TOTAL_CONTIG_OPAQUE {n[0]}", flush=True)
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
