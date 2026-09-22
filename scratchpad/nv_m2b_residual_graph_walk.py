#!/usr/bin/env python3
"""Walk the captured decode graph and print, for every E_32_32_4 / E_128_32_3 /
ffn GEMV / rmsnorm body, the input buffer producers (kernel names) and dtypes,
so the residual-family chain is ground truth, not ledger guesses."""
from __future__ import annotations

import sys
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad.helpers import Context
from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
from tinygrad.engine.jit import GraphAdmissionCensus, observe_graph_admissions
from extra.llm_research.decode.nv_epilogue_absorption_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt
from tinygrad.uop.ops import Ops

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

def main() -> int:
  from tinygrad.helpers import TRACEMETA
  with Context(DEBUG=0, TRACEMETA=1, CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1,
               CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    model, _ = _model("candidate", MODEL, 32768)
    gen = model.generate(_prompt(MODEL, 64), chunk_size=32, temperature=0.0)
    admission = GraphAdmissionCensus()
    try:
      int(next(gen))
      for index in range(3):
        if index == 1:
          with observe_graph_admissions(admission):
            int(next(gen))
        else:
          next(gen)
    finally:
      gen.close()
  # map program name -> rendered source presence via the admission records
  interesting = ("E_32_32_4", "E_128_32_3", "q4k_g3_lanemap_gemv_4096_12288", "q6k_gen_coop_4096_12288",
                 "reduce_output_rmsnorm", "rmsnorm_q8", "r_16_256", "q4k_g3_lanemap_gemv_epi_resadd",
                 "w1w3fused", "flash_fused_gmax_combine", "flash_block_tiled")
  for record in admission.records:
    name = record.program_name or ""
    if not any(p in name for p in interesting): continue
    print(f"{name[:60]}", flush=True)
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
