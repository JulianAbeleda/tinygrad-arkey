#!/usr/bin/env python3
"""Trace REDUCE_OUTPUT marker admission across the full decode graph (CPU).

Runs the candidate arm conditions (callify flags + block promotion) with
REDUCE_OUTPUT_TRACE=1 and dumps the snapshot: marker parents at rangeify,
selector entry/reject reasons per warp/lane/per-lane association, and the
bounded parent chains.  Short prompt on purpose; the decode graph shape is
independent of prompt length.
"""
from __future__ import annotations

import json, sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad.helpers import Context
from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
from tinygrad.llm.reduce_output_trace import REDUCE_OUTPUT_TRACE, reset_reduce_output_trace, reduce_output_trace_snapshot
from extra.llm_research.decode.nv_reduce_output_fp32_qk_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


def main() -> int:
  with Context(DEBUG=0,
               CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1,
               CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1,
               REDUCE_OUTPUT_TRACE=1):
    reset_reduce_output_trace()
    model, _ = _model("candidate", MODEL, 32768)
    gen = model.generate(_prompt(MODEL, 64), chunk_size=32, temperature=0.0)
    try:
      int(next(gen))
      int(next(gen))
    finally:
      gen.close()
    snap = reduce_output_trace_snapshot()
  print(json.dumps(snap, indent=1, default=str))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
