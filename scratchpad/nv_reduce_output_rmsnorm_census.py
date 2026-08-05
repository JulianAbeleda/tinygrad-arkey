#!/usr/bin/env python3
"""Capture only the forced decode-only REDUCE_OUTPUT production graph."""
import argparse, collections, json, pathlib

from extra.llm_research.decode.decode_harness import DEFAULT_MODEL
from extra.llm_research.decode.decode_runtime_overhead import _make_prompt, capture_decode_graph


def main() -> int:
  ap=argparse.ArgumentParser()
  ap.add_argument("--model",default=DEFAULT_MODEL); ap.add_argument("--depth",type=int,default=512)
  ap.add_argument("--max-context",type=int,default=4608); ap.add_argument("--chunk-size",type=int,default=32)
  ap.add_argument("--typed-semantic-producer",action="store_true",
                  help="enable only the closed typed CALL-input producer route with callify redirect")
  ap.add_argument("--out",required=True); args=ap.parse_args()
  from dataclasses import replace
  from tinygrad.llm.generate import load_model_and_tokenizer
  model,tokenizer=load_model_and_tokenizer(args.model,args.max_context,seed=20260617)
  model.config=replace(model.config,prefill_tc_attn=False,prefill_custom_kernel_attn=False)
  # This census is defined over the production-qualified direct greedy flash
  # capture.  Keep the local harness on that existing route explicitly; it
  # must not silently fall back to the legacy rollout_jit name.
  model._decode_direct_greedy_promoted=True
  model._decode_reduce_output_rmsnorm_promoted=True
  for block in model.blk:
    block.config=replace(block.config,prefill_tc_attn=False,prefill_custom_kernel_attn=False)
    block._decode_reduce_output_rmsnorm_promoted=True
  # Observe the exact pre-callify values passed by production model call sites.
  # This is census-local and leaves model semantics unchanged.
  import tinygrad.llm.model as model_module
  from tinygrad.helpers import Context
  from tinygrad.llm.reduce_output_trace import REDUCE_OUTPUT_TRACE, reset_reduce_output_trace, reduce_output_trace_snapshot
  original_marker=model_module._decode_reduce_output_rmsnorm
  marker_inputs=[]
  def observed_marker(norm,x,promoted):
    if promoted:
      marker_inputs.append({"op":x.uop.op.name,"base_op":x.uop.base.op.name,"shape":list(x.shape),"dtype":str(x.dtype),
                            "buffer_identity":bool(x.uop.has_buffer_identity()),
                            "precompiled_output_identity":bool(x.uop.has_precompiled_output_identity())})
    return original_marker(norm,x,promoted)
  model_module._decode_reduce_output_rmsnorm=observed_marker
  ids=(tokenizer.prefix() if hasattr(tokenizer,"prefix") else [])+tokenizer.encode("the quick brown fox jumps. "*800)
  reset_reduce_output_trace()
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  with Context(REDUCE_OUTPUT_TRACE=1,
               CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=int(args.typed_semantic_producer),
               CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=int(args.typed_semantic_producer)):
    census=capture_decode_graph(model,_make_prompt(ids,args.depth),args.chunk_size,3).to_dict()
  names=[record.get("program_name","") for record in census["records"]]
  census["capture"]={"phase":"decode","fixed_depth":args.depth,"jit":"rollout_greedy_jit_flash",
                     "callify_owned_redirect":int(args.typed_semantic_producer),
                     "typed_semantic_input_producer":int(args.typed_semantic_producer)}
  census["reduce_output"]={"count":sum("reduce_output_rmsnorm" in name for name in names),
                            "program_names":sorted({name for name in names if "reduce_output_rmsnorm" in name}),
                            "total_calls":len(names), "marker_input_count":len(marker_inputs),
                            "marker_input_histogram":dict(collections.Counter(json.dumps(row,sort_keys=True) for row in marker_inputs)),
                            "marker_input_examples":marker_inputs[:16],
                            "stage_trace":reduce_output_trace_snapshot()}
  out=pathlib.Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
  out.write_text(json.dumps(census,indent=2,sort_keys=True)+"\n")
  print(json.dumps(census["reduce_output"],sort_keys=True))
  return 0

if __name__ == "__main__": raise SystemExit(main())
