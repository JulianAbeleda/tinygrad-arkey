#!/usr/bin/env python3
"""Measure the installed dense decode route without changing model policies.

The backend is selected by ``DEV``. This is intentionally neutral measurement
tooling for native-NV versus CUDA-graph route qualification.
"""
from __future__ import annotations
import argparse,json,os,pathlib,subprocess,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[3]))
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import DEFAULT_MODEL,_load,_prompt
from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows

def main():
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--model',default=DEFAULT_MODEL); ap.add_argument('--depth',type=int,default=512); ap.add_argument('--count',type=int,default=24); ap.add_argument('--reps',type=int,default=4); ap.add_argument('--max-context',type=int,default=1024); ap.add_argument('--single-graph-live-bands',action='store_true'); ap.add_argument('--eager-prefill',action='store_true'); ap.add_argument('--generic-prefill',action='store_true'); ap.add_argument('--out',type=pathlib.Path,required=True); a=ap.parse_args()
 from tinygrad import Device
 model=_load(a.model,a.max_context)
 if a.single_graph_live_bands:
  model._decode_direct_greedy_promoted=False; model._decode_feedback_pingpong_promoted=False
 if a.generic_prefill: object.__setattr__(model.config,'prefill_v2',False)
 # Bound production prewarming to the actual measured request.  This keeps a
 # fixed-depth decode authority from compiling unrelated future context bands.
 expected_output_tokens=7+a.count*a.reps+int(a.eager_prefill)
 active_horizon_lease=bool(getattr(model,'_flash_decode_active_horizon_lease',False))
 if a.eager_prefill: model._flash_decode_active_horizon_lease=False
 gen=model.generate(_prompt(a.model,a.depth),chunk_size=32,temperature=0.0,
                    expected_output_tokens=expected_output_tokens)
 if a.eager_prefill:
  from tinygrad.helpers import Context
  try:
   with Context(JIT=0): next(gen)
  finally: model._flash_decode_active_horizon_lease=active_horizon_lease
  # The following discarded decode tokens are the normal lazy TinyJit
  # warmup/capture sequence for the one band this request can reach.
  model._flash_decode_active_horizon_prewarmed=active_horizon_lease
 try: row=_settled_continuous_windows(gen,Device[Device.DEFAULT],a.count,a.reps)
 finally: gen.close()
 out={'schema':'tinygrad.dense_route_endpoint.v1','commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'DEV':os.environ.get('DEV'),'CUDA_GRAPH_STREAMS':os.environ.get('CUDA_GRAPH_STREAMS'),'depth':a.depth,'count':a.count,'reps':a.reps,'max_context':a.max_context,'expected_output_tokens':expected_output_tokens,'single_graph_live_bands':a.single_graph_live_bands,'eager_prefill':a.eager_prefill,'generic_prefill':a.generic_prefill,**row}
 a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
