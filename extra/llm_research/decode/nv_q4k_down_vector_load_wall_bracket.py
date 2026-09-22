#!/usr/bin/env python3
"""Reverse production wall bracket for bit-exact Q4_K FFN-down vector loads."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.qk_norm_rope_wall_bracket import MODEL, LOCK, _gpu_state


def _set_load_style(model, vector:bool) -> list[int]:
  from tinygrad.llm.q4k_ffn_down_mmvq import Q4KFFNDownMMVQAdmission
  from tinygrad.llm.qk_primitives import Q4KPrimitiveLinear
  indices=[]
  for index, block in enumerate(model.blk):
    ffn=getattr(block,"ffn_down",None)
    if not isinstance(ffn,Q4KPrimitiveLinear) or getattr(ffn,"route_role","") != "ffn_down": continue
    if (getattr(ffn,"out_features",None),getattr(ffn,"in_features",None)) != (4096,12288): continue
    current=getattr(ffn,"_q4k_ffn_down_mmvq_admission",None)
    if current is None or not getattr(current,"fp16_fma",False):
      raise RuntimeError(f"block {index} lacks the promoted fp16 Q4-down admission")
    ffn._q4k_ffn_down_mmvq_admission=Q4KFFNDownMMVQAdmission(index,fp16_fma=True,vector_loads=vector)
    indices.append(index)
  if len(indices) != 18: raise RuntimeError(f"expected 18 Q4-down blocks, got {indices}")
  return indices


def timing_child(depth:int,count:int,max_context:int,reps:int,vector:bool,out:pathlib.Path) -> dict:
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model=_load(MODEL,max_context); indices=_set_load_style(model,vector)
  model._decode_direct_greedy_promoted=False; model._decode_feedback_pingpong_promoted=False
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
  try: settled=_settled_continuous_windows(gen,Device[Device.DEFAULT],count,reps)
  finally: gen.close()
  result={"schema":"tinygrad.nv_q4k_down_vector_load_wall_bracket.v1",
    "load_style":"vector" if vector else "scalar","q4_down_blocks":indices,
    "gpu_state":_gpu_state(),"depth":depth,"count":count,"reps":reps,
    "settled_continuous":True,**settled}
  out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  return result


def bracket(depth:int,count:int,max_context:int,reps:int,out:pathlib.Path) -> dict:
  root=pathlib.Path(str(out).removesuffix(".json")); root.mkdir(parents=True,exist_ok=True)
  def child(vector:bool,label:str) -> dict:
    arm_out=root/f"{label}.json"
    cmd=["timeout","1800","flock","-w","600",LOCK,sys.executable,str(pathlib.Path(__file__).resolve()),
      "--mode","timing-child","--depth",str(depth),"--count",str(count),"--max-context",str(max_context),
      "--reps",str(reps),"--out",str(arm_out)]
    if vector: cmd.append("--vector")
    run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
      env={**os.environ,"PYTHONPATH":str(ROOT),"DEV":"NV"})
    if run.returncode: raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-4000:]}")
    return json.loads(arm_out.read_text())
  arms=[child(False,"control_a"),child(True,"candidate"),child(False,"control_c")]
  midpoint=statistics.median((arms[0]["median_ms_per_token"],arms[2]["median_ms_per_token"]))
  candidate=arms[1]["median_ms_per_token"]; hashes={x["token_stream_hash"] for x in arms}
  result={"schema":"tinygrad.nv_q4k_down_vector_load_wall_bracket.v1","mode":"reverse-bracket",
    "depth":depth,"count":count,"reps":reps,"control_a_ms_per_token":arms[0]["median_ms_per_token"],
    "candidate_ms_per_token":candidate,"control_c_ms_per_token":arms[2]["median_ms_per_token"],
    "control_midpoint_ms_per_token":midpoint,"recovery_us_per_token":(midpoint-candidate)*1000,
    "candidate_speedup_pct":(midpoint/candidate-1)*100,"all_token_hashes_equal":len(hashes)==1,
    "token_stream_hash":sorted(hashes)[0] if len(hashes)==1 else sorted(hashes),
    "verdict":"WALL_PASS" if len(hashes)==1 and candidate<min(arms[0]["median_ms_per_token"],arms[2]["median_ms_per_token"])
      else "NO_GO_WALL","arms":arms}
  out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("timing","timing-child"),default="timing")
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--count",type=int,default=32)
  ap.add_argument("--max-context",type=int,default=1024); ap.add_argument("--reps",type=int,default=9)
  ap.add_argument("--vector",action="store_true"); ap.add_argument("--out",type=pathlib.Path,required=True); args=ap.parse_args()
  result=(timing_child(args.depth,args.count,args.max_context,args.reps,args.vector,args.out)
    if args.mode=="timing-child" else bracket(args.depth,args.count,args.max_context,args.reps,args.out))
  print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
