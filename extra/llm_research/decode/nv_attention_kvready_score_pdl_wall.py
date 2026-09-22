#!/usr/bin/env python3
"""Unprofiled reverse wall bracket for KV-ready -> flash-score split-phase PDL."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from extra.llm_research.decode.qk_norm_rope_wall_bracket import MODEL,LOCK,_gpu_state

PRODUCER="reduce_output_rmsnorm_rope_kv_cache_8_128"
CONSUMER="flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128"

def run_child(candidate:bool,depth:int,count:int,max_context:int,reps:int,out:pathlib.Path)->dict:
  if candidate:
    os.environ["NV_PDL_PRODUCER_PROGRAMS"]=PRODUCER
    os.environ["NV_PDL_CONSUMER_PROGRAMS"]=CONSUMER
    os.environ["NV_PDL_TRIGGER_POSITION"]="start"
  else:
    for key in ("NV_PDL_PRODUCER_PROGRAMS","NV_PDL_CONSUMER_PROGRAMS","NV_PDL_TRIGGER_POSITION"):os.environ.pop(key,None)
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model=_load(MODEL,max_context);model._decode_direct_greedy_promoted=True;model._decode_feedback_pingpong_promoted=True
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
  try:settled=_settled_continuous_windows(gen,Device[Device.DEFAULT],count,reps)
  finally:gen.close()
  result={"schema":"tinygrad.nv_attention_kvready_score_pdl_wall.v1","arm":"candidate" if candidate else "control",
    "producer":PRODUCER,"consumer":CONSUMER,"trigger_position":"start" if candidate else None,
    "depth":depth,"count":count,"reps":reps,"gpu_state":_gpu_state(),**settled}
  out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return result

def bracket(depth:int,count:int,max_context:int,reps:int,out:pathlib.Path)->dict:
  root=pathlib.Path(str(out).removesuffix(".json"));root.mkdir(parents=True,exist_ok=True)
  def child(candidate:bool,label:str)->dict:
    arm=root/f"{label}.json";cmd=["timeout","1800","flock","-w","600",LOCK,sys.executable,str(pathlib.Path(__file__).resolve()),
      "--mode","child","--depth",str(depth),"--count",str(count),"--max-context",str(max_context),"--reps",str(reps),"--out",str(arm)]
    if candidate:cmd.append("--candidate")
    env={k:v for k,v in os.environ.items() if k not in ("PROFILE","HCQ_GRAPH_PROFILE_JSON","NV_PDL_PRODUCER_PROGRAMS","NV_PDL_CONSUMER_PROGRAMS","NV_PDL_TRIGGER_POSITION")}
    run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**env,"PYTHONPATH":str(ROOT),"DEV":"NV"})
    if run.returncode:raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-5000:]}")
    return json.loads(arm.read_text())
  arms=[child(False,"control_a"),child(True,"candidate"),child(False,"control_c")]
  a,b,c=[float(x["median_ms_per_token"]) for x in arms];mid=statistics.median((a,c));hashes={x["token_stream_hash"] for x in arms}
  result={"schema":"tinygrad.nv_attention_kvready_score_pdl_wall.v1","mode":"reverse-bracket","depth":depth,"count":count,"reps":reps,
    "producer":PRODUCER,"consumer":CONSUMER,"trigger_position":"start","control_a_ms_per_token":a,"candidate_ms_per_token":b,
    "control_c_ms_per_token":c,"control_midpoint_ms_per_token":mid,"midpoint_recovery_us_per_token":(mid-b)*1000,
    "conservative_recovery_us_per_token":(min(a,c)-b)*1000,"all_token_hashes_equal":len(hashes)==1,
    "token_stream_hash":sorted(hashes)[0] if len(hashes)==1 else sorted(hashes),
    "verdict":"WALL_PASS" if len(hashes)==1 and b<min(a,c) else "NO_GO_WALL","arms":arms}
  out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return result

def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=("bracket","child"),default="bracket");ap.add_argument("--candidate",action="store_true")
  ap.add_argument("--depth",type=int,default=512);ap.add_argument("--count",type=int,default=32);ap.add_argument("--max-context",type=int,default=1024)
  ap.add_argument("--reps",type=int,default=9);ap.add_argument("--out",type=pathlib.Path,required=True);a=ap.parse_args()
  result=run_child(a.candidate,a.depth,a.count,a.max_context,a.reps,a.out) if a.mode=="child" else bracket(a.depth,a.count,a.max_context,a.reps,a.out)
  print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
