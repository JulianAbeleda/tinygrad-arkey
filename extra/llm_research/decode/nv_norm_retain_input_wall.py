#!/usr/bin/env python3
"""Reverse production bracket for native 4096 RMSNorm register retention."""
from __future__ import annotations
import argparse,json,os,pathlib,statistics,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from extra.llm_research.decode.qk_norm_rope_wall_bracket import MODEL,LOCK,_gpu_state

def child(control:bool,depth:int,count:int,max_context:int,reps:int,out:pathlib.Path):
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model=_load(MODEL,max_context);model._decode_direct_greedy_promoted=True;model._decode_feedback_pingpong_promoted=True
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
  try: row=_settled_continuous_windows(gen,Device[Device.DEFAULT],count,reps)
  finally: gen.close()
  result={"schema":"tinygrad.nv_norm_retain_input_wall.v1","control":control,"gpu_state":_gpu_state(),**row}
  out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

def bracket(depth:int,count:int,max_context:int,reps:int,out:pathlib.Path):
  root=pathlib.Path(str(out).removesuffix(".json"));root.mkdir(parents=True,exist_ok=True)
  def arm(control:bool,label:str):
    dst=root/f"{label}.json";env={**os.environ,"PYTHONPATH":str(ROOT),"DEV":"NV"}
    if control:env["TINYGRAD_NATIVE_NORM_RETAIN_INPUT_DISABLE"]="1"
    else:env.pop("TINYGRAD_NATIVE_NORM_RETAIN_INPUT_DISABLE",None)
    cmd=["timeout","1800","flock","-w","600",LOCK,sys.executable,__file__,"--mode","child","--depth",str(depth),"--count",str(count),"--max-context",str(max_context),"--reps",str(reps),"--out",str(dst)]+(["--control"] if control else [])
    r=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if r.returncode:raise RuntimeError(f"{label} rc={r.returncode}: {r.stderr[-4000:]}")
    return json.loads(dst.read_text())
  a,b,c=arm(True,"control_a"),arm(False,"candidate"),arm(True,"control_c")
  mid=(a["median_ms_per_token"]+c["median_ms_per_token"])/2; cand=b["median_ms_per_token"]
  result={"schema":"tinygrad.nv_norm_retain_input_wall.v1","control_a_ms":a["median_ms_per_token"],"candidate_ms":cand,"control_c_ms":c["median_ms_per_token"],"control_midpoint_ms":mid,"recovery_us":(mid-cand)*1000,"all_hashes_equal":len({x["token_stream_hash"] for x in (a,b,c)})==1,"arms":[a,b,c]}
  result["verdict"]="WALL_PASS" if result["all_hashes_equal"] and cand<min(a["median_ms_per_token"],c["median_ms_per_token"]) else "NO_GO_WALL"
  out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__":
  p=argparse.ArgumentParser();p.add_argument("--mode",choices=("timing","child"),default="timing");p.add_argument("--control",action="store_true");p.add_argument("--depth",type=int,default=512);p.add_argument("--count",type=int,default=16);p.add_argument("--max-context",type=int,default=1024);p.add_argument("--reps",type=int,default=9);p.add_argument("--out",type=pathlib.Path,required=True);a=p.parse_args()
  child(a.control,a.depth,a.count,a.max_context,a.reps,a.out) if a.mode=="child" else bracket(a.depth,a.count,a.max_context,a.reps,a.out)
