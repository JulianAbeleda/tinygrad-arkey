#!/usr/bin/env python3
"""Reverse production wall for a static public persisting descriptor on Flash K/V loads."""
from __future__ import annotations
import argparse,json,os,pathlib,statistics,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from extra.llm_research.decode.nv_flash_llama_vec_wide_qualification import MODEL,LOCK,PYTHON

def child(arm,depth,count,max_context,reps,out):
  os.environ.update(DEV="NV",PROFILE="0",JIT_BATCH_SIZE="33")
  if arm.startswith("candidate"):os.environ["NV_FLASH_KV_STATIC_PERSISTING"]=os.environ.get("NV_FLASH_KV_PERSIST_NUMERATOR","15")
  else:os.environ.pop("NV_FLASH_KV_STATIC_PERSISTING",None)
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  dev=Device["NV"];model=_load(MODEL,max_context);model._decode_direct_greedy_promoted=True;model._decode_feedback_pingpong_promoted=True
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
  try:settled=_settled_continuous_windows(gen,dev,count,reps)
  finally:gen.close()
  r={"schema":"tinygrad.nv_flash_kv_static_persist_wall.v1","arm":arm,"settled":settled};out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n");return r

def launch(name,root,a):
  out=root/f"{name}.json";cmd=["timeout","1800","flock","-w","600",LOCK,str(PYTHON),str(pathlib.Path(__file__).resolve()),"--arm",name,"--depth",str(a.depth),"--count",str(a.count),"--max-context",str(a.max_context),"--reps",str(a.reps),"--out",str(out)]
  env={**os.environ,"PYTHONPATH":str(ROOT)};env.pop("NV_FLASH_KV_STATIC_PERSISTING",None);run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
  if run.returncode:raise RuntimeError(f"{name} failed: {run.stderr[-6000:]}")
  return json.loads(out.read_text())

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=("control_a","candidate","control_c"));ap.add_argument("--depth",type=int,default=512);ap.add_argument("--count",type=int,default=16);ap.add_argument("--max-context",type=int,default=1024);ap.add_argument("--reps",type=int,default=7);ap.add_argument("--out",type=pathlib.Path,required=True);a=ap.parse_args()
  if a.arm:r=child(a.arm,a.depth,a.count,a.max_context,a.reps,a.out)
  else:
    root=pathlib.Path(str(a.out).removesuffix(".json"));root.mkdir(parents=True,exist_ok=True);arms=[launch(x,root,a) for x in ("control_a","candidate","control_c")];walls=[x["settled"]["median_ms_per_token"]*1000 for x in arms];ctl=statistics.median((walls[0],walls[2]));cand=walls[1];hashes={x["settled"]["token_stream_hash"] for x in arms}
    r={"schema":"tinygrad.nv_flash_kv_static_persist_wall.v1","mode":"reverse-bracket","walls_us_per_token":{"control_a":walls[0],"candidate":cand,"control_c":walls[2],"control":ctl,"candidate_delta":cand-ctl},"tokens_per_second":{"control":1e6/ctl,"candidate":1e6/cand,"candidate_delta":1e6/cand-1e6/ctl},"all_token_hashes_equal":len(hashes)==1,"verdict":"WALL_PASS" if len(hashes)==1 and cand<min(walls[0],walls[2]) else "NO_GO_WALL","arms":arms};a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
  print(json.dumps(r,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
