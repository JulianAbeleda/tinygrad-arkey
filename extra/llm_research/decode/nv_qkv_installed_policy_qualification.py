#!/usr/bin/env python3
"""Fresh-process rollback and composed wall gate for installed QKV policies."""
from __future__ import annotations
import argparse,json,os,pathlib,statistics,subprocess,sys

ROOT=pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import MODEL,LOCK,PYTHON,_gpu_state

DISABLE={"s44":"TINYGRAD_SHARED_Q8_Q4Q4_QKV_FULL_DISABLE","o44":"TINYGRAD_Q4K_Q4Q4_QKV_FULL_DISABLE"}

def _set_arm(candidate:bool,variant:str):
  keys=DISABLE if variant=="composed" else {variant:DISABLE[variant]}
  for key in keys.values():
    if candidate: os.environ.pop(key,None)
    else: os.environ[key]="1"

def child(candidate:bool,variant:str,depth:int,count:int,reps:int,max_context:int,out:pathlib.Path)->dict:
  _set_arm(candidate,variant); os.environ["DEV"]="NV"
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  from tinygrad.llm.q4k_kv_pair import Q4KQKVAdmission
  from tinygrad.llm.shared_q8_attention import SharedQ8AttentionAdmission
  model=_load(MODEL,max_context); model._decode_direct_greedy_promoted=False; model._decode_feedback_pingpong_promoted=False
  census={"s44":[i for i,b in enumerate(model.blk) if isinstance((a:=getattr(b,"_shared_q8_attention_admission",None)),SharedQ8AttentionAdmission) and a.q4_qkv_triple_output],
    "o44":[i for i,b in enumerate(model.blk) if isinstance(getattr(b,"_q4k_qkv_admission",None),Q4KQKVAdmission)],
    "producer_sink":sum(hasattr(b,"_producer_kv_cache_sink_admission") for b in model.blk)}
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
  try:settled=_settled_continuous_windows(gen,Device["NV"],count,reps)
  finally:gen.close()
  result={"schema":"tinygrad.nv_qkv_installed_policy_qualification.v1","mode":"timing-child","variant":variant,
    "arm":"candidate" if candidate else "control","depth":depth,"count":count,"reps":reps,"max_context":max_context,
    "gpu_state":_gpu_state(),"census":census,**settled}
  out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return result

def _run(candidate:bool,label:str,args,root:pathlib.Path)->dict:
  out=root/f"{label}.json";cmd=["timeout","1800","flock","-w","600",LOCK,str(PYTHON),str(pathlib.Path(__file__).resolve()),
    "--mode","timing-child","--variant",args.variant,"--depth",str(args.depth),"--count",str(args.count),
    "--reps",str(args.reps),"--max-context",str(args.max_context),"--out",str(out)]
  if candidate:cmd.append("--candidate")
  env={**os.environ,"PYTHONPATH":str(ROOT),"DEV":"NV"}
  keys=DISABLE if args.variant=="composed" else {args.variant:DISABLE[args.variant]}
  for key in keys.values():
    if candidate:env.pop(key,None)
    else:env[key]="1"
  run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
  if run.returncode:raise RuntimeError(f"{label} rc={run.returncode}: {run.stderr[-5000:]}")
  return json.loads(out.read_text())

def driver(args)->dict:
  root=args.out.parent/(args.out.stem+"_arms");root.mkdir(parents=True,exist_ok=True)
  arms=[_run(False,"control_a",args,root),_run(True,"candidate",args,root),_run(False,"control_c",args,root)]
  midpoint=statistics.median((arms[0]["median_ms_per_token"],arms[2]["median_ms_per_token"]))
  cand=arms[1]["median_ms_per_token"];hashes={a["token_stream_hash"] for a in arms}
  result={"schema":"tinygrad.nv_qkv_installed_policy_qualification.v1","mode":"timing","variant":args.variant,
    "depth":args.depth,"count":args.count,"reps":args.reps,"control_a_ms_per_token":arms[0]["median_ms_per_token"],
    "candidate_ms_per_token":cand,"control_c_ms_per_token":arms[2]["median_ms_per_token"],
    "control_midpoint_ms_per_token":midpoint,"recovery_us_per_token":(midpoint-cand)*1000,
    "all_token_hashes_equal":len(hashes)==1,"token_stream_hash":sorted(hashes)[0] if len(hashes)==1 else sorted(hashes),
    "verdict":"WALL_PASS" if len(hashes)==1 and cand<min(arms[0]["median_ms_per_token"],arms[2]["median_ms_per_token"]) else "NO_GO_WALL",
    "arms":arms}
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return result

def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=("timing","timing-child"),default="timing")
  ap.add_argument("--candidate",action="store_true");ap.add_argument("--variant",choices=("s44","o44","composed"),required=True)
  ap.add_argument("--depth",type=int,default=512);ap.add_argument("--count",type=int,default=32);ap.add_argument("--reps",type=int,default=7)
  ap.add_argument("--max-context",type=int,default=1024);ap.add_argument("--out",type=pathlib.Path,required=True);args=ap.parse_args()
  result=child(args.candidate,args.variant,args.depth,args.count,args.reps,args.max_context,args.out) if args.mode=="timing-child" else driver(args)
  print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
